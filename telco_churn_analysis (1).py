import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Ρυθμίσεις εμφάνισης
# --------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titleweight"] = "bold"

PALETTE = {"No": "#2ecc71", "Yes": "#e74c3c"}
ACCENT = "#3498db"

FIGURES_DIR = "figures"
OUTPUT_DIR = "output"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================================
# 1. ΦΟΡΤΩΣΗ & ΚΑΘΑΡΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ
# ==========================================================================
DATASET_HANDLE = "blastchar/telco-customer-churn"
DATASET_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
LOCAL_CSV_FALLBACK = DATASET_FILE  # αν υπάρχει ήδη τοπικά, το χρησιμοποιούμε


def load_data() -> pd.DataFrame:
    """
    Φορτώνει το Telco Churn dataset μέσω kagglehub (νεότερο PANDAS adapter API).
    Αν δεν υπάρχει σύνδεση/λογαριασμός Kaggle, κάνει fallback σε τοπικό CSV
    (π.χ. αν το έχεις κατεβάσει χειροκίνητα στον ίδιο φάκελο με το script).
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            DATASET_HANDLE,
            DATASET_FILE,
        )
        print(f"✔ Δεδομένα φορτώθηκαν από Kaggle (kagglehub): "
              f"{df.shape[0]} πελάτες, {df.shape[1]} στήλες")
        return df
    except Exception as e:
        print(f"⚠ Αποτυχία φόρτωσης μέσω kagglehub ({e}). "
              f"Δοκιμή τοπικού αρχείου '{LOCAL_CSV_FALLBACK}'...")
        df = pd.read_csv(LOCAL_CSV_FALLBACK)
        print(f"✔ Δεδομένα φορτώθηκαν τοπικά: {df.shape[0]} πελάτες, {df.shape[1]} στήλες")
        return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Καθαρίζει τιμές και τύπους δεδομένων."""
    df = df.copy()

    # Το TotalCharges περιέχει κενά strings σε νέους πελάτες (tenure=0)
    n_blank = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    print(f"✔ TotalCharges: {n_blank} κενές τιμές εντοπίστηκαν, "
          f"{n_missing} μετατράπηκαν σε NaN")

    # Γεμίζουμε τις ελλείπουσες τιμές με 0 (πελάτες με tenure=0 δεν έχουν χρεωθεί ακόμα)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Καθαρισμός δυαδικού SeniorCitizen -> Yes/No για συνέπεια με τις άλλες στήλες
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Δημιουργεί επιπλέον χρήσιμα χαρακτηριστικά."""
    df = df.copy()
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    )

    # Πλήθος πρόσθετων υπηρεσιών που έχει ο πελάτης (proxy για "engagement")
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)

    # Μέσο μηνιαίο κόστος ανά μήνα παραμονής (avoid div/0)
    df["ChargesPerTenure"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    return df


# ==========================================================================
# 2. ΠΕΡΙΓΡΑΦΙΚΗ ΣΤΑΤΙΣΤΙΚΗ
# ==========================================================================
def print_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("ΒΑΣΙΚΑ ΣΤΑΤΙΣΤΙΚΑ")
    print("=" * 60)
    print(f"Συνολικοί πελάτες:      {len(df):,}")
    print(f"Churn Rate:             {df['Churn_bin'].mean():.2%}")
    print(f"Μέσο tenure:            {df['tenure'].mean():.1f} μήνες")
    print(f"Μέσα μηνιαία έσοδα:     ${df['MonthlyCharges'].mean():.2f}")
    print(f"Μέσο TotalCharges:      ${df['TotalCharges'].mean():.2f}")
    print(f"Μέσος αρ. υπηρεσιών:    {df['NumServices'].mean():.1f}")


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Συγκεντρωτικός πίνακας churn rate ανά βασική διάσταση -> CSV."""
    rows = []
    dims = ["Contract", "PaymentMethod", "InternetService", "TenureGroup",
            "SeniorCitizen", "Dependents", "Partner", "PaperlessBilling"]
    for dim in dims:
        grp = df.groupby(dim, observed=True)["Churn_bin"].agg(["mean", "count"])
        for level, row in grp.iterrows():
            rows.append({
                "Διάσταση": dim,
                "Κατηγορία": level,
                "ChurnRate": round(row["mean"], 4),
                "ΠλήθοςΠελατών": int(row["count"]),
            })
    summary = pd.DataFrame(rows).sort_values("ChurnRate", ascending=False)
    out_path = os.path.join(OUTPUT_DIR, "churn_summary.csv")
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✔ Summary table αποθηκεύτηκε: {out_path}")
    return summary


# ==========================================================================
# 3. ΓΡΑΦΗΜΑΤΑ
# ==========================================================================
def _annotate_bars(ax, fmt="{:.1%}"):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")


def plot_churn_overview(df: pd.DataFrame) -> None:
    """Dashboard 2x2: churn ανά contract, tenure group, internet service, payment method."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    specs = [
        ("Contract", ["Month-to-month", "One year", "Two year"], axes[0, 0], "Τύπος Συμβολαίου"),
        ("TenureGroup", ["0-12", "13-24", "25-48", "49-72"], axes[0, 1], "Ομάδα Tenure"),
        ("InternetService", ["DSL", "Fiber optic", "No"], axes[1, 0], "Τύπος Internet"),
        ("PaymentMethod", None, axes[1, 1], "Μέθοδος Πληρωμής"),
    ]

    for col, order, ax, title in specs:
        if order is None:
            order = df.groupby(col)["Churn_bin"].mean().sort_values(ascending=False).index
        sns.barplot(data=df, x=col, y="Churn_bin", order=order, errorbar=None,
                    color=ACCENT, ax=ax)
        ax.set_title(f"Churn Rate ανά {title}", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("Churn Rate")
        ax.tick_params(axis="x", rotation=20 if col == "PaymentMethod" else 0)
        _annotate_bars(ax)

    fig.suptitle("Επισκόπηση Churn Rate ανά Βασικές Διαστάσεις", fontsize=17, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "01_churn_overview_dashboard.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_tenure_distribution(df: pd.DataFrame) -> None:
    """Κατανομή tenure ανά Churn (histogram + KDE)."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
                 palette=PALETTE, bins=30, edgecolor="white")
    plt.title("Κατανομή Tenure ανά Churn", fontsize=16, pad=15)
    plt.xlabel("Tenure (μήνες)")
    plt.ylabel("Πλήθος Πελατών")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_tenure_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_monthly_charges_vs_churn(df: pd.DataFrame) -> None:
    """Boxplot MonthlyCharges ανά Churn."""
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=PALETTE)
    ax.set_title("Μηνιαία Χρέωση ανά Churn", fontsize=16, pad=15)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Μηνιαία Χρέωση ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_monthly_charges_boxplot.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap συσχέτισης αριθμητικών μεταβλητών."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                "NumServices", "ChargesPerTenure", "Churn_bin"]
    corr = df[num_cols].corr()

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Συσχέτιση Αριθμητικών Μεταβλητών", fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_correlation_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_services_impact(df: pd.DataFrame) -> None:
    """Churn rate ανά πλήθος πρόσθετων υπηρεσιών."""
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=df, x="NumServices", y="Churn_bin", errorbar=None, color="#9b59b6")
    ax.set_title("Churn Rate ανά Πλήθος Πρόσθετων Υπηρεσιών", fontsize=16, pad=15)
    ax.set_xlabel("Αριθμός Πρόσθετων Υπηρεσιών")
    ax.set_ylabel("Churn Rate")
    _annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_services_impact.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_results(model, X_test, y_test, feature_names) -> None:
    """ROC curve + feature importance (συντελεστές logistic regression)."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ROC
    axes[0].plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve — Πρόβλεψη Churn", fontsize=14)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # Feature importance (top 10 apόλυτοι συντελεστές)
    coefs = pd.Series(model.coef_[0], index=feature_names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)[::-1]
    colors = [PALETTE["Yes"] if v > 0 else PALETTE["No"] for v in top.values]
    axes[1].barh(top.index, top.values, color=colors)
    axes[1].set_title("Top 10 Παράγοντες Επιρροής στο Churn", fontsize=14)
    axes[1].set_xlabel("Συντελεστής (θετικός = αυξάνει το churn)")
    axes[1].axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "06_model_results.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix", fontsize=15, pad=12)
    plt.xlabel("Πρόβλεψη")
    plt.ylabel("Πραγματικό")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "07_confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ==========================================================================
# 4. ΠΡΟΒΛΕΠΤΙΚΟ ΜΟΝΤΕΛΟ
# ==========================================================================
def train_churn_model(df: pd.DataFrame):
    """Εκπαιδεύει Logistic Regression για πρόβλεψη churn και τυπώνει metrics."""
    features = df.drop(columns=["customerID", "Churn", "Churn_bin", "TenureGroup"])
    features = pd.get_dummies(features, drop_first=True)
    target = df["Churn_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    print("\n" + "=" * 60)
    print("ΑΠΟΤΕΛΕΣΜΑΤΑ ΜΟΝΤΕΛΟΥ (Logistic Regression)")
    print("=" * 60)
    print(f"AUC-ROC: {auc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    plot_model_results(model, X_test_scaled, y_test, features.columns)
    plot_confusion_matrix(y_test, y_pred)

    return model


# ==========================================================================
# 5. MAIN
# ==========================================================================
def main():
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    print_overview(df)
    build_summary_table(df)

    print("\n" + "=" * 60)
    print("ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ")
    print("=" * 60)
    plot_churn_overview(df)
    plot_tenure_distribution(df)
    plot_monthly_charges_vs_churn(df)
    plot_correlation_heatmap(df)
    plot_services_impact(df)

    train_churn_model(df)

    print(f"\n✔ Όλα τα γραφήματα αποθηκεύτηκαν στον φάκελο ./{FIGURES_DIR}/")
    print(f"✔ Το summary table αποθηκεύτηκε στον φάκελο ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
