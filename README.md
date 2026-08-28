# Telco Customer Churn Analysis

Exploratory analysis and churn prediction for a telecom company's customer base. The goal is to identify which factors are most associated with customers leaving, and build a baseline model that predicts churn from account and usage data.

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, 7,043 customers, 21 columns).

## What's in here

- Data cleaning (the dataset stores `TotalCharges` as text with blank values for brand-new customers)
- Feature engineering: tenure buckets, count of add-on services, average spend per month of tenure
- EDA covering churn rate by contract type, tenure, internet service, payment method, and number of add-on services
- A correlation heatmap of the numeric features
- Model comparison between Logistic Regression and Random Forest using 5-fold cross-validation, followed by a full evaluation of the better model (ROC curve, confusion matrix, feature importance)
- A `.py` script and an equivalent `.ipynb` notebook, so you can run it as a script or explore it interactively

## Findings

| Factor | Effect on churn |
|---|---|
| Month-to-month contract | Substantially higher churn than one- or two-year contracts |
| Low tenure (0-12 months) | New customers are the highest-risk group |
| Fiber optic internet | Associated with higher churn than DSL |
| Electronic check payment | Higher churn than automatic bank/card payments |
| More add-on services (tech support, online security, etc.) | Associated with lower churn |

These come from `churn_summary.csv` after a run on the full dataset — numbers will differ slightly from run to run since the train/test split is stratified but not deterministic across dataset versions.

## Project structure

```
.
├── churn_analysis.py              # main script
├── telco_churn_analysis.ipynb     # same analysis as a notebook
├── requirements.txt
├── LICENSE
├── figures/                       # generated on run, not tracked in git
│   ├── 01_churn_overview.png
│   ├── 02_tenure_distribution.png
│   ├── 03_monthly_charges_boxplot.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_services_impact.png
│   ├── 06_model_comparison.png
│   ├── 07_final_model_results.png
│   └── 08_confusion_matrix.png
└── output/
    └── churn_summary.csv          # generated on run, not tracked in git
```

## Running it

```bash
pip install -r requirements.txt
python churn_analysis.py
```

or open `telco_churn_analysis.ipynb` in Jupyter / Colab and run all cells.

The script pulls the dataset via `kagglehub`, which requires a Kaggle account and API token (see the [kagglehub docs](https://github.com/Kaggle/kagglehub) for setup). If `kagglehub` isn't available or you're not authenticated, the script automatically falls back to a local `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same directory — download that manually from the [Kaggle page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) if you'd rather skip the API setup.

## Notes on the model

This is a baseline comparison (Logistic Regression vs. Random Forest with default-ish hyperparameters), not a tuned production model. Reasonable next steps: hyperparameter search, gradient boosting (XGBoost/LightGBM), and testing on a more recent or larger churn dataset.

## License

MIT — see [LICENSE](LICENSE).
