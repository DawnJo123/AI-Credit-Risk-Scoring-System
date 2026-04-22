# AI Credit Risk Scoring with Explainable AI + LLM Layer

A comprehensive machine learning system for predicting loan default risk with explicit focus on **explainability** and **human-readable explanations**. This project combines predictive modeling, SHAP-based interpretability, and LLM-powered natural language explanations via OpenRouter API.

## Overview

This project addresses a critical need in banking and financial services: making ML-driven credit decisions **transparent and explainable**. The system:

- Predicts loan default probability using XGBoost
- Provides global and local explanations using SHAP
- Generates natural language explanations via LLM integration (OpenRouter API)
- Categorizes applicants into risk tiers with optimized decision thresholds

**Dataset:** LendingClub Loan Data (2007-2018) - 1.34 million loan records

## Project Structure

```
AI Credit Risk Scoring + Explainable AI + LLM Layer/
│
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory Data Analysis
│   ├── 02_feature_eng.ipynb             # Feature Engineering
│   ├── 03_modeling.ipynb                # Model Training & Evaluation
│   ├── 04_explainability.ipynb          # SHAP Explainability (Global + Local)
│   └── 05_llm_layer.ipynb              # LLM Explanation Layer (OpenRouter)
│
├── models/
│   ├── xgboost_model.pkl                # Trained XGBoost classifier
│   ├── scaler.pkl                       # StandardScaler for features
│   ├── encoder.pkl                      # Categorical encoder
│   ├── feature_names.pkl                # Feature names list (44 features)
│   ├── threshold.pkl                    # Optimized decision threshold (0.4)
│   ├── shap_explainer.pkl               # Pre-computed SHAP TreeExplainer
│   └── cap_values.pkl                   # Outlier capping thresholds
│
├── .env.example                         # Environment variable template
├── .gitignore
├── requirements.txt                     # Python dependencies
├── LICENSE                              # Apache 2.0 License
└── README.md
```
## Notebook Pipeline

Run the notebooks in order — each builds on the previous:

| # | Notebook | Phase | What It Does |
|---|----------|-------|-------------|
| 1 | `01_eda.ipynb` | EDA | Missing values, outliers, distributions, correlation heatmaps, class imbalance analysis |
| 2 | `02_feature_eng.ipynb` | Feature Engineering | Creates `fico_avg`, `loan_to_income`, `installment_to_income`, `credit_per_account` |
| 3 | `03_modeling.ipynb` | Model Training | Trains Logistic Regression, Random Forest, XGBoost; threshold tuning; saves artifacts |
| 4 | `04_explainability.ipynb` | SHAP | Global feature importance, local waterfall plots, `explain_prediction()` function |
| 5 | `05_llm_layer.ipynb` | LLM Layer | Converts SHAP output → LLM prompt → plain English explanation via OpenRouter |

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 65.98% | 32.28% | 64.19% | 42.96% | 71.14% |
| Random Forest | 64.13% | 31.41% | 67.33% | 42.83% | 71.20% |
| **XGBoost (Selected)** | **64.94%** | **32.13%** | **68.04%** | **43.65%** | **72.12%** |

**Why XGBoost?**
- Highest ROC-AUC (0.7212) — best at distinguishing defaults from non-defaults
- High recall (68.04%) — catches most actual defaults
- Optimized threshold at 0.4 (instead of default 0.5) to favor catching risky loans

## Feature Engineering

44 features used for prediction:

**Original Features:** `loan_amnt`, `funded_amnt`, `int_rate`, `term_in_months`, `annual_inc`, `dti`, `emp_length_num`, `fico_range_low/high`, `delinq_2yrs`, `revol_util`, `revol_bal`, `open_acc`, `total_acc`, `pub_rec`, and one-hot encoded `home_ownership`, `purpose`, `verification_status`

**Engineered Features:**
| Feature | Formula | Correlation with Default |
|---------|---------|------------------------|
| `fico_avg` | (fico_low + fico_high) / 2 | -0.131 |
| `loan_to_income` | loan_amnt / annual_inc | +0.133 |
| `installment_to_income` | (installment * 12) / annual_inc | +0.117 |
| `credit_per_account` | revol_bal / open_acc | -0.032 |

## Explainability

### SHAP (Notebook 04)

**Global Explanations** — Which features matter most across all loans?
- SHAP summary plots and bar charts of mean absolute SHAP values

**Local Explanations** — Why was THIS specific loan flagged?
- SHAP waterfall plots showing feature-by-feature contribution to the prediction

### LLM Explanation Layer (Notebook 05)

The LLM layer converts model output + SHAP values into plain English via OpenRouter API:

```
Pipeline: Customer Data → XGBoost (probability) → SHAP (why?) → LLM (explain in English)
```

**Example Output:**
```
Input:  Prediction = High Risk (78%), Top factors: sub-grade, loan term, FICO score

Output: "This loan application has been assessed as Very High Risk with a 78% 
         probability of default. The primary concerns are the applicant's loan 
         sub-grade rating and the extended 60-month loan term, which significantly 
         increase repayment risk. The applicant's FICO score also falls below 
         preferred thresholds..."
```

**Key functions in the notebook:**
- `get_shap_explanation()` — Extracts risk/protective factors from SHAP values
- `build_explanation_prompt()` — Structures prediction + SHAP into an LLM prompt
- `generate_explanation()` — Calls OpenRouter API for natural language output
- `explain_loan_decision()` — Full end-to-end pipeline in one function

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "AI Credit Risk Scoring + Explainable AI + LLM Layer"
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up OpenRouter API key for the LLM layer:
```bash
cp .env.example .env
# Edit .env and add your key from https://openrouter.ai/keys
```

### Data Setup

The raw LendingClub dataset is not included in the repo due to size. To reproduce the full pipeline:

1. Download `accepted_2007_to_2018Q4.csv` from [Kaggle - LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Place it in `data/accepted_2007_to_2018q4.csv/`
3. Run notebooks 01 through 05 in order

> If you only want to test the LLM layer (notebook 05), the saved model artifacts in `models/` are sufficient.

## Saved Model Artifacts

| File | Description | Size |
|------|-------------|------|
| `xgboost_model.pkl` | Trained XGBoost classifier | ~480 KB |
| `shap_explainer.pkl` | SHAP TreeExplainer for the XGBoost model | ~1.5 MB |
| `scaler.pkl` | Fitted StandardScaler | ~2.4 KB |
| `encoder.pkl` | OneHotEncoder (drop='first') | ~1 KB |
| `feature_names.pkl` | Ordered list of 44 feature names | ~831 B |
| `threshold.pkl` | Optimized classification threshold (0.4) | ~21 B |
| `cap_values.pkl` | Outlier cap thresholds (annual_inc, revol_bal, dti) | ~183 B |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                           │
│  Raw Data → EDA → Cleaning → Feature Engineering → SMOTE   │
│  (Notebooks 01-02)                                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                           │
│  Logistic Regression │ Random Forest │ XGBoost (selected)  │
│  Threshold optimization: 0.4 (cost-sensitive)              │
│  (Notebook 03)                                             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 EXPLAINABILITY LAYER                        │
│  SHAP TreeExplainer → Global + Local Explanations          │
│  (Notebook 04)                                             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM LAYER                                │
│  SHAP output → Prompt Engineering → OpenRouter API → Text  │
│  (Notebook 05)                                             │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data** | pandas, numpy, matplotlib, seaborn, plotly |
| **ML** | scikit-learn, xgboost, imbalanced-learn (SMOTE) |
| **Explainability** | shap |
| **LLM** | openai (OpenRouter-compatible), python-dotenv |
| **Utilities** | joblib, pickle |

## Business Value

1. **Regulatory Compliance** — Explainable decisions meet GDPR, ECOA, FCRA requirements
2. **Risk Management** — Accurate default prediction reduces portfolio risk
3. **Operational Efficiency** — Automated scoring speeds up loan processing
4. **Customer Communication** — LLM explanations enable clear communication of decisions
5. **Model Governance** — SHAP values provide audit trails for model decisions

## Future Enhancements

- [ ] Streamlit web dashboard for real-time predictions
- [ ] REST API endpoint for batch and real-time inference
- [ ] Model monitoring and drift detection
- [ ] Automated retraining pipeline

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Data Source:** [LendingClub](https://www.lendingclub.com/) loan data (2007-2018)
- **Explainability:** [SHAP](https://github.com/slundberg/shap) library by Scott Lundberg
- **LLM Gateway:** [OpenRouter](https://openrouter.ai/) for LLM API access
