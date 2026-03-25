# AI Credit Risk Scoring with Explainable AI + LLM Layer

A comprehensive machine learning system for predicting loan default risk with explicit focus on **explainability** and **human-readable explanations**. This project combines predictive modeling, SHAP-based interpretability, LLM-powered natural language explanations, and Power BI dashboards for business intelligence.

## Overview

This project addresses a critical need in banking and financial services: making ML-driven credit decisions **transparent and explainable**. The system:

- Predicts loan default probability using XGBoost
- Provides global and local explanations using SHAP
- Generates natural language explanations via LLM integration (OpenAI)
- Visualizes insights through Power BI dashboards

**Dataset:** LendingClub Loan Data (2007-2018) - 1.34 million loan records

## Key Features

| Feature | Description |
|---------|-------------|
| **Credit Risk Prediction** | Binary classification with probability scores and risk categorization (Low/Medium/High) |
| **SHAP Explainability** | Global feature importance + local explanations for individual predictions |
| **LLM Explanations** | Human-readable explanations generated via OpenAI GPT |
| **Power BI Dashboards** | Executive summaries, risk analysis, and detailed loan exploration |
| **Class Imbalance Handling** | SMOTE oversampling for rare default events |

## Project Structure

```
AI Credit Risk Scoring + Explainable AI + LLM Layer/
├── data/
│   ├── accepted_2007_to_2018q4.csv      # Raw LendingClub data
│   ├── cleaned_data/
│   │   └── cleaned_data.csv             # Preprocessed data (1.34M records)
│   └── featured_data/
│       └── featured_data.csv            # Engineered features (45 features)
│
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory Data Analysis
│   ├── 02_feature_eng.ipynb             # Feature Engineering
│   └── 03_modeling.ipynb                # Model Training & Evaluation
│
├── models/
│   ├── xgboost_model.pkl                # Trained XGBoost classifier
│   ├── scaler.pkl                       # StandardScaler for features
│   ├── encoder.pkl                      # Categorical encoder
│   ├── feature_names.pkl                # Feature names for inference
│   └── cap_values.pkl                   # Outlier capping thresholds
│
├── powerbi/
│   ├── generate_excel_for_powerbi.py    # Script to generate Power BI data
│   ├── Credit_Risk_PowerBI.xlsx         # Multi-sheet Excel for dashboards
│   └── PowerBI_Dashboard_Guide.md       # Dashboard creation guide
│
├── docs/                                # Project documentation
├── src/                                 # Source code directory
├── requirements.txt                     # Python dependencies
└── LICENSE                              # Apache 2.0 License
```

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

4. (Optional) Set up OpenAI API key for LLM explanations:
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-api-key" > .env
```

## Usage

### 1. Run the Analysis Pipeline

Execute the notebooks in order:

```
notebooks/01_eda.ipynb         → Exploratory Data Analysis
notebooks/02_feature_eng.ipynb → Feature Engineering
notebooks/03_modeling.ipynb    → Model Training & Evaluation
```

### 2. Generate Power BI Data

```bash
cd powerbi
python generate_excel_for_powerbi.py
```

This creates `Credit_Risk_PowerBI.xlsx` with 8 sheets ready for Power BI import.

### 3. Model Inference

```python
import pickle
import pandas as pd

# Load model artifacts
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Prepare input data (45 features)
X_new = pd.DataFrame([...])
X_scaled = scaler.transform(X_new)

# Predict
probability = model.predict_proba(X_scaled)[:, 1]
prediction = model.predict(X_scaled)  # 0 = No Default, 1 = Default
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 65.98% | 32.28% | 64.19% | 42.96% | 71.14% |
| Random Forest | 64.13% | 31.41% | 67.33% | 42.83% | 71.20% |
| **XGBoost (Selected)** | **64.94%** | **32.13%** | **68.04%** | **43.65%** | **72.12%** |

**Model Selection Rationale:**
- Highest ROC-AUC (0.7212) - best discrimination between defaults/non-defaults
- High recall (68.04%) - catches most actual defaults
- Balanced trade-off between false positives and false negatives

## Feature Engineering

The pipeline engineers 45 features from raw loan data:

### Original Features
- Loan details: `loan_amnt`, `funded_amnt`, `int_rate`, `term_in_months`
- Borrower profile: `annual_inc`, `dti`, `emp_length_num`
- Credit history: `fico_range_low`, `fico_range_high`, `delinq_2yrs`, `revol_util`
- One-hot encoded: `home_ownership`, `purpose`, `verification_status`

### Engineered Features
- `fico_avg`: Average FICO score
- `loan_to_income`: Loan amount / Annual income
- `installment_to_income`: Monthly installment / Monthly income
- `credit_per_account`: Credit utilization per account
- `credit_history_months`: Length of credit history

## Explainability

### SHAP Integration

The system provides two levels of explanations:

**Global Explanations:** Which features matter most across all loans?
- SHAP summary plots
- Feature importance rankings

**Local Explanations:** Why was THIS specific loan flagged?
- SHAP waterfall plots
- Individual feature contributions

### LLM Natural Language Explanations

Model outputs are translated into human-readable explanations:

```
Input: Model prediction + SHAP values

Output: "The applicant has a high risk of default (82% probability) due to:
1. High debt-to-income ratio (35%) - significantly above acceptable levels
2. Relatively low annual income ($55,000) - limits repayment capacity
3. Limited credit history (3 years) - insufficient payment track record

Recommendation: Consider additional verification or reject application."
```

## Power BI Dashboard

The generated Excel file contains:

| Sheet | Contents |
|-------|----------|
| Loans | Sample of 10,000 loan records with predictions |
| Risk Summary | Aggregated metrics by risk category |
| FICO Analysis | Default rates by credit score band |
| Purpose Analysis | Default rates by loan purpose |
| Model Metrics | Confusion matrix, ROC-AUC, accuracy |
| Model Comparison | Side-by-side model performance |
| Feature Importance | Top 15 predictive features |
| KPI Summary | Single row for dashboard KPI cards |

See [PowerBI_Dashboard_Guide.md](powerbi/PowerBI_Dashboard_Guide.md) for detailed setup instructions.

## Tech Stack

### Data Science
- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computing
- **matplotlib** >= 3.7.0 - Visualization
- **seaborn** >= 0.12.0 - Statistical plots
- **plotly** >= 5.15.0 - Interactive charts

### Machine Learning
- **scikit-learn** >= 1.3.0 - ML algorithms
- **xgboost** >= 2.0.0 - Gradient boosting
- **imbalanced-learn** >= 0.11.0 - SMOTE

### Explainability & LLM
- **shap** >= 0.42.0 - Model interpretability
- **openai** >= 1.0.0 - GPT integration

### Deployment
- **streamlit** >= 1.28.0 - Web application
- **joblib** >= 1.3.0 - Model serialization
- **python-dotenv** >= 1.0.0 - Environment config

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  Raw Data → EDA → Cleaning → Feature Engineering → SMOTE        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                             │
├─────────────────────────────────────────────────────────────────┤
│  Logistic Regression │ Random Forest │ XGBoost (selected)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPLAINABILITY LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  SHAP Values → Feature Importance → Local/Global Explanations   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  Model Output + SHAP → OpenAI GPT → Natural Language Explanation│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard │ Power BI Reports │ API Endpoints         │
└─────────────────────────────────────────────────────────────────┘
```

## Business Value

This project addresses key challenges in modern banking:

1. **Regulatory Compliance:** Explainable decisions meet regulatory requirements (GDPR, ECOA, FCRA)
2. **Risk Management:** Accurate default prediction reduces portfolio risk
3. **Operational Efficiency:** Automated scoring speeds up loan processing
4. **Customer Communication:** LLM explanations enable clear communication of decisions
5. **Model Governance:** SHAP values provide audit trails for model decisions

## Future Enhancements

- [ ] Streamlit web application for real-time predictions
- [ ] API endpoint for batch and real-time inference
- [ ] Model monitoring and drift detection
- [ ] Automated retraining pipeline
- [ ] A/B testing framework for model updates
- [ ] Feature store for consistent feature generation

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Data Source:** [LendingClub](https://www.lendingclub.com/) loan data (2007-2018)
- **Explainability:** [SHAP](https://github.com/slundberg/shap) library by Scott Lundberg
- **Visualization:** Power BI by Microsoft
