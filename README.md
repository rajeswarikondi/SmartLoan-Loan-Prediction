


# SmartLoan — Loan Prediction Data Analysis System

A college-project-ready Streamlit application for exploratory loan data analysis and Random Forest based loan-status prediction.

## Main modules
- Dashboard with KPIs and visual analysis
- Applicant Analysis
- Financial Analysis
- Model Performance
- Confusion Matrix
- Accuracy, Precision, Recall and F1 Score
- 5-fold cross-validation accuracy
- Feature importance
- Loan Prediction form
- Prediction probability chart
- Download prediction as CSV
- Download prediction report as PDF
- Data Quality / missing-value analysis
- Dataset explorer and filtered CSV download

## Run in VS Code (Windows)

### Option 1 — easiest
1. Extract this folder.
2. Open the `loan_project` folder in VS Code.
3. Double-click `setup.bat` once.
4. Double-click `run.bat`.
5. The browser opens the Streamlit dashboard.

### Option 2 — terminal
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run SmartLoan_Enhanced.py
```

## Dataset
The included `loan-train.csv` contains 614 loan application records and 13 columns.

## Model
- Algorithm: Random Forest Classifier
- Train/test split: 80/20
- Stratified split with fixed random state 42
- 5-fold stratified cross-validation
- Categorical preprocessing: most-frequent imputation + one-hot encoding
- Numerical preprocessing: median imputation

## Important academic note
The prediction is intended for demonstration of machine learning and data analysis. It should not be used as a real-world lending decision system.