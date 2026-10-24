"""
Evaluate the Random Forest model used on the Loan Prediction page of app.py.

Prints train/test accuracy, confusion matrix, precision / recall / F1-score,
5-fold cross-validation accuracy, feature importances and four sample
predictions.   Run:   python evaluate_model.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = Path(__file__).resolve().parent / "loan-train.csv"

CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERICAL = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

raw = pd.read_csv(DATA_FILE)
X, y = raw[CATEGORICAL + NUMERICAL], raw["Loan_Status"]

# Same preprocessing and model settings as the Loan Prediction page of app.py
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERICAL),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("=" * 60)
print(f"Records: {len(raw)}   Train: {len(X_train)}   Test: {len(X_test)}")
print(f"Train accuracy : {accuracy_score(y_train, model.predict(X_train)):.4f}")
print(f"Test accuracy  : {accuracy_score(y_test, predictions):.4f}")
print("\nConfusion matrix (rows = actual Y, N / columns = predicted Y, N):")
print(confusion_matrix(y_test, predictions, labels=["Y", "N"]))
print("\nClassification report:")
print(classification_report(y_test, predictions, digits=3))

cv_scores = cross_val_score(model, X, y, scoring="accuracy",
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print("5-fold CV accuracy per fold:", cv_scores.round(4))
print(f"Mean = {cv_scores.mean():.4f}   Std = {cv_scores.std():.4f}")

names = model.named_steps["preprocessor"].get_feature_names_out()
importance = pd.Series(model.named_steps["model"].feature_importances_,
                       index=[name.split("__", 1)[1] for name in names])
print("\nTop 10 feature importances:")
print(importance.sort_values(ascending=False).head(10).round(4))

# Four sample applicants (the cases used in the project report)
base = {"Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate",
        "Self_Employed": "No", "ApplicantIncome": 5000, "CoapplicantIncome": 0.0,
        "LoanAmount": 120.0, "Loan_Amount_Term": 360.0, "Credit_History": 1.0,
        "Property_Area": "Semiurban"}
cases = {
    "A": {},
    "B": {"Credit_History": 0.0},
    "C": {"Education": "Not Graduate", "ApplicantIncome": 2500, "LoanAmount": 250.0, "Property_Area": "Rural"},
    "D": {"Credit_History": 0.0, "Property_Area": "Rural", "Married": "No", "ApplicantIncome": 3000, "LoanAmount": 200.0},
}
print("\nSample predictions:")
for name, change in cases.items():
    applicant = pd.DataFrame([{**base, **change}])
    label = model.predict(applicant)[0]
    probability = dict(zip(model.classes_, model.predict_proba(applicant)[0]))
    result = "Approved" if label == "Y" else "Rejected"
    print(f"  Case {name}: {result:<9} approval {probability['Y'] * 100:5.1f}%   rejection {probability['N'] * 100:5.1f}%")
