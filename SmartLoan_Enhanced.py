"""
SmartLoan - Loan Prediction & Analysis System (Streamlit)

Run with:   streamlit run SmartLoan_Enhanced.py
Data file:  loan-train.csv (same folder) or upload a CSV from the sidebar.

Units used by the dataset (and therefore by this app):
    ApplicantIncome / CoapplicantIncome : rupees PER MONTH
    LoanAmount                          : rupees in THOUSANDS
    Loan_Amount_Term                    : months

Code layout
    1. Configuration         4. Business rules (score, EMI, recommendation)
    2. Data layer            5. Persistence + PDF/CSV reports
    3. Model layer           6. UI helpers, pages, main()
"""
from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "loan-train.csv"
HISTORY_FILE = BASE_DIR / "applicant_history.csv"
APPLICATION_FILE = BASE_DIR / "loan_applications.csv"

ID_COL = "Loan_ID"
TARGET = "Loan_Status"

# Gender is deliberately NOT a model/score input (fairness). Marital status is kept
# because it is part of the project dataset; remove "Married" here to drop it too.
CATEGORICAL = ["Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
INCOME_LIKE = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]   # median-imputed
DISCRETE_NUM = ["Loan_Amount_Term", "Credit_History"]                  # mode-imputed
NUMERICAL = INCOME_LIKE + DISCRETE_NUM
FEATURES = CATEGORICAL + NUMERICAL
REQUIRED = [ID_COL] + FEATURES + [TARGET]

TERM_OPTIONS = [12, 36, 60, 84, 120, 180, 240, 360]
DEFAULT_RATE = 9.0        # assumed annual interest rate (%) for EMI-based rules
MAX_EMI_SHARE = 0.40      # EMI should not exceed 40% of monthly household income
STATUS_COLORS = {"Approved": "#059669", "Rejected": "#ea580c"}
DISCLAIMER = ("Academic demonstration only - the prediction, score and recommendation are for this "
              "project and must not be used for real-world lending decisions.")

HISTORY_COLUMNS = (["Generated_At"] + FEATURES +
                   ["Prediction", "Approval_Probability_Percent", "Eligibility_Score"])
APPLICATION_COLUMNS = (["Application_ID", "Submitted_At", "Applicant_Name", "Phone", "Email", "Gender"] +
                       FEATURES + ["Model_Screening", "Approval_Probability_Percent", "Eligibility_Score"])

FRIENDLY = {
    "Married": "Marital status", "Dependents": "Number of dependents", "Education": "Education",
    "Self_Employed": "Self-employment", "Property_Area": "Property area",
    "ApplicantIncome": "Applicant income", "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Loan amount", "Loan_Amount_Term": "Loan term", "Credit_History": "Credit history",
}

# Streamlit >= 1.50 uses width="stretch"; older versions use use_container_width=True.
try:
    _ST_VERSION = tuple(int(p) for p in re.findall(r"\d+", st.__version__)[:2])
except Exception:  # pragma: no cover
    _ST_VERSION = (0, 0)
NEW_WIDTH_API = _ST_VERSION >= (1, 50)


# =============================================================================
# 2. DATA LAYER
# =============================================================================
_POSITIVE = {"Y", "YES", "1", "TRUE", "APPROVED"}
_NEGATIVE = {"N", "NO", "0", "FALSE", "REJECTED"}
_TARGET_MAP = {**{k: "Y" for k in _POSITIVE}, **{k: "N" for k in _NEGATIVE}}


def normalise_target(series: pd.Series) -> pd.Series:
    """Map Y/Yes/1/Approved ... to 'Y' and N/No/0/Rejected ... to 'N'; anything else -> NaN."""
    text = series.astype("object").map(lambda v: str(v).strip().upper() if pd.notna(v) else np.nan)
    return text.map(_TARGET_MAP)


_MISSING_TEXT = {"", "nan", "none", "null", "na", "n/a"}


def _clean_dependents(value):
    if pd.isna(value) or str(value).strip().lower() in _MISSING_TEXT:
        return np.nan
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    if text.endswith("+") or (text.isdigit() and int(text) >= 3):
        return "3+"
    return text or np.nan


def _clean_text(value):
    if pd.isna(value) or str(value).strip().lower() in _MISSING_TEXT:
        return np.nan
    return str(value).strip()


@st.cache_data(show_spinner=False)
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Type-fix and validate rows. Missing FEATURE values are left as NaN on purpose:
    they are imputed inside the model pipeline so no information from the test fold leaks."""
    d = raw.copy()
    d.columns = d.columns.astype(str).str.strip()
    d = d.drop_duplicates()
    d[ID_COL] = d[ID_COL].astype(str).str.strip()
    for col in CATEGORICAL:
        d[col] = d[col].map(_clean_dependents if col == "Dependents" else _clean_text).astype("object")
    for col in NUMERICAL:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d.loc[~d["Credit_History"].isin([0, 1]), "Credit_History"] = np.nan
    for col in INCOME_LIKE:                       # negative money / zero loan are data errors
        d.loc[d[col] < 0, col] = np.nan
    d[TARGET] = normalise_target(d[TARGET])
    d = d.dropna(subset=[TARGET])                 # unlabeled rows cannot be used for training
    return d.reset_index(drop=True)


def impute_for_analysis(d: pd.DataFrame) -> pd.DataFrame:
    """Filled copy used ONLY for charts/tables (the model uses its own pipeline imputers)."""
    v = d.copy()
    for col in CATEGORICAL + DISCRETE_NUM:
        mode = v[col].mode(dropna=True)
        v[col] = v[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    for col in INCOME_LIKE:
        v[col] = v[col].fillna(v[col].median())
    v["Status"] = v[TARGET].map({"Y": "Approved", "N": "Rejected"})
    v["Approved"] = (v[TARGET] == "Y").astype(int)
    v["Total_Income"] = v["ApplicantIncome"] + v["CoapplicantIncome"]
    return v


def quality_stats(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    r = raw.copy()
    r.columns = r.columns.astype(str).str.strip()
    deduped = r.drop_duplicates()
    unlabeled = int(normalise_target(deduped[TARGET]).isna().sum())
    return {"original": len(r), "duplicates": len(r) - len(deduped),
            "unlabeled": unlabeled, "clean": len(clean)}


def check_trainable(data: pd.DataFrame) -> str | None:
    """Return an error message if the dataset cannot support training, else None."""
    if len(data) < 30:
        return f"Only {len(data)} usable labelled rows were found; at least 30 are needed."
    counts = data[TARGET].value_counts()
    if len(counts) < 2 or counts.min() < 6:
        return "Both Approved (Y) and Rejected (N) classes need at least 6 rows each for validation."
    return None


@st.cache_data(show_spinner=False)
def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df.columns = df.columns.astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def _read_csv_path(path: str, mtime: float) -> pd.DataFrame:   # mtime busts the cache on edit
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_raw(upload) -> pd.DataFrame:
    try:
        if upload is not None:
            raw = _read_csv_bytes(upload.getvalue())
        elif DATA_FILE.exists():
            raw = _read_csv_path(str(DATA_FILE), DATA_FILE.stat().st_mtime)
        else:
            st.error(f"{DATA_FILE.name} was not found next to the app. Please upload a CSV dataset.")
            st.stop()
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"The CSV file could not be read: {exc}")
        st.stop()
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()
    return raw


# =============================================================================
# 3. MODEL LAYER
# =============================================================================
def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), INCOME_LIKE),
        ("disc", SimpleImputer(strategy="most_frequent"), DISCRETE_NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    # Regularised forest: shallow trees + min leaf size reduce over-fitting on small data.
    model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    return Pipeline([("preprocessor", pre), ("model", model)])


def approval_probability(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X[FEATURES])[:, list(pipe.classes_).index("Y")]


def _importance_by_feature(pipe: Pipeline) -> pd.Series:
    """Sum one-hot importances back to the original input feature."""
    names = [n.split("__", 1)[-1] for n in pipe.named_steps["preprocessor"].get_feature_names_out()]
    raw = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    out = {}
    for name, value in raw.items():
        owner = next((f for f in FEATURES if name == f or name.startswith(f + "_")), name)
        out[owner] = out.get(owner, 0.0) + value
    return pd.Series(out).sort_values(ascending=False)


@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model(data: pd.DataFrame) -> dict:
    """Evaluate honestly (hold-out + 5-fold CV), then fit the deployed model on ALL rows."""
    X, y = data[FEATURES], data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    holdout = build_pipeline().fit(X_tr, y_tr)
    pred = holdout.predict(X_te)
    proba = approval_probability(holdout, X_te)
    cv = cross_val_score(build_pipeline(), X, y, scoring="accuracy",
                         cv=StratifiedKFold(5, shuffle=True, random_state=42))

    final = build_pipeline().fit(X, y)
    reference = {c: float(X[c].median()) for c in INCOME_LIKE}
    for c in CATEGORICAL + DISCRETE_NUM:
        reference[c] = X[c].mode(dropna=True).iloc[0]
    if "Loan_Amount_Term" in reference:
        reference["Loan_Amount_Term"] = float(reference["Loan_Amount_Term"])
    if "Credit_History" in reference:
        reference["Credit_History"] = float(reference["Credit_History"])

    report = classification_report(y_te, pred, labels=["Y", "N"], target_names=["Approved", "Rejected"],
                                   output_dict=True, zero_division=0)
    report.pop("accuracy", None)   # scalar row would be mis-shaped in a table
    return {
        "pipeline": final,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, pos_label="Y", zero_division=0),
        "recall": recall_score(y_te, pred, pos_label="Y", zero_division=0),
        "f1": f1_score(y_te, pred, pos_label="Y", zero_division=0),
        "roc_auc": roc_auc_score((y_te == "Y").astype(int), proba),
        "cv_mean": cv.mean(), "cv_std": cv.std(),
        "baseline": float((y == y.mode().iloc[0]).mean()),   # always-predict-majority accuracy
        "matrix": confusion_matrix(y_te, pred, labels=["Y", "N"]),
        "report": report,
        "importance": _importance_by_feature(final),
        "reference": reference,
        "ranges": {c: (float(X[c].min()), float(X[c].max())) for c in INCOME_LIKE},
        "n_train": len(X_tr), "n_test": len(X_te),
    }


def explain_prediction(pipe: Pipeline, inputs: dict, reference: dict) -> pd.DataFrame:
    """What-if explanation: replace one feature at a time with the typical applicant's value and
    measure how the approval probability moves. Positive impact = this value HELPS the applicant."""
    base = float(approval_probability(pipe, pd.DataFrame([inputs]))[0])
    variants = []
    for f in FEATURES:
        alt = dict(inputs)
        alt[f] = reference[f]
        variants.append(alt)
    alt_probs = approval_probability(pipe, pd.DataFrame(variants))
    rows = [{"Feature": FRIENDLY[f], "Your value": inputs[f], "Typical value": reference[f],
             "Impact (pp)": round((base - p) * 100, 2)} for f, p in zip(FEATURES, alt_probs)]
    out = pd.DataFrame(rows)
    out["Your value"] = out["Your value"].astype(str)
    out["Typical value"] = out["Typical value"].map(lambda v: f"{v:,.0f}" if isinstance(v, float) else str(v))
    return out.reindex(out["Impact (pp)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def out_of_range_warnings(inputs: dict, ranges: dict) -> list[str]:
    msgs = []
    for col, (lo, hi) in ranges.items():
        v = float(inputs[col])
        if v < lo or v > hi:
            msgs.append(f"{FRIENDLY[col]} ({v:,.0f}) is outside the range seen in the training data "
                        f"({lo:,.0f} - {hi:,.0f}); the prediction is less reliable.")
    return msgs


# =============================================================================
# 4. BUSINESS RULES
# =============================================================================
def emi_calculator(principal: float, annual_rate: float, months: int) -> float:
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / months
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def max_principal(emi: float, annual_rate: float, months: int) -> float:
    """Largest loan whose EMI equals `emi` (inverse of emi_calculator)."""
    r = annual_rate / 12 / 100
    if emi <= 0 or months <= 0:
        return 0.0
    if r == 0:
        return emi * months
    growth = (1 + r) ** months
    return emi * (growth - 1) / (r * growth)


def household_income(inputs: dict) -> float:
    return float(inputs["ApplicantIncome"]) + float(inputs["CoapplicantIncome"])


def affordability(inputs: dict, rate: float = DEFAULT_RATE) -> dict:
    """LoanAmount is in THOUSANDS of rupees, incomes are MONTHLY rupees."""
    principal = float(inputs["LoanAmount"]) * 1000
    months = int(inputs["Loan_Amount_Term"])
    emi = emi_calculator(principal, rate, months)
    income = household_income(inputs)
    return {"principal": principal, "months": months, "emi": emi,
            "emi_ratio": emi / income if income > 0 else float("inf")}


def score_band(score: int) -> str:
    return "Excellent" if score >= 75 else "Good" if score >= 55 else "Fair" if score >= 40 else "Low"


def eligibility_score(inputs: dict) -> tuple[int, pd.DataFrame]:
    """Transparent 0-100 demo score with a per-criterion breakdown (not a real credit score)."""
    income = household_income(inputs)
    ratio = affordability(inputs)["emi_ratio"]
    parts = []

    def add(name, points, maximum, reason):
        parts.append({"Criterion": name, "Points": points, "Max": maximum, "Reason": reason})

    good_credit = float(inputs["Credit_History"]) == 1
    add("Credit history", 35 if good_credit else 0, 35,
        "Good credit record" if good_credit else "No / poor credit record")

    if income <= 0:
        pts, why = 0, "No income entered"
    elif ratio <= 0.30:
        pts, why = 30, f"EMI is {ratio:.0%} of income (comfortable)"
    elif ratio <= MAX_EMI_SHARE:
        pts, why = 22, f"EMI is {ratio:.0%} of income (acceptable)"
    elif ratio <= 0.50:
        pts, why = 12, f"EMI is {ratio:.0%} of income (stretched)"
    else:
        pts, why = 4, f"EMI is {ratio:.0%} of income (very high)"
    add("EMI affordability", pts, 30, why)

    pts = 15 if income >= 10000 else 11 if income >= 6000 else 7 if income >= 3000 else 3 if income > 0 else 0
    add("Household income", pts, 15, f"Monthly household income Rs. {income:,.0f}")

    add("Education", 5 if inputs["Education"] == "Graduate" else 3, 5, str(inputs["Education"]))
    add("Employment", 5 if inputs["Self_Employed"] == "No" else 3, 5,
        "Salaried / not self-employed" if inputs["Self_Employed"] == "No" else "Self-employed")
    add("Dependents", {"0": 5, "1": 4, "2": 3}.get(str(inputs["Dependents"]), 2), 5,
        f"{inputs['Dependents']} dependent(s)")
    add("Property area", 5 if inputs["Property_Area"] in ("Urban", "Semiurban") else 3, 5,
        str(inputs["Property_Area"]))

    table = pd.DataFrame(parts)
    return int(min(100, round(table["Points"].sum()))), table


def loan_recommendation(inputs: dict, rate: float = DEFAULT_RATE) -> dict:
    """Affordability-based suggestion: EMI <= 40% of monthly household income (credit-adjusted)."""
    income = household_income(inputs)
    credit_factor = 1.0 if float(inputs["Credit_History"]) == 1 else 0.60
    cap_emi = income * MAX_EMI_SHARE * credit_factor
    requested = float(inputs["LoanAmount"]) * 1000
    req_term = int(inputs["Loan_Amount_Term"])

    suggested_term = next((t for t in TERM_OPTIONS if max_principal(cap_emi, rate, t) >= requested),
                          max(TERM_OPTIONS))
    suggested = min(requested, max_principal(cap_emi, rate, suggested_term))
    return {
        "requested_amount": requested, "requested_term": req_term,
        "fits_requested_term": max_principal(cap_emi, rate, req_term) >= requested,
        "max_at_requested_term": max_principal(cap_emi, rate, req_term),
        "suggested_amount": float(np.floor(suggested / 1000) * 1000),
        "suggested_term": suggested_term,
        "suggested_emi": emi_calculator(suggested, rate, suggested_term),
        "emi_cap": cap_emi, "rate": rate, "credit_adjusted": credit_factor < 1,
    }


def run_prediction(ctx: "Context", inputs: dict) -> dict:
    """Single entry point used by every page that needs a prediction."""
    pipe = ctx.model["pipeline"]
    approval = float(approval_probability(pipe, pd.DataFrame([inputs]))[0]) * 100
    label = "APPROVED" if approval >= 50 else "REJECTED"
    score, breakdown = eligibility_score(inputs)
    result = {
        "inputs": inputs, "prediction": label, "approval": approval, "score": score,
        "breakdown": breakdown, "recommendation": loan_recommendation(inputs),
        "affordability": affordability(inputs),
        "explanation": explain_prediction(pipe, inputs, ctx.model["reference"]),
        "warnings": out_of_range_warnings(inputs, ctx.model["ranges"]),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    result["csv"] = csv_prediction(result)
    result["pdf"] = pdf_report(result, ctx.model)
    return result


# =============================================================================
# 5. PERSISTENCE + REPORTS
# =============================================================================
def _ensure_schema(path: Path, columns: list[str]) -> None:
    """If an older file has a different header, migrate it instead of corrupting it on append."""
    if not path.exists() or path.stat().st_size == 0:
        return
    existing = pd.read_csv(path)
    if list(existing.columns) != columns:
        existing.reindex(columns=columns).to_csv(path, index=False)


def append_row(path: Path, row: dict, columns: list[str]) -> bool:
    try:
        _ensure_schema(path, columns)
        pd.DataFrame([row]).reindex(columns=columns).to_csv(
            path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
        return True
    except OSError as exc:      # e.g. read-only hosting
        st.warning(f"Could not save to {path.name}: {exc}")
        return False


def read_records(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str) if path.exists() and path.stat().st_size else pd.DataFrame()
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def save_history(result: dict) -> bool:
    row = {"Generated_At": result["generated_at"], **result["inputs"],
           "Prediction": result["prediction"],
           "Approval_Probability_Percent": round(result["approval"], 2),
           "Eligibility_Score": result["score"]}
    return append_row(HISTORY_FILE, row, HISTORY_COLUMNS)


def csv_prediction(result: dict) -> bytes:
    row = {**result["inputs"], "Prediction": result["prediction"],
           "Approval_Probability_Percent": round(result["approval"], 2),
           "Eligibility_Score": result["score"], "Generated_At": result["generated_at"]}
    return pd.DataFrame([row]).to_csv(index=False).encode()


def pdf_report(result: dict, metrics: dict) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()

    def table(rows, header_color, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 6)]))
        return t

    inputs_rows = [["Applicant field", "Value"]] + [
        [FRIENDLY.get(k, k.replace("_", " ")), str(v)] for k, v in result["inputs"].items()]
    inputs_rows.append(["Units", "Income: Rs./month | Loan amount: Rs. thousand"])
    m = metrics
    metric_rows = [["Metric", "Value"],
                   ["Test accuracy", f"{m['accuracy'] * 100:.2f}%"],
                   ["Cross-validation accuracy", f"{m['cv_mean'] * 100:.2f}%"],
                   ["Precision (Approved)", f"{m['precision'] * 100:.2f}%"],
                   ["Recall (Approved)", f"{m['recall'] * 100:.2f}%"],
                   ["F1 score", f"{m['f1'] * 100:.2f}%"],
                   ["ROC-AUC", f"{m['roc_auc']:.3f}"]]
    story = [
        Paragraph("SmartLoan - Loan Prediction Report", styles["Title"]), Spacer(1, 8),
        Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles["Normal"]),
        Spacer(1, 14),
        Paragraph(f"Prediction: <b>{escape(result['prediction'])}</b>", styles["Heading2"]),
        Paragraph(f"Approval probability: <b>{result['approval']:.2f}%</b> &nbsp;|&nbsp; "
                  f"Eligibility score: <b>{result['score']}/100</b>", styles["Normal"]),
        Spacer(1, 10), table(inputs_rows, "#0f172a", [70 * mm, 90 * mm]), Spacer(1, 14),
        Paragraph("Model Evaluation", styles["Heading2"]),
        table(metric_rows, "#2563eb", [90 * mm, 70 * mm]), Spacer(1, 14),
        Paragraph(escape(DISCLAIMER), styles["Italic"]),
    ]
    doc.build(story)
    return buf.getvalue()


# =============================================================================
# 6. UI HELPERS
# =============================================================================
CSS = """
<style>
.stApp { background:#f8fafc; }
.block-container { padding-top:1.5rem; padding-bottom:2rem; max-width:1450px; }
[data-testid="stSidebar"] { background:#0f172a; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] small { color:#e2e8f0; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color:#0f172a !important; }
.hero { background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%); padding:28px 32px; border-radius:18px;
        color:white; margin-bottom:20px; box-shadow:0 8px 25px rgba(15,23,42,.12); }
.hero h1 { margin:0 0 6px 0; font-size:32px; color:white; }
.hero p { margin:0; color:#dbeafe; font-size:15px; }
.kpi { background:white; border:1px solid #e2e8f0; border-radius:14px; padding:17px 18px;
       box-shadow:0 2px 8px rgba(15,23,42,.04); height:100%; }
.kpi-label { color:#64748b; font-size:13px; margin-bottom:5px; }
.kpi-value { color:#0f172a; font-size:25px; font-weight:750; }
.section-title { color:#0f172a; font-size:20px; font-weight:700; margin:22px 0 10px; }
.result-approved { background:#ecfdf5; border:1px solid #86efac; border-radius:14px; padding:22px; color:#064e3b; }
.result-rejected { background:#fff7ed; border:1px solid #fdba74; border-radius:14px; padding:22px; color:#7c2d12; }
.result-approved h2, .result-rejected h2, .result-approved p, .result-rejected p { color:inherit; }
.small-note { color:#64748b; font-size:13px; }
div[data-testid="stMetric"] { background:white; border:1px solid #e2e8f0; padding:14px; border-radius:14px; }
</style>
"""


def _stretch_call(fn, *args, **kwargs):
    """Call a Streamlit element so it fills its container on both old and new Streamlit versions."""
    if NEW_WIDTH_API:
        try:
            return fn(*args, width="stretch", **kwargs)
        except Exception:      # unknown width option on this build -> use legacy flag
            pass
    return fn(*args, use_container_width=True, **kwargs)


def show_df(df: pd.DataFrame) -> None:
    _stretch_call(st.dataframe, df, hide_index=True)


def chart(fig) -> None:
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=45, b=10),
                      font=dict(family="Arial", color="#334155"))
    _stretch_call(st.plotly_chart, fig, config={"displaylogo": False})


def kpi(label: str, value, note: str | None = None) -> None:
    note_html = f'<div class="small-note">{escape(str(note))}</div>' if note else ""
    st.markdown(f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
                f'<div class="kpi-value">{escape(str(value))}</div>{note_html}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def money(value: float) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"₹{value:,.0f}"


def need_prediction() -> dict | None:
    last = st.session_state.get("last_result")
    if not last:
        st.info("Generate a prediction first from the **🔮 Loan Prediction** page "
                "(or submit the **📝 Application Form**).")
    return last


def applicant_fields(prefix: str, with_gender: bool = False, coapp_default: float = 1500.0) -> dict:
    """Shared input widgets (must be called inside an st.form). Returns the model inputs + extras."""
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender (record only - not used by the model)", ["Male", "Female", "Other"],
                              key=f"{prefix}_gender") if with_gender else None
        married = st.selectbox("Married", ["Yes", "No"], key=f"{prefix}_married")
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], key=f"{prefix}_dep")
    with c2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], key=f"{prefix}_edu")
        self_emp = st.selectbox("Self employed", ["No", "Yes"], key=f"{prefix}_self")
        area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"], key=f"{prefix}_area")
    with c3:
        income = st.number_input("Applicant income (₹ / month)", min_value=0.0, value=5000.0,
                                 step=500.0, key=f"{prefix}_inc")
        coapp = st.number_input("Co-applicant income (₹ / month)", min_value=0.0, value=coapp_default,
                                step=500.0, key=f"{prefix}_coinc")
        loan = st.number_input("Loan amount (₹ thousand)", min_value=0.0, value=150.0, step=10.0,
                               key=f"{prefix}_loan")
    c1, c2, _ = st.columns(3)
    with c1:
        term = st.selectbox("Loan term (months)", sorted(TERM_OPTIONS, reverse=True), key=f"{prefix}_term")
    with c2:
        credit = st.selectbox("Credit history", [1.0, 0.0], key=f"{prefix}_credit",
                              format_func=lambda x: "Good / available (1)" if x == 1.0
                              else "Not available / poor (0)")
    inputs = {"Married": married, "Dependents": dependents, "Education": education,
              "Self_Employed": self_emp, "Property_Area": area,
              "ApplicantIncome": float(income), "CoapplicantIncome": float(coapp),
              "LoanAmount": float(loan), "Loan_Amount_Term": float(term), "Credit_History": float(credit)}
    return {"inputs": inputs, "gender": gender}


def validate_money(inputs: dict) -> list[str]:
    errors = []
    if inputs["LoanAmount"] <= 0:
        errors.append("Loan amount must be greater than zero.")
    if household_income(inputs) <= 0:
        errors.append("Enter at least one income (applicant or co-applicant).")
    return errors


def validate_contact(name: str, phone: str, email: str) -> list[str]:
    errors = []
    if len(name.strip()) < 2:
        errors.append("Applicant name is required.")
    digits = re.sub(r"[\s\-]", "", phone)
    digits = digits[3:] if digits.startswith("+91") else digits
    if not re.fullmatch(r"[6-9]\d{9}", digits):
        errors.append("Enter a valid 10-digit Indian mobile number.")
    if email.strip() and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        errors.append("Email address is not valid.")
    return errors


@dataclass
class Context:
    raw: pd.DataFrame
    data: pd.DataFrame      # cleaned, un-imputed rows used for training
    view: pd.DataFrame      # imputed copy used for charts
    model: dict
    quality: dict


# ----------------------------------------------------------------------------- pages
def render_result(ctx: Context, r: dict) -> None:
    """Rendered from session_state so it survives the reruns caused by download buttons."""
    approved = r["prediction"] == "APPROVED"
    st.markdown(
        f'<div class="{"result-approved" if approved else "result-rejected"}">'
        f'<h2>{"✅" if approved else "⚠️"} Loan {r["prediction"]}</h2>'
        f'<p>Estimated approval probability: <b>{r["approval"]:.2f}%</b> &nbsp;|&nbsp; '
        f'Eligibility score: <b>{r["score"]}/100 ({score_band(r["score"])})</b></p></div>',
        unsafe_allow_html=True)
    for msg in r["warnings"]:
        st.warning(msg)

    aff, rec = r["affordability"], r["recommendation"]
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Eligibility score", f"{r['score']}/100", score_band(r["score"]))
    with c2:
        kpi("Estimated EMI", money(aff["emi"]), f"at {DEFAULT_RATE:.1f}% p.a., {aff['months']} months")
    with c3:
        kpi("EMI / income", "N/A" if not np.isfinite(aff["emi_ratio"]) else f"{aff['emi_ratio']:.0%}",
            f"guideline: ≤ {MAX_EMI_SHARE:.0%}")
    with c4:
        kpi("Suggested loan", money(rec["suggested_amount"]), f"over {rec['suggested_term']} months")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=r["approval"], number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Approval probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"},
                   "steps": [{"range": [0, 50], "color": "#fed7aa"}, {"range": [50, 100], "color": "#bbf7d0"}],
                   "threshold": {"line": {"color": "#0f172a", "width": 3}, "value": 50}}))
        chart(fig)
    with c2:
        st.subheader("Applicant summary")
        show = pd.DataFrame({"Field": [FRIENDLY[k] for k in r["inputs"]],
                             "Value": [str(v) for v in r["inputs"].values()]})
        show_df(show)

    st.subheader("🤖 Why this result?")
    explanation_lines(r["explanation"])

    st.markdown("### Download prediction")
    d1, d2 = st.columns(2)
    with d1:
        _stretch_call(st.download_button, "⬇️ Download CSV prediction", data=r["csv"],
                      file_name="loan_prediction.csv", mime="text/csv")
    with d2:
        if r["pdf"]:
            _stretch_call(st.download_button, "📄 Download PDF report", data=r["pdf"],
                          file_name="loan_prediction_report.pdf", mime="application/pdf")
        else:
            st.info("PDF download needs reportlab:  pip install reportlab")
    st.caption(DISCLAIMER)


def explanation_lines(expl: pd.DataFrame) -> None:
    helps = expl[expl["Impact (pp)"] >= 1].head(3)
    hurts = expl[expl["Impact (pp)"] <= -1].head(3)
    for _, row in helps.iterrows():
        st.write(f"✅ **{row['Feature']}** ({row['Your value']}) raises the approval probability by "
                 f"{row['Impact (pp)']:.1f} points compared with a typical applicant ({row['Typical value']}).")
    for _, row in hurts.iterrows():
        st.write(f"⚠️ **{row['Feature']}** ({row['Your value']}) lowers the approval probability by "
                 f"{abs(row['Impact (pp)']):.1f} points compared with a typical applicant ({row['Typical value']}).")
    if helps.empty and hurts.empty:
        st.write("This profile is close to the typical applicant; no single factor moves the result much.")
    st.caption("Impacts come from re-scoring the applicant with one factor at a time set to the typical value "
               "(a what-if analysis of the trained model).")


def page_dashboard(ctx: Context) -> None:
    d, v = ctx.data, ctx.view
    hero("SmartLoan - Loan Prediction",
         "Data analysis, applicant insights and machine-learning based loan status prediction.")
    total = len(d)
    approved = int((d[TARGET] == "Y").sum())
    cols = st.columns(5)
    items = [("Total applications", f"{total:,}"), ("Approved loans", f"{approved:,}"),
             ("Rejected loans", f"{total - approved:,}"), ("Approval rate", f"{approved / total * 100:.1f}%"),
             ("Model accuracy", f"{ctx.model['accuracy'] * 100:.1f}%")]
    for c, (a, b) in zip(cols, items):
        with c:
            kpi(a, b)
    section("Loan portfolio overview")
    c1, c2 = st.columns(2)
    with c1:
        s = v["Status"].value_counts().reset_index()
        s.columns = ["Status", "Applications"]
        chart(px.pie(s, names="Status", values="Applications", hole=.52, color="Status",
                     color_discrete_map=STATUS_COLORS, title="Approved vs rejected applications"))
    with c2:
        p = v["Property_Area"].value_counts().reset_index()
        p.columns = ["Property Area", "Applications"]
        chart(px.bar(p, x="Property Area", y="Applications", text_auto=True,
                     title="Applications by property area"))
    c1, c2 = st.columns(2)
    with c1:
        chart(px.histogram(v, x="ApplicantIncome", color="Status", nbins=35, color_discrete_map=STATUS_COLORS,
                           title="Applicant income distribution (₹ / month)"))
    with c2:
        chart(px.scatter(v, x="ApplicantIncome", y="LoanAmount", color="Status",
                         color_discrete_map=STATUS_COLORS,
                         title="Applicant income (₹/month) vs loan amount (₹ thousand)"))


def page_applicants(ctx: Context) -> None:
    v = ctx.view
    st.title("👤 Applicant Analysis")
    c1, c2 = st.columns(2)
    for col, feat, title in [(c1, "Education", "Education vs loan status"),
                             (c2, "Property_Area", "Property area vs loan status")]:
        with col:
            q = v.groupby([feat, "Status"]).size().reset_index(name="Applications")
            chart(px.bar(q, x=feat, y="Applications", color="Status", barmode="group", title=title,
                         color_discrete_map=STATUS_COLORS))
    c1, c2 = st.columns(2)
    with c1:
        chart(px.box(v, x="Status", y="ApplicantIncome", color="Status", color_discrete_map=STATUS_COLORS,
                     title="Applicant income by loan status"))
    with c2:
        dep = v.groupby("Dependents").size().reset_index(name="Applications")
        chart(px.bar(dep, x="Dependents", y="Applications", text_auto=True,
                     category_orders={"Dependents": ["0", "1", "2", "3+"]},
                     title="Applications by dependents"))


def page_financial(ctx: Context) -> None:
    v = ctx.view
    st.title("💰 Financial Analysis")
    st.caption("Incomes are monthly rupees; loan amounts are shown in rupees (dataset stores them in thousands).")
    est_emi = v.apply(lambda r: emi_calculator(r["LoanAmount"] * 1000, DEFAULT_RATE, int(r["Loan_Amount_Term"])),
                      axis=1)
    burden = (est_emi / v["Total_Income"].where(v["Total_Income"] > 0)).dropna()
    a, b, c, d = st.columns(4)
    with a:
        kpi("Avg applicant income", money(v["ApplicantIncome"].mean()), "per month")
    with b:
        kpi("Avg co-applicant income", money(v["CoapplicantIncome"].mean()), "per month")
    with c:
        kpi("Avg loan amount", money(v["LoanAmount"].mean() * 1000))
    with d:
        kpi("Median EMI / income", f"{burden.median():.0%}" if len(burden) else "N/A",
            f"at {DEFAULT_RATE:.1f}% p.a.")
    c1, c2 = st.columns(2)
    with c1:
        chart(px.histogram(v, x="LoanAmount", nbins=35, title="Loan amount distribution (₹ thousand)"))
    with c2:
        chart(px.scatter(v, x="ApplicantIncome", y="CoapplicantIncome", color="Status",
                         color_discrete_map=STATUS_COLORS, title="Applicant vs co-applicant income (₹ / month)"))
    if len(burden):
        tmp = v.loc[burden.index].assign(EMI_to_Income=burden.clip(upper=2))
        chart(px.box(tmp, x="Status", y="EMI_to_Income", color="Status", color_discrete_map=STATUS_COLORS,
                     title="Estimated EMI-to-income ratio by loan status (capped at 200%)"))


def page_performance(ctx: Context) -> None:
    m = ctx.model
    st.title("🎯 Model Performance")
    st.caption(f"Hold-out test: stratified 80/20 split ({m['n_train']} train / {m['n_test']} test rows) and "
               "5-fold cross-validation. The deployed model is re-fitted on all rows. "
               "Precision / recall / F1 are for the 'Approved' class.")
    items = [("Accuracy", f"{m['accuracy'] * 100:.2f}%"), ("Precision", f"{m['precision'] * 100:.2f}%"),
             ("Recall", f"{m['recall'] * 100:.2f}%"), ("F1 score", f"{m['f1'] * 100:.2f}%"),
             ("ROC-AUC", f"{m['roc_auc']:.3f}"),
             ("5-fold CV", f"{m['cv_mean'] * 100:.2f}%", f"± {m['cv_std'] * 100:.2f}")]
    for c, item in zip(st.columns(len(items)), items):
        with c:
            kpi(*item)
    st.caption(f"Baseline: always predicting the majority class gives {m['baseline'] * 100:.1f}% accuracy.")

    section("Confusion matrix")
    cm = m["matrix"]
    fig = go.Figure(go.Heatmap(z=cm, x=["Approved", "Rejected"], y=["Approved", "Rejected"], text=cm,
                               texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title="Actual vs predicted loan status (test set)", xaxis_title="Predicted",
                      yaxis_title="Actual", yaxis=dict(autorange="reversed"))
    chart(fig)

    section("Feature importance")
    imp = m["importance"].rename(index=FRIENDLY).head(12).sort_values()
    chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Importance by input feature",
                 labels={"x": "Importance", "y": "Feature"}))
    report = pd.DataFrame(m["report"]).T.reset_index().rename(columns={"index": "Class"})
    show_df(report.round(3))


def page_prediction(ctx: Context) -> None:
    hero("🔮 Loan Prediction", "Enter an applicant profile and generate a machine-learning prediction.")
    with st.form("prediction_form"):
        fields = applicant_fields("pred")
        submitted = _stretch_call(st.form_submit_button, "Generate Prediction", type="primary")
    if submitted:
        errors = validate_money(fields["inputs"])
        if errors:
            for e in errors:
                st.error(e)
        else:
            result = run_prediction(ctx, fields["inputs"])
            save_history(result)
            st.session_state["last_result"] = result
    last = st.session_state.get("last_result")
    if last:
        render_result(ctx, last)


def page_eligibility(ctx: Context) -> None:
    st.title("📈 Loan Eligibility Score")
    st.info("A transparent 0-100 project score (not a real bank credit score).")
    last = need_prediction()
    if not last:
        return
    st.metric("Latest eligibility score", f"{last['score']}/100", score_band(last["score"]))
    st.progress(min(max(last["score"] / 100, 0.0), 1.0))
    bd = last["breakdown"]
    chart(px.bar(bd.iloc[::-1], x="Points", y="Criterion", orientation="h", text="Points",
                 title="Points earned by criterion", range_x=[0, 36]))
    show_df(bd)
    st.write("Model prediction:", last["prediction"])


def page_recommendation(ctx: Context) -> None:
    st.title("💡 Loan Recommendation")
    last = need_prediction()
    if not last:
        return
    rec = last["recommendation"]
    st.info(f"Rule: monthly EMI should stay within {MAX_EMI_SHARE:.0%} of household income "
            f"(assumed rate {rec['rate']:.1f}% p.a.)"
            + ("; capped further because credit history is poor / unavailable." if rec["credit_adjusted"] else "."))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Requested loan", money(rec["requested_amount"]), f"{rec['requested_term']} months")
    with c2:
        kpi("Max at requested term", money(rec["max_at_requested_term"]))
    with c3:
        kpi("Suggested loan", money(rec["suggested_amount"]), f"over {rec['suggested_term']} months")
    with c4:
        kpi("Suggested EMI", money(rec["suggested_emi"]), f"limit {money(rec['emi_cap'])}")
    st.write("")
    if rec["fits_requested_term"]:
        st.success("The requested amount fits the affordability rule at the requested term."
                   + (f" A shorter term of {rec['suggested_term']} months would also work."
                      if rec["suggested_term"] < rec["requested_term"] else ""))
    elif rec["suggested_amount"] < rec["requested_amount"]:
        st.warning("The requested amount is above what the income supports even at the longest term; "
                   "consider a smaller loan or adding a co-applicant income.")
    else:
        st.warning(f"The requested amount needs a longer term - about {rec['suggested_term']} months.")


def page_emi(ctx: Context) -> None:
    st.title("🏦 Loan EMI Calculator")
    c1, c2, c3 = st.columns(3)
    with c1:
        principal = st.number_input("Loan amount (₹)", min_value=0.0, value=200000.0, step=10000.0)
    with c2:
        rate = st.number_input("Annual interest rate (%)", min_value=0.0, max_value=60.0, value=8.5, step=0.1)
    with c3:
        months = int(st.number_input("Tenure (months)", min_value=1, max_value=480, value=120, step=12))
    emi = emi_calculator(principal, rate, months)
    total = emi * months
    interest = total - principal
    a, b, c = st.columns(3)
    with a:
        kpi("Monthly EMI", f"₹{emi:,.2f}")
    with b:
        kpi("Total payment", f"₹{total:,.2f}")
    with c:
        kpi("Total interest", f"₹{interest:,.2f}")
    if principal <= 0:
        st.info("Enter a loan amount above zero to see the breakdown.")
        return
    c1, c2 = st.columns(2)
    with c1:
        chart(px.pie(pd.DataFrame({"Part": ["Principal", "Interest"], "Amount": [principal, max(interest, 0)]}),
                     names="Part", values="Amount", hole=.55, title="Principal vs interest"))
    with c2:
        r = rate / 12 / 100
        balance, rows = principal, [(0, principal)]
        for m_ in range(1, months + 1):
            balance = balance * (1 + r) - emi
            if m_ % 12 == 0 or m_ == months:
                rows.append((m_, max(balance, 0.0)))
        sched = pd.DataFrame(rows, columns=["Month", "Outstanding balance (₹)"])
        chart(px.line(sched, x="Month", y="Outstanding balance (₹)", title="Outstanding balance over time"))


def page_application(ctx: Context) -> None:
    st.title("📝 Loan Application Form")
    with st.form("application_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Applicant name *")
        with c2:
            phone = st.text_input("Mobile number *", placeholder="10-digit number")
        with c3:
            email = st.text_input("Email")
        fields = applicant_fields("app", with_gender=True, coapp_default=0.0)
        submitted = _stretch_call(st.form_submit_button, "💾 Submit Application", type="primary")

    if submitted:
        errors = validate_contact(name, phone, email) + validate_money(fields["inputs"])
        if errors:
            for e in errors:
                st.error(e)
        else:
            result = run_prediction(ctx, fields["inputs"])
            app_id = f"APP-{datetime.now():%y%m%d}-{uuid.uuid4().hex[:5].upper()}"
            record = {"Application_ID": app_id, "Submitted_At": result["generated_at"],
                      "Applicant_Name": name.strip(), "Phone": phone.strip(), "Email": email.strip(),
                      "Gender": fields["gender"], **fields["inputs"],
                      "Model_Screening": result["prediction"],
                      "Approval_Probability_Percent": round(result["approval"], 2),
                      "Eligibility_Score": result["score"]}
            saved = append_row(APPLICATION_FILE, record, APPLICATION_COLUMNS)
            st.session_state["last_result"] = result
            st.session_state["last_application"] = {
                "record": record, "saved": saved,
                "csv": pd.DataFrame([record]).reindex(columns=APPLICATION_COLUMNS).to_csv(index=False).encode()}

    app = st.session_state.get("last_application")
    if app:
        rec = app["record"]
        if app["saved"]:
            st.success(f"Application **{rec['Application_ID']}** saved successfully.")
        st.info(f"Preliminary model screening: **{rec['Model_Screening']}** "
                f"({rec['Approval_Probability_Percent']:.1f}% approval probability). "
                "This is not a final lending decision.")
        _stretch_call(st.download_button, "⬇️ Download this application", data=app["csv"],
                      file_name=f"{rec['Application_ID']}.csv", mime="text/csv")


def _contains(series: pd.Series, term: str) -> pd.Series:
    return series.astype(str).str.contains(term, case=False, na=False, regex=False)


def page_search(ctx: Context) -> None:
    st.title("🔍 Search Applicant")
    term = st.text_input("Search by Loan ID, Application ID, name, phone or email").strip()
    tab1, tab2 = st.tabs(["Dataset records", "Submitted applications"])
    with tab1:
        if term:
            res = ctx.view[_contains(ctx.view[ID_COL], term)]
            if res.empty:
                st.warning("No matching dataset record.")
            else:
                st.success(f"Found {len(res)} matching record(s).")
                show_df(res)
        else:
            st.caption("Type a Loan ID such as LP001002.")
    with tab2:
        apps = read_records(APPLICATION_FILE)
        if apps.empty:
            st.caption("No applications have been submitted yet.")
        elif term:
            cols = [c for c in ["Application_ID", "Applicant_Name", "Phone", "Email"] if c in apps.columns]
            mask = pd.concat([_contains(apps[c], term) for c in cols], axis=1).any(axis=1)
            res = apps[mask]
            if res.empty:
                st.warning("No matching application.")
            else:
                st.success(f"Found {len(res)} matching application(s).")
                show_df(res)
        else:
            st.caption(f"{len(apps)} application(s) stored. Enter a search term to filter.")


def page_explanation(ctx: Context) -> None:
    st.title("🧠 Prediction Explanation")
    last = need_prediction()
    if not last:
        return
    st.subheader(f"Prediction: {last['prediction']}")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Approval probability", f"{last['approval']:.2f}%")
    with c2:
        st.metric("Eligibility score", f"{last['score']}/100")
    expl = last["explanation"]
    explanation_lines(expl)
    plot = expl.sort_values("Impact (pp)")
    fig = px.bar(plot, x="Impact (pp)", y="Feature", orientation="h",
                 color=plot["Impact (pp)"] >= 0, color_discrete_map={True: "#059669", False: "#ea580c"},
                 title="Effect of each factor on approval probability (percentage points)")
    fig.update_layout(showlegend=False)
    chart(fig)
    show_df(expl)


def page_advanced(ctx: Context) -> None:
    v = ctx.view
    st.title("📊 Advanced Analytics")
    c1, c2 = st.columns(2)
    with c1:
        rate = v.groupby("Credit_History")["Approved"].mean().mul(100).reset_index(name="Approval rate (%)")
        rate["Credit_History"] = rate["Credit_History"].map({1.0: "Good (1)", 0.0: "Poor / none (0)"})
        chart(px.bar(rate, x="Credit_History", y="Approval rate (%)", text_auto=".1f",
                     title="Approval rate by credit history"))
    with c2:
        rate = v.groupby("Property_Area")["Approved"].mean().mul(100).reset_index(name="Approval rate (%)")
        chart(px.bar(rate, x="Property_Area", y="Approval rate (%)", text_auto=".1f",
                     title="Approval rate by property area"))
    c1, c2 = st.columns(2)
    with c1:
        chart(px.box(v, x="Status", y="LoanAmount", color="Status", color_discrete_map=STATUS_COLORS,
                     title="Loan amount (₹ thousand) by status"))
    with c2:
        chart(px.scatter(v, x="ApplicantIncome", y="LoanAmount", color="Status",
                         symbol=v["Credit_History"].map({1.0: "Good credit", 0.0: "Poor credit"}),
                         color_discrete_map=STATUS_COLORS, hover_data=["Education", "Property_Area"],
                         title="Income vs loan amount (symbol = credit history)"))
    corr = v[INCOME_LIKE + DISCRETE_NUM + ["Approved"]].corr().round(2)
    chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    title="Correlation matrix (numeric features and approval)"))


def page_quality(ctx: Context) -> None:
    st.title("🧹 Data Quality")
    q = ctx.quality
    a, b, c, d = st.columns(4)
    with a:
        kpi("Original records", f"{q['original']:,}")
    with b:
        kpi("Duplicate rows removed", f"{q['duplicates']:,}")
    with c:
        kpi("Rows without a valid label", f"{q['unlabeled']:,}", "removed")
    with d:
        kpi("Clean records", f"{q['clean']:,}")
    raw = ctx.raw[[c for c in REQUIRED if c in ctx.raw.columns]]
    missing = raw.isna().sum().reset_index()
    missing.columns = ["Column", "Missing values"]
    missing["Missing %"] = (missing["Missing values"] / max(len(raw), 1) * 100).round(1)
    missing = missing.sort_values("Missing values", ascending=False)
    chart(px.bar(missing.head(12), x="Missing values", y="Column", orientation="h",
                 title="Missing values in the original file"))
    st.caption("Missing feature values are filled inside the model pipeline (median / most frequent), "
               "fitted on training data only.")
    show_df(missing)


def page_dataset(ctx: Context) -> None:
    v = ctx.view
    st.title("🗃️ Dataset Explorer")
    st.caption(f"Showing {len(v):,} cleaned records. Missing values are filled for display only.")
    areas_all = sorted(v["Property_Area"].dropna().unique())
    c1, c2 = st.columns(2)
    with c1:
        status = st.multiselect("Loan status", ["Approved", "Rejected"], default=["Approved", "Rejected"])
    with c2:
        areas = st.multiselect("Property area", areas_all, default=areas_all)
    filtered = v[v["Status"].isin(status) & v["Property_Area"].isin(areas)].drop(columns=["Approved"])
    show_df(filtered)
    _stretch_call(st.download_button, "⬇️ Download filtered dataset", data=filtered.to_csv(index=False).encode(),
                  file_name="filtered_loan_dataset.csv", mime="text/csv")
    history = read_records(HISTORY_FILE)
    if not history.empty:
        st.subheader("🧾 Recent prediction history")
        show_df(history.tail(20).iloc[::-1])
        _stretch_call(st.download_button, "⬇️ Download prediction history",
                      data=history.to_csv(index=False).encode(),
                      file_name="applicant_history.csv", mime="text/csv")


PAGES = {
    "🏠 Dashboard": page_dashboard,
    "👤 Applicant Analysis": page_applicants,
    "💰 Financial Analysis": page_financial,
    "🎯 Model Performance": page_performance,
    "🔮 Loan Prediction": page_prediction,
    "📈 Eligibility Score": page_eligibility,
    "💡 Loan Recommendation": page_recommendation,
    "🏦 EMI Calculator": page_emi,
    "📝 Application Form": page_application,
    "🔍 Search Applicant": page_search,
    "🧠 Prediction Explanation": page_explanation,
    "📊 Advanced Analytics": page_advanced,
    "🧹 Data Quality": page_quality,
    "🗃️ Dataset": page_dataset,
}


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="SmartLoan | Loan Prediction", page_icon="💳", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("# 💳 SmartLoan")
        st.caption("Loan Prediction Data Analysis System")
        st.markdown("---")
        page = st.radio("Modules", list(PAGES), label_visibility="collapsed")
        st.markdown("---")
        upload = st.file_uploader("📥 Upload loan dataset (CSV)", type="csv")

        raw = load_raw(upload)
        data = clean_data(raw)
        problem = check_trainable(data)
        if problem:
            st.error(problem)
            st.stop()
        model = train_model(data)
        ctx = Context(raw=raw, data=data, view=impute_for_analysis(data), model=model,
                      quality=quality_stats(raw, data))

        st.markdown("---")
        st.caption(f"Dataset: {len(data):,} records")
        st.caption("Model: Random Forest")
        st.caption(f"Test accuracy: {model['accuracy'] * 100:.1f}%")

    PAGES[page](ctx)


if __name__ == "__main__":
    main()