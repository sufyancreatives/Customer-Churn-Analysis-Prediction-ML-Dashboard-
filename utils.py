# utils.py — Helper functions for Telco Churn Dashboard
# Data loading · Preprocessing · Training · Evaluation

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)

# Optional: SMOTE & XGBoost
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    # pyrefly: ignore [missing-import]
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join("models", "best_churn_model.pkl")

FALLBACK_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "SeniorCitizen"
]
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"

PALETTE = {
    "primary":  "#bd00ff",   # Neon Purple
    "danger":   "#ff007f",   # Neon Pink / Churn indicator
    "success":  "#00f0ff",   # Neon Cyan / Active indicator
    "purple":   "#8b5cf6",   # Electric Violet
    "yellow":   "#e0aaff",   # Pale Purple / Highlights
    "bg":       "#06020f",   # Deep Space Purple Background
    "card":     "#120924",   # Glass-panel Purple
    "muted":    "#9d8fb3",   # Violet Muted Text
}

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load and clean the Telco dataset from disk or fallback URL."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        try:
            df = pd.read_csv(FALLBACK_URL)
        except Exception:
            df = _synthetic_data()

    df = _clean(df)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps in one place."""
    df = df.copy()

    # TotalCharges → numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # SeniorCitizen 0/1 → No/Yes
    if df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Churn → binary
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    # Derived features
    df["Tenure_Group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
    )
    df["Charge_Tier"] = pd.cut(
        df["MonthlyCharges"], bins=[0, 35, 65, 95, 200],
        labels=["Low (<$35)", "Mid ($35-65)", "High ($65-95)", "Premium (>$95)"]
    )
    return df


def _synthetic_data() -> pd.DataFrame:
    """Generate realistic synthetic fallback data."""
    np.random.seed(42)
    n = 7043
    return pd.DataFrame({
        "customerID":      [f"ID-{i:05d}" for i in range(n)],
        "gender":          np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":   np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":         np.random.choice(["Yes", "No"], n),
        "Dependents":      np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure":          np.random.randint(1, 73, n),
        "PhoneService":    np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "MultipleLines":   np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":  np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":    np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection":np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":     np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":     np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":        np.random.choice([
            "Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]
        ),
        "PaperlessBilling":np.random.choice(["Yes", "No"], n),
        "PaymentMethod":   np.random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges":  np.round(np.random.uniform(18, 119, n), 2),
        "TotalCharges":    [str(round(np.random.uniform(18, 8500), 2)) for _ in range(n)],
        "Churn":           np.random.choice(["Yes", "No"], n, p=[0.265, 0.735]),
    })

# ─────────────────────────────────────────────
#  PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def build_preprocessor():
    """Return a ColumnTransformer for num + cat features."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERICAL_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def get_X_y(df: pd.DataFrame):
    """Return features DataFrame and binary target series."""
    feat_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].copy()
    y = df["Churn_Binary"].copy()
    return X, y

# ─────────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────────
def get_model_registry() -> dict:
    """Return dict of {name: estimator}."""
    registry = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.5, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, random_state=42
        ),
    }
    if XGB_AVAILABLE:
        registry["XGBoost"] = XGBClassifier(
            n_estimators=150, learning_rate=0.08, use_label_encoder=False,
            eval_metric="logloss", random_state=42, scale_pos_weight=3
        )
    return registry

# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train_model(df: pd.DataFrame, model_name: str, test_size: float = 0.2):
    """
    Full training pipeline with stratified split and SMOTE (if available).
    Returns (metrics_dict, cm, fpr, tpr, precision_vals, recall_vals, pipeline).
    """
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()
    model_registry = get_model_registry()
    estimator = model_registry[model_name]

    if SMOTE_AVAILABLE:
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", estimator),
        ])
    else:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)

    return metrics, cm, fpr, tpr, prec_vals, rec_vals, pipeline

# ─────────────────────────────────────────────
#  MODEL LOAD / SAVE
# ─────────────────────────────────────────────
def load_saved_model():
    """Load .pkl model from disk, return None if not found."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def save_model(pipeline, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)

# ─────────────────────────────────────────────
#  PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_single(pipeline, input_dict: dict) -> tuple[str, float]:
    """
    Given a fitted pipeline and an input dict, return (label, probability).
    label: 'Churn' | 'No Churn'
    """
    row = pd.DataFrame([input_dict])
    proba = pipeline.predict_proba(row)[0][1]
    label = "Churn" if proba >= 0.5 else "No Churn"
    return label, float(proba)

# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE EXTRACTOR
# ─────────────────────────────────────────────
def get_feature_importance(pipeline, top_n: int = 10) -> pd.DataFrame:
    """
    Extract feature importances from a fitted pipeline.
    Works with tree-based models and Logistic Regression.
    """
    try:
        model_step = pipeline.named_steps.get("model")
        pre_step   = pipeline.named_steps.get("preprocessor")

        if model_step is None or pre_step is None:
            return pd.DataFrame()

        # Get feature names after OHE
        num_names = NUMERICAL_FEATURES.copy()
        cat_names = []
        for name, trans, cols in pre_step.transformers_:
            if name == "cat" and hasattr(trans.named_steps["ohe"], "get_feature_names_out"):
                cat_names = list(trans.named_steps["ohe"].get_feature_names_out(CATEGORICAL_FEATURES))
        all_names = num_names + cat_names

        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
        elif hasattr(model_step, "coef_"):
            importances = np.abs(model_step.coef_[0])
        else:
            return pd.DataFrame()

        n = min(len(importances), len(all_names))
        feat_df = pd.DataFrame({
            "Feature": all_names[:n],
            "Importance": importances[:n],
        }).sort_values("Importance", ascending=False).head(top_n)

        return feat_df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────
#  METRICS SUMMARY TABLE
# ─────────────────────────────────────────────
def build_leaderboard(results: dict) -> pd.DataFrame:
    """
    results = {model_name: metrics_dict}
    Returns sorted DataFrame with rank column.
    """
    rows = []
    for name, m in results.items():
        rows.append({"Model": name, **m})
    df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", ["🥇", "🥈", "🥉"] + [""] * max(0, len(df) - 3))
    return df
# KPI computation helper

def compute_kpis(df: pd.DataFrame) -> dict:
    """Calculate high‑level KPIs for the dashboard.
    Returns a dictionary with:
        - total_customers
        - churn_rate (percentage)
        - avg_monthly_charges
        - avg_tenure (months)
    """
    total_customers = df.shape[0]
    # Ensure churn binary exists
    churn_col = "Churn_Binary" if "Churn_Binary" in df.columns else "Churn"
    churn_series = df["Churn_Binary"] if churn_col == "Churn_Binary" else (df["Churn"] == "Yes").astype(int)
    churn_rate = round(churn_series.mean() * 100, 2)
    avg_monthly_charges = round(df["MonthlyCharges"].mean(), 2)
    avg_tenure = round(df["tenure"].mean(), 2)
    return {
        "total_customers": total_customers,
        "churn_rate": churn_rate,
        "avg_monthly_charges": avg_monthly_charges,
        "avg_tenure": avg_tenure,
    }

NUMERIC_FEATURES = NUMERICAL_FEATURES
XGBOOST_AVAILABLE = XGB_AVAILABLE

# Compatibility alias: older code expects get_feature_target
def get_feature_target(df: pd.DataFrame):
    """Alias for get_X_y for backward compatibility."""
    return get_X_y(df)

# Train all models helper – returns a dict of model name → (metrics, model pipeline)
def train_all_models(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Train every model in the registry and collect results.
    Returns a dict mapping model names to a tuple of (metrics_dict, pipeline).
    """
    results = {}
    for name in get_model_registry().keys():
        metrics, _, _, _, _, _, pipeline = train_model(df, name, test_size)
        results[name] = (metrics, pipeline)
    return results


