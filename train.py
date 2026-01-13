import os
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

import joblib

# If xgboost isn't installed yet: pip install xgboost
from xgboost import XGBClassifier

DATA_PATH = Path("data/diabetic_data.csv")
ARTIFACTS_DIR = Path("artifacts")
RUN_EDA = True

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def run_minimal_eda(df: pd.DataFrame):
    print("\n===== MINIMAL EDA =====")

    # Dataset overview
    n_rows, n_cols = df.shape
    print(f"Total rows: {n_rows}")
    print(f"Total columns: {n_cols}")

    # Target distribution
    print("\nReadmission distribution (counts):")
    print(df["readmitted"].value_counts())

    print("\nReadmission distribution (percentages):")
    print((df["readmitted"].value_counts(normalize=True) * 100).round(2))

    # Missing value analysis ("?" represents missing)
    missing_count = (df == "?").sum()
    missing_pct = (missing_count / n_rows) * 100

    missing_df = (
        pd.DataFrame({
            "missing_count": missing_count,
            "missing_percentage": missing_pct
        })
        .query("missing_count > 0")
        .sort_values("missing_percentage", ascending=False)
    )

    print("\nTop columns with missing values (count & %):")
    print(missing_df.head(10).round(2))

    print("\n===== END EDA =====\n")


def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    # Replace '?' with NaN
    df = df.replace("?", np.nan)

    # Binary target: early readmission < 30 days
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

    # Drop columns that are IDs or high-leakage / not useful
    # (encounter_id is a unique visit identifier; patient_nbr can appear multiple times -> leakage risk)
    drop_cols = [
        "readmitted",      # original target label (keep only binary target)
        "encounter_id",
        "patient_nbr",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop columns with heavy missingness in this dataset > ~30%
    # (weight, payer_code, medical_specialty often have many missing values)
    heavy_missing = ["weight", "payer_code", "medical_specialty"]
    df = df.drop(columns=[c for c in heavy_missing if c in df.columns])

    return df


def split_xy(df: pd.DataFrame):
    y = df["readmitted_binary"].astype(int)
    X = df.drop(columns=["readmitted_binary"])
    return X, y


def build_preprocessors(X: pd.DataFrame):

    # Force these integer-coded IDs(category codes) to be treated as categorical
    categorical_id_cols = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Move the *_id columns from numeric -> categorical
    for c in categorical_id_cols:
        if c in numeric_cols:
            numeric_cols.remove(c)
        if c in X.columns and c not in categorical_cols:
            categorical_cols.append(c)
    

    # Pipeline for linear/KNN/NB: impute + one-hot; scale numeric
    numeric_pipe_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe_ohe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # set min_frequency to avoid too many sparse columns
            ("onehot", OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=50)),
        ]
    )
    preprocessor_ohe = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe_scaled, numeric_cols),
            ("cat", categorical_pipe_ohe, categorical_cols),
        ],
        remainder="drop",
    )

    # Pipeline for tree/RF/XGB: ordinal encode categories; numeric impute only (no scaling needed)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe_ord = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessor_ord = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe_ord, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor_ohe, preprocessor_ord, numeric_cols, categorical_cols

def evaluate_binary(y_true, y_pred, y_proba):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    return metrics

def get_models():
    # Reasonable defaults for an assignment; we’ll tune later if you want.
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=None,
            class_weight="balanced",  # helps imbalance
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=None,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=15, n_jobs=-1
        ),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }
    return models

def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    df_raw = load_data()

    if RUN_EDA:
        run_minimal_eda(df_raw)

    df = preprocess_base(df_raw)

    # Target distribution
    print("\nReadmission distribution (counts):")
    print(df["readmitted_binary"].value_counts())

    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # For KNN, limit training size to reduce memory/time
    X_knn, _, y_knn, _ = train_test_split(
        X_train,
        y_train,
        train_size=15000,  # try 10k–20k
        stratify=y_train,
        random_state=42,
    )

    # Reconstruct test dataframe with original columns + label
    test_df = df_raw.loc[X_test.index].copy()

    # Save test CSV for later verification/testing
    TEST_CSV_PATH = ARTIFACTS_DIR / "test_data.csv"
    test_df.to_csv(TEST_CSV_PATH, index=False)

    print(f"\n✅ Test dataset saved for later verification/testing: {TEST_CSV_PATH}")
    print(f"   Rows: {test_df.shape[0]}, Columns: {test_df.shape[1]}")

    pre_ohe, pre_ord, num_cols, cat_cols = build_preprocessors(X_train)

    print("✅ Preprocessing ready")
    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
    print("Target distribution (train):")
    print(y_train.value_counts(normalize=True))

    print("\nNumeric cols:", len(num_cols), "Categorical cols:", len(cat_cols))
    print("Example numeric cols:", num_cols[:11])    
    print("Example categorical cols:", cat_cols[:33])

    models = get_models()
    results = []
    saved = {}

    for name, model in models.items():
        print(f"\n=== Training: {name} ===")

        # Choose preprocessor by model family
        if name in ["logistic_regression", "knn", "naive_bayes"]:
            pre = pre_ohe
        else:
            pre = pre_ord

        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        if name == "knn":
            pipe.fit(X_knn, y_knn)
        else:
            pipe.fit(X_train, y_train)
        
        # Predictions + probabilities
        y_pred = pipe.predict(X_test)

        # Some models may provide decision_function; we standardize to proba for AUC
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # fallback (rare): convert decision scores to [0,1] via sigmoid
            scores = pipe.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-scores))

        metrics = evaluate_binary(y_test, y_pred, y_proba)
        metrics["model"] = name
        results.append(metrics)

        # Save for Streamlit
        model_path = ARTIFACTS_DIR / f"{name}.joblib"
        joblib.dump(pipe, model_path,compress=3)
        saved[name] = str(model_path)

        print("Metrics:", {k: round(v, 4) for k, v in metrics.items() if k != "model"})
        print("Saved:", model_path)

    metrics_df = pd.DataFrame(results).set_index("model").sort_values("auc", ascending=False)
    metrics_csv = ARTIFACTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv)

    print("\n\n===== FINAL METRICS (sorted by AUC) =====")
    print(metrics_df.round(4))
    print("\nSaved metrics to:", metrics_csv)
    print("Saved models:")
    for k, v in saved.items():
        print(f" - {k}: {v}")
    for f in Path("artifacts").glob("*.joblib"):
        print(f.name, os.path.getsize(f)/1024/1024, "MB")   


if __name__ == "__main__":
    main()
