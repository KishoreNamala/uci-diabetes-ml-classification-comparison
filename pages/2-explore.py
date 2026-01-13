import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_curve
)
from styles import inject_global_css

inject_global_css()

ARTIFACTS_DIR = Path("artifacts")

MODEL_FILES = {
    "Logistic Regression": ARTIFACTS_DIR / "logistic_regression.joblib",
    "Decision Tree": ARTIFACTS_DIR / "decision_tree.joblib",
    "KNN": ARTIFACTS_DIR / "knn.joblib",
    "Naive Bayes": ARTIFACTS_DIR / "naive_bayes.joblib",
    "Random Forest": ARTIFACTS_DIR / "random_forest.joblib",
    "XGBoost": ARTIFACTS_DIR / "xgboost.joblib",
}

st.set_page_config(page_title="Explore a Model", page_icon="🧪", layout="wide")
st.title("🧪 Explore a Model ")



st.markdown(
    """
**Sequence of steps:**  

:one: Upload test dataset (CSV)  
:two: Select a trained model  
:three: View evaluation metrics + confusion matrix + classification report + ROC curve

**Recommended upload file:** `artifacts/test_data.csv`
"""
)

def make_binary_target(df: pd.DataFrame) -> pd.Series:
    if "readmitted" not in df.columns:
        raise ValueError("Uploaded CSV must include a 'readmitted' column.")
    return (df["readmitted"] == "<30").astype(int)

def compute_metrics(y_true, y_pred, y_proba):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        out["ROC AUC"] = roc_auc_score(y_true, y_proba)
    except Exception:
        out["ROC AUC"] = np.nan
    return out

def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(1, 1),dpi=130)
    ax.imshow(cm, cmap="Blues")
    # ax.set_title("Confusion Matrix",fontsize=11)
    ax.set_xlabel("Predicted",fontsize=6)
    ax.set_ylabel("Actual",fontsize=6)

    ax.set_xticks([0, 1],)
    ax.set_yticks([0, 1],)

    # Reduce tick label size
    ax.set_xticklabels(["0", "1"], fontsize=6)
    ax.set_yticklabels(["0", "1"], fontsize=6)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center",fontsize=6)
    plt.tight_layout(pad=0.2)
    return fig

def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(2.8, 2.0),dpi=130)
    ax.plot(fpr, tpr,linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--",linewidth=1)
    # ax.set_title("ROC Curve",fontsize=11)
    ax.set_xlabel("False Positive Rate",fontsize=6)
    ax.set_ylabel("True Positive Rate",fontsize=6)
    ax.tick_params(axis="both", labelsize=6)
    plt.tight_layout(pad=0.2)
    return fig

@st.cache_data
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(file_bytes)).replace("?", np.nan)


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

# Sidebar controls
st.sidebar.markdown("### Controls")

st.sidebar.header(":one:) Upload test CSV")
uploaded = st.sidebar.file_uploader("CSV (test data only)", type=["csv"])

st.sidebar.header(":two:) Choose model")
model_label = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()), index=5)

st.sidebar.caption("Adjust the threshold to explore precision–recall trade-offs.")
threshold = st.sidebar.slider("Threshold for class 1 (<30)", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("Lower threshold → higher Recall, often lower Precision.")

if uploaded is None:
    st.info("Upload a CSV to proceed (recommended: artifacts/test_data.csv).")
    st.stop()

file_bytes = uploaded.getvalue()
df = load_uploaded_csv(file_bytes)

y_true = make_binary_target(df)
X = df.drop(columns=["readmitted"])

model_path = MODEL_FILES[model_label]
if not model_path.exists():
    st.error(f"Missing model file: {model_path}. Run train.py first.")
    st.stop()

pipe = load_model(model_path)

if hasattr(pipe, "predict_proba"):
    y_proba = pipe.predict_proba(X)[:, 1]
else:
    scores = pipe.decision_function(X)
    y_proba = 1 / (1 + np.exp(-scores))

y_pred = (y_proba >= threshold).astype(int)
metrics = compute_metrics(y_true, y_pred, y_proba)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"Model: {model_label} - Performance ")

r1 = st.columns(3)
r1[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
r1[1].metric("ROC AUC", "N/A" if np.isnan(metrics["ROC AUC"]) else f"{metrics['ROC AUC']:.4f}")
r1[2].metric("MCC", f"{metrics['MCC']:.4f}")

r2 = st.columns(3)
r2[0].metric("Precision", f"{metrics['Precision']:.4f}")
r2[1].metric("Recall", f"{metrics['Recall']:.4f}")
r2[2].metric("F1 Score", f"{metrics['F1 Score']:.4f}")

# Layout
left, right = st.columns(2)

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.pyplot(plot_confusion(cm), clear_figure=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ROC Curve")
    st.pyplot(plot_roc(y_true, y_proba), clear_figure=True)
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("📋 Classification Report (Detailed)"):
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    styled_report = (
    report_df
    .style
    .set_table_styles([
        {
            "selector": "th",
            "props": [
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("background-color", "#f5f5f5"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("text-align", "center"),
            ],
        },
    ])
    .format("{:.4f}")
    )
    st.table(styled_report)
    # st.dataframe(report_df, width='stretch')

with st.expander("🔍 Preview uploaded data"):
    st.dataframe(df.head(25), width='stretch')
