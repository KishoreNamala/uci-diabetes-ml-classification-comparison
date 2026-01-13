import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from styles import inject_global_css

inject_global_css()
METRICS_PATH = Path("artifacts/metrics.csv")

st.set_page_config(page_title="Compare All Models", page_icon="📊", layout="wide")
st.title("📊 Compare All Models")

st.markdown(
    """
This page compares **all six models** trained on the same dataset.

### How to read this
- **Accuracy** can be misleading on imbalanced data.
- Prefer **ROC AUC** (threshold-independent ranking) and **MCC** (robust for imbalance).
"""
)

if not METRICS_PATH.exists():
    st.error("Missing artifacts/metrics.csv. Run train.py to generate it.")
    st.stop()

df = pd.read_csv(METRICS_PATH, index_col=0).copy()

# Normalize naming: support both auc/roc_auc naming
if "roc_auc" in df.columns and "auc" not in df.columns:
    df["auc"] = df["roc_auc"]

# Keep expected columns if they exist
cols = [c for c in ["accuracy", "auc", "precision", "recall", "f1_score", "mcc"] if c in df.columns]
df = df[cols].sort_values(by="auc", ascending=False)

best_auc_model = df["auc"].idxmax()
st.markdown(f"### 🏆 Best ROC AUC: `{best_auc_model}` (**{df.loc[best_auc_model, 'auc']:.4f}**)")

st.markdown("---")
st.markdown("### Model Performance (sorted by ROC AUC)")
st.dataframe(df.round(4), width='stretch')

st.markdown("---")
st.markdown("### ROC AUC comparison (bar chart)")

fig, ax = plt.subplots(figsize=(7.5, 3.5))
ax.bar(df.index.tolist(), df["auc"].tolist())
ax.set_ylabel("ROC AUC")
ax.set_xlabel("Model")
ax.set_title("ROC AUC by Model")
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
st.pyplot(fig, clear_figure=True)
