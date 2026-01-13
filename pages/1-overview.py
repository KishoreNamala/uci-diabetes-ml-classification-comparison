import pandas as pd
import streamlit as st
from pathlib import Path
from styles import inject_global_css

DATA_PATH = Path("data/diabetic_data.csv")

st.set_page_config(page_title="Overview of Dataset", page_icon="📘", layout="wide")
inject_global_css()

st.title("📘 Overview of Dataset")

st.markdown(
"""
### Problem Statement
The goal is to determine the early readmission of the patient within 30 days of discharge. The problem is important for the following reasons. Despite high-quality evidence showing improved clinical outcomes for diabetic patients who receive various preventive and therapeutic interventions, many patients do not receive them. This can be partially attributed to arbitrary diabetes management in hospital environments, which fail to attend to glycemic control. Failure to provide proper diabetes care not only increases the managing costs for the hospitals (as the patients are readmitted) but also impacts the morbidity and mortality of the patients, who may face complications associated with diabetes. 

### Label definition
Original label: `readmitted ∈ {NO, >30, <30}`  
Binary label used in this project:
- **1** → `readmitted == "<30"`
- **0** → `readmitted in {NO, >30}`

""",unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Task", "Binary Classification")

st.markdown("### Target Distribution (Original)")
dist = df["readmitted"].value_counts()
dist_pct = (df["readmitted"].value_counts(normalize=True) * 100).round(2)
dist_df = pd.DataFrame({"count": dist, "percent": dist_pct})
st.dataframe(dist_df, width='stretch')

st.markdown(
    """
### Why metrics beyond accuracy?
This dataset is **imbalanced** (early readmission is relatively rare).  
So we report:
- **ROC AUC** (ranking quality across thresholds)
- **MCC** (robust for imbalance)
- Precision/Recall/F1 (class-specific performance)
"""
)

st.markdown('<div class="section-title">Dataset Columns Overview</div>', unsafe_allow_html=True)

# Identify column types
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Make both lists the same length for tabular display
max_len = max(len(numeric_cols), len(categorical_cols))
numeric_cols += [""] * (max_len - len(numeric_cols))
categorical_cols += [""] * (max_len - len(categorical_cols))

col_table = pd.DataFrame({
    "Numeric Columns": numeric_cols,
    "Categorical Columns": categorical_cols
})

with st.expander("View numeric vs categorical columns", expanded=True):
    st.dataframe(col_table, width='stretch', height=420)

with st.expander("View full column list"):
    st.write(sorted(df.columns.tolist()))

st.success("Next: go to **Explore a Model** to upload test CSV and evaluate models.")
