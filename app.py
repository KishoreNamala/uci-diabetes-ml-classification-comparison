import streamlit as st
from styles import inject_global_css

st.set_page_config(page_title="Diabetes Readmission App", page_icon="🩺", layout="wide")

inject_global_css()
st.title("🩺 Diabetes Readmission Classification App")
st.markdown(
"""
<div class="body-text">
Welcome! This Streamlit app demonstrates multiple ML classification models trained on the
**UCI Diabetes 130-US Hospitals (1999–2008)** dataset to predict **early readmission (<30 days)**.

The Diabetes 130-US Hospitals dataset was obtained from the UCI Machine Learning Repository [1].
Six classification models were implemented using scikit-learn and XGBoost [2][3].
Model performance was evaluated using Accuracy, ROC AUC, Precision, Recall, F1-score, and Matthews
Correlation Coefficient (MCC) [4].


### What you can do
- **Overview of Dataset**: Understand the problem + target definition + class imbalance
- **Explore a Model**: Upload *test CSV*, select a model, view metrics + confusion matrix/report
- **Compare All Models**: Leaderboard-style comparison across all 6 trained models

➡️ Use the **left sidebar** to navigate between pages.
</div>

### References
<div class="small-note">

[1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

[2] Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, pp. 2825-2830, 2011.

[3] T. Chen and C. Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, pages 785–794, New York, NY, USA, 2016. ACM.

[4] Chicco, D. and Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics 21, 6 (2020).
</div>
""",
unsafe_allow_html=True
)

st.info("Tip: Upload `artifacts/test_data.csv` in the **Explore a Model** page for a reliable demo.")
