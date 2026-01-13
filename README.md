# Diabetes Readmission Classification using Machine Learning

## a. Problem Statement

Hospital readmissions are costly and often indicate gaps in post-discharge care.The objective of this project is to predict whether a diabetic patient will be readmitted within 30 days of discharge using historical clinical and administrative data.


The dataset represents ten years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. Each row concerns hospital records of patients diagnosed with diabetes, who underwent laboratory, medications, and stayed up to 14 days. 


This is formulated as a binary classification problem, where the target variable
indicates early readmission (less than 30 days) versus no early readmission.

---

## b. Dataset Description

- Dataset Name: [Diabetes 130-US Hospitals Dataset (1999–2008)]([text](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008))
- Source: UCI Machine Learning Repository
- Number of instances: 101,766 patient encounters
- Number of features: 50 (44 used after preprocessing)

### Target Variable

Original label:
readmitted ∈ {NO, >30, <30}

Binary transformation used in this project:
- 1 : readmitted == "<30" (early readmission)
- 0 : readmitted ∈ {NO, >30}

### Feature Types

- Numeric features: time in hospital, number of lab procedures, number of
  medications, number of inpatient, outpatient, and emergency visits.
- Categorical features: demographics, diagnosis codes, admission details,
  discharge information, and medication indicators.

The dataset is highly imbalanced, with early readmissions forming a minority
class. Therefore, evaluation metrics beyond accuracy are required.

---

## c. Models Used and Evaluation Metrics

All models were trained and evaluated on the same dataset using identical
preprocessing steps.

### Evaluation Metrics

- Accuracy
- ROC AUC
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.6607 | 0.6758 | 0.1808 | 0.5782 | 0.2755 | 0.1638 |
| Decision Tree | 0.8111 | 0.5289 | 0.1617 | 0.1656 | 0.1636 | 0.0571 |
| k-Nearest Neighbors | 0.8883 | 0.5901 | 0.4792 | 0.0101 | 0.0198 | 0.0568 |
| Naive Bayes | 0.2672 | 0.5432 | 0.1194 | 0.8736 | 0.2101 | 0.0525 |
| Random Forest (Ensemble) | 0.8887 | 0.6576 | 0.7143 | 0.0044 | 0.0088 | 0.0502 |
| XGBoost (Ensemble) | 0.8888 | 0.6870 | 0.5946 | 0.0097 | 0.0191 | 0.0655 |

---

## Model-wise Observations

| ML Model | Observation about model performance |
|---------|------------------------------------|
| Logistic Regression | Achieved the best balance between recall and precision among all models. It identified a substantial proportion of early readmissions and performed robustly on this imbalanced dataset, as reflected by the highest MCC. |
| Decision Tree | Showed moderate accuracy but low AUC and MCC, indicating limited generalization ability and sensitivity to class imbalance. |
| k-Nearest Neighbors | Achieved high accuracy due to dominance of the majority class but exhibited extremely poor recall, making it ineffective for detecting early readmissions. |
| Naive Bayes | Demonstrated very high recall but low precision, resulting in many false positives. This behavior is consistent with the strong feature independence assumption of the model. |
| Random Forest (Ensemble) | Produced high overall accuracy but near-zero recall at the default threshold, indicating a strong bias toward the majority class without threshold tuning. |
| XGBoost (Ensemble) | Achieved the highest ROC AUC, indicating strong ranking capability. However, recall remained low at the default threshold, suggesting that threshold optimization is necessary to improve sensitivity. |

---
## Project Structure

```
uci-diabetes-ml-classification-comparison/
├── .github/
│   └── workflows/
│       └── deploy.yml            # GitHub Actions workflow
├── .streamlit/
│   └── config.toml               # Streamlit app configuration
├── app.py                        # Main Streamlit application
├── artifacts/                    # Directory for artifacts - models, metrics and test data
│   ├── metrics.csv               # Evaluation metrics
│   ├── test_data.csv             # Test dataset
│   └── *.joblib                  # Trained model files
├── data/
│   └── diabetic_data.csv         # Input dataset
├── download.py                   # Dataset download module
├── pages/
│   ├── 1-overview.py             # App overview
│   ├── 2-explore.py              # Model exploration
│   └── 3-compare.py              # Comparison of models
├── README.md                     # README file
├── requirements.txt              # Python dependencies
└── train.py                      # Pre-processing & model training
|__ styles.py                     # Streamlit App Styles (HTML/CSS)
```
---

## Setup

1. Install python 3.11 or higher
2. Clone the repository
   ```bash
    cd uci-diabetes-ml-classification-comparison
   ```
3. Install python packages
    ```bash
    pip install -r requirements.txt
    ```
4. Download the dataset
   ```bash
   python download.py
   ```
5. Preprocess and train the models.
   ```bash
   python train.py
   ```
6. Run the streamlit app
   ```bash
   streamlit run app.py
   ```
---

## Deployment (on Streamlit Community Cloud)

1. Push the complete project repository to GitHub.
2. Ensure the following files and folders are present:
   - app.py
   - pages/
   - requirements.txt
   - artifacts/ (trained models and metrics)
3. Go to https://streamlit.io/cloud
4. Click "New App" and connect your GitHub repository.
5. Select app.py as the entry point.
6. Deploy the app and obtain the public URL.

---

## Author

Kishore Namala

## License
This project is developed for educational purposes as part of a college assignment.

## References

1. UCI Machine Learning Repository. Diabetes 130-US Hospitals Dataset.
   https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
   Journal of Machine Learning Research.
3. Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
   Proceedings of the 22nd ACM SIGKDD Conference.
4. BITS Pilani Assignment
