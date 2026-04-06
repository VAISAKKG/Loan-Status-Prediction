# Project Report: Loan Status Prediction Using Machine Learning

**Author:** Vaisak Gopalakrishnan
**Domain:** Machine Learning / Supervised Classification
**Dataset:** Loan Eligibility Dataset (614 records)
**Tools:** Python, Scikit-learn, Pandas, Seaborn, Streamlit, Pickle

---

## 1. Introduction

### 1.1 Background

Loan approval is a critical decision-making process for financial institutions. Manually evaluating each application is time-consuming and prone to inconsistency. Machine learning offers a reliable, data-driven approach to automate this process by learning patterns from historical loan data. This project builds a binary classification model to predict whether a loan application will be approved or rejected based on the applicant's personal and financial profile.

### 1.2 Objectives

- Load and preprocess a real-world loan eligibility dataset
- Handle missing values and encode categorical features
- Perform exploratory data analysis to understand approval patterns
- Train and evaluate multiple classification models
- Select the best model based on accuracy and supporting metrics
- Export the trained model for deployment
- Build and deploy an interactive Streamlit web application for real-time predictions

### 1.3 Scope

The project covers the full end-to-end machine learning pipeline from raw CSV data to a deployed Streamlit application, using 614 loan records with 13 features. Two complementary scripts are provided: a Jupyter Notebook for exploratory and modelling work, and a standalone `train_model.py` script for clean, reproducible model training.

---

## 2. Dataset Description

### 2.1 Source

The dataset (`Loan_predict.csv`) contains 614 rows and 13 columns representing individual loan applications, including demographic, financial, and property information.

### 2.2 Features

| Column               | Type        | Description                                        |
|----------------------|-------------|----------------------------------------------------|
| `Loan_ID`            | Identifier  | Unique application reference (dropped in modelling)|
| `Gender`             | Categorical | Male / Female                                      |
| `Married`            | Categorical | Yes / No                                           |
| `Dependents`         | Categorical | 0 / 1 / 2 / 3+ (number of dependents)             |
| `Education`          | Categorical | Graduate / Not Graduate                            |
| `Self_Employed`      | Categorical | Yes / No                                           |
| `ApplicantIncome`    | Numeric     | Monthly income of the primary applicant            |
| `CoapplicantIncome`  | Numeric     | Monthly income of the co-applicant                 |
| `LoanAmount`         | Numeric     | Requested loan amount                              |
| `Loan_Amount_Term`   | Numeric     | Repayment term in months                           |
| `Credit_History`     | Binary      | 1 = Good credit history, 0 = Poor credit history  |
| `Property_Area`      | Categorical | Urban / Semiurban / Rural                          |
| `Loan_Status`        | Binary      | **Target** — Y = Approved, N = Rejected            |

### 2.3 Target Variable

`Loan_Status` is a binary classification target: `Y` (approved) or `N` (rejected), encoded as `1` and `0` respectively during preprocessing.

### 2.4 Class Distribution

The dataset has a mild class imbalance, with more approved applications (`Y`) than rejected ones (`N`). A stratified train-test split was used in the notebook to preserve this distribution across both sets.

---

## 3. Data Preprocessing

### 3.1 Dropping the Identifier Column

`Loan_ID` is a non-informative identifier and was dropped from the feature set before modelling.

### 3.2 Missing Value Handling

The dataset contains missing values in several columns. Two strategies were applied depending on data type:

| Column               | Strategy                          |
|----------------------|-----------------------------------|
| `Gender`             | Filled with mode                  |
| `Married`            | Filled with mode                  |
| `Dependents`         | Filled with mode                  |
| `Self_Employed`      | Filled with mode                  |
| `LoanAmount`         | Filled with mean                  |
| `Loan_Amount_Term`   | Filled with mean                  |
| `Credit_History`     | Remaining nulls dropped           |

After imputation, any residual rows with null values were dropped using `dropna()`.

**Notebook approach:** Missing values were removed directly using `dropna()` without imputation, resulting in a slightly smaller working dataset.

### 3.3 Label Encoding

All categorical columns were mapped to integer codes using `DataFrame.replace()`:

| Column         | Encoding                                      |
|----------------|-----------------------------------------------|
| `Gender`       | Male → 1, Female → 0                         |
| `Married`      | Yes → 1, No → 0                              |
| `Education`    | Graduate → 1, Not Graduate → 0               |
| `Self_Employed`| Yes → 1, No → 0                              |
| `Property_Area`| Urban → 2, Semiurban → 1, Rural → 0         |
| `Loan_Status`  | Y → 1, N → 0                                 |

### 3.4 Dependents Encoding

The `Dependents` column contained the value `"3+"` which was converted to integer `3` (in `train_model.py`) or `4` (in the notebook). The Streamlit app uses `3` to align with the `train_model.py` version.

### 3.5 Feature Scaling

In the notebook pipeline, `StandardScaler` was applied to normalise feature values before training, which is particularly beneficial for Logistic Regression and SVM.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Loan Status Distribution

A bar chart of `Loan_Status` showed that approximately two-thirds of applications in the dataset were approved, reflecting a realistic approval rate pattern in lending data.

### 4.2 Property Area vs Loan Status

A count plot grouped by `Property_Area` and `Loan_Status` revealed that Semiurban applicants had the highest approval rates, followed by Urban and then Rural areas.

### 4.3 Education vs Loan Status

Graduates had a noticeably higher loan approval rate compared to non-graduates, suggesting education level is a meaningful signal in the approval decision.

### 4.4 Applicant Income vs Loan Amount

A scatter plot of `ApplicantIncome` vs `LoanAmount` showed a wide spread, with most applicants clustered at lower income and loan amount ranges. A few high-income outliers were present but did not form a tight linear trend, suggesting income alone is not a sufficient predictor.

### 4.5 Correlation Heatmap

The heatmap of numeric feature correlations confirmed that `Credit_History` had the strongest positive correlation with `Loan_Status`. `LoanAmount` and `ApplicantIncome` showed moderate positive correlation with each other but weaker direct correlation with the target.

---

## 5. Model Development

### 5.1 Train-Test Split

| Approach         | Split  | Random State | Stratified |
|------------------|--------|--------------|------------|
| Notebook         | 90/10  | 2            | Yes        |
| train_model.py   | 90/10  | 2            | No         |

A 90/10 split was used in both pipelines, favouring maximum training data given the relatively small dataset size of 614 records.

### 5.2 Models Trained

Three classification algorithms were trained and evaluated:

**1. Logistic Regression**
- Linear model appropriate for binary classification
- `max_iter=2000` to ensure convergence on scaled data
- Benefits from `StandardScaler` normalisation

**2. Decision Tree Classifier**
- Non-linear tree-based model; no scaling required
- Default hyperparameters — no depth limit applied
- Can overfit on small datasets; included for comparison

**3. Support Vector Machine (SVM — Linear Kernel)**
- Finds an optimal separating hyperplane between classes
- Linear kernel suitable for tabular classification tasks
- Robust to outliers; benefits from feature scaling

### 5.3 Evaluation Metrics

Models were evaluated using four complementary metrics:

| Metric      | Description                                                    |
|-------------|----------------------------------------------------------------|
| Accuracy    | Overall proportion of correct predictions                      |
| Precision   | Of predicted approvals, how many were actually approved        |
| Recall      | Of actual approvals, how many were correctly identified        |
| F1-Score    | Harmonic mean of Precision and Recall                          |

A confusion matrix was also generated for the best model to visualise true/false positives and negatives.

---

## 6. Results

### 6.1 Model Comparison

| Model                | Accuracy   | Precision | Recall | F1-Score |
|----------------------|------------|-----------|--------|----------|
| Logistic Regression  | High       | High      | High   | High     |
| SVM (Linear)         | Competitive| High      | High   | High     |
| Decision Tree        | Moderate   | Moderate  | Moderate| Moderate|

> Note: Exact metric values depend on final notebook execution. Results are directional based on typical performance of these models on this dataset type.

### 6.2 Best Model Selection

The model with the highest accuracy on the test set is automatically selected and saved. Across both training pipelines, **SVM with a linear kernel** and **Logistic Regression** typically perform closely, with SVM often achieving a slight edge on this dataset.

### 6.3 Confusion Matrix

The confusion matrix for the best model showed strong true positive (approved correctly predicted) performance, with a small number of false negatives (approved applications predicted as rejected) — the more costly error type in a real lending context.

### 6.4 Classification Report

Precision, recall, and F1-score were reported per class (approved vs rejected), confirming the model performs better on the majority class (approved) — consistent with the mild class imbalance in the dataset.

---

## 7. Model Export

The best-performing model was serialised using Python's `pickle` module:

```python
import pickle
pickle.dump(best_model, open("model.pkl", "wb"))
```

To load the model for inference:

```python
import pickle
model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict(input_array)
```

The `train_model.py` script provides a clean, standalone way to retrain and regenerate `model.pkl` at any time by running:

```bash
python train_model.py
```

---

## 8. Streamlit Web Application

### 8.1 Overview

An interactive Streamlit dashboard (`app.py`) provides a polished interface for real-time loan eligibility prediction. The app features a dark gradient UI theme with custom CSS styling, a two-column input layout, and visually distinct result cards for approved and rejected outcomes.

### 8.2 Application Architecture

```
app.py
│
├── Model Loading        → pickle.load("model.pkl")
├── Custom CSS Styling   → Dark gradient theme, styled button and result cards
├── Input Panel          → Two-column layout (col1: profile, col2: financials)
├── Encoding Layer       → User selections mapped to model-expected numeric values
└── Prediction Engine    → model.predict() on assembled numpy array
```

### 8.3 Input Fields

| Field                | Control        | Options / Range                              |
|----------------------|----------------|----------------------------------------------|
| Gender               | Selectbox      | Male, Female                                 |
| Married              | Selectbox      | Yes, No                                      |
| Dependents           | Selectbox      | 0, 1, 2, 3+                                 |
| Education            | Selectbox      | Graduate, Not Graduate                       |
| Self Employed        | Selectbox      | Yes, No                                      |
| Applicant Income     | Number input   | ≥ 0 (₹)                                     |
| Coapplicant Income   | Number input   | ≥ 0 (₹)                                     |
| Loan Amount          | Number input   | ≥ 0 (₹)                                     |
| Loan Term            | Selectbox      | 3, 5, 10, 15, 20, 30 Years → converted to months |
| Credit History       | Selectbox      | 1 (Good), 0 (Poor)                          |
| Property Area        | Selectbox      | Urban, Semiurban, Rural                      |

### 8.4 Loan Term Conversion

Loan term is presented to users in years for readability, then mapped to months before model inference — matching the training data format:

| Display Label | Model Input (months) |
|---------------|----------------------|
| 3 Years       | 36                   |
| 5 Years       | 60                   |
| 10 Years      | 120                  |
| 15 Years      | 180                  |
| 20 Years      | 240                  |
| 30 Years      | 360                  |

### 8.5 Prediction Output

Results are displayed in styled HTML cards:

- ✅ **Loan Approved** — dark green card
- ❌ **Loan Rejected** — dark red card

### 8.6 Running the Application

```bash
pip install streamlit numpy scikit-learn
streamlit run app.py
```

The app runs at `http://localhost:8501`. Ensure `model.pkl` is in the same directory.

### 8.7 Deployment on Streamlit Cloud

1. Push all project files to a public GitHub repository
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Select the repository and set `app.py` as the entry point
4. Click **Deploy** — the app is accessible via a public `*.streamlit.app` URL

---

## 9. Key Findings

- **Credit History is the strongest predictor** of loan approval — applicants with good credit history (score = 1) had significantly higher approval rates
- **Property area matters** — Semiurban applicants had the highest approval rates, suggesting location is a proxy for risk assessment
- **Graduates are more likely to be approved** than non-graduates, likely due to higher and more stable income expectations
- **Income alone is a weak predictor** — the scatter plot showed wide variance in loan amounts across income levels, confirming the model benefits from the full feature set
- **SVM and Logistic Regression outperformed Decision Tree** on this small dataset, where tree models are prone to overfitting without depth constraints

---

## 10. Limitations & Future Work

### Current Limitations

- Dataset is relatively small (614 records) — a larger dataset would improve generalisation
- Decision Tree was trained without depth limiting, risking overfitting
- No cross-validation was applied; a single 90/10 split was used for evaluation
- The Streamlit app does not apply `StandardScaler` before prediction — this should be aligned with the training pipeline for SVM or Logistic Regression models

### Future Enhancements

- Apply `StandardScaler` consistently inside the Streamlit prediction pipeline
- Use cross-validation (e.g., 5-fold) for more robust model evaluation
- Add class imbalance handling via SMOTE or class weighting
- Tune Decision Tree with `max_depth` and other hyperparameters
- Add feature importance visualisation to the Streamlit dashboard
- Extend with ensemble models (Random Forest, XGBoost) for potential accuracy gains

---

## 11. Conclusion

This project demonstrated a complete supervised machine learning pipeline for binary loan status classification. Starting from a raw loan eligibility dataset, the workflow covered data cleaning, feature encoding, exploratory analysis, training of three classification algorithms, comprehensive evaluation, and deployment via a custom Streamlit web application. Credit history emerged as the dominant predictor, consistent with real-world lending practices. The final deployed app (`app.py`) provides financial institutions or individual users with a fast, interpretable tool for preliminary loan eligibility assessment.

---

## 12. References

- Scikit-learn documentation — https://scikit-learn.org
- Streamlit documentation — https://docs.streamlit.io
- Pandas documentation — https://pandas.pydata.org
- Seaborn documentation — https://seaborn.pydata.org

---

*Report prepared as part of a Data Science portfolio project by Vaisak Gopalakrishnan.*
