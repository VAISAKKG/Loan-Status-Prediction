# 🏦 Loan Status Prediction — Machine Learning ProjectMachine Learning / Supervised Classification


![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classification-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A supervised machine learning project to predict loan approval status based on applicant profile and financial details. The project covers the complete data science pipeline — from data preprocessing and exploratory analysis to multi-model training, evaluation, and deployment via an interactive Streamlit web application.

---

## 📁 Project Structure

```
loan-status-prediction/
│
├── ML_Project_Loan_Status_Prediction.ipynb   # Jupyter Notebook (EDA + Modelling)
├── train_model.py                            # Standalone model training script
├── app.py                                    # Streamlit web application
├── Loan_predict.csv                          # Dataset (614 loan records)
├── model.pkl                                 # Saved best-performing model (pickle)
├── README.md                                 # Project overview (this file)
└── PROJECT_REPORT.md                         # Detailed technical report
```

---

## 📊 Dataset Overview

| Property       | Details                                      |
|----------------|----------------------------------------------|
| Source         | Loan eligibility dataset (CSV)               |
| Records        | 614 rows                                     |
| Features       | 13 columns (raw), 11 used for modelling      |
| Target         | `Loan_Status` (Y = Approved, N = Rejected)   |
| File           | `Loan_predict.csv`                           |

**Features used for modelling:**

| Feature              | Type        | Description                                  |
|----------------------|-------------|----------------------------------------------|
| `Gender`             | Categorical | Male / Female                                |
| `Married`            | Categorical | Yes / No                                     |
| `Dependents`         | Numeric     | Number of dependents (0, 1, 2, 3+)          |
| `Education`          | Categorical | Graduate / Not Graduate                      |
| `Self_Employed`      | Categorical | Yes / No                                     |
| `ApplicantIncome`    | Numeric     | Monthly income of applicant                  |
| `CoapplicantIncome`  | Numeric     | Monthly income of co-applicant               |
| `LoanAmount`         | Numeric     | Loan amount requested                        |
| `Loan_Amount_Term`   | Numeric     | Term of loan in months                       |
| `Credit_History`     | Binary      | 1 = Good credit history, 0 = Poor            |
| `Property_Area`      | Categorical | Urban / Semiurban / Rural                    |

---

## ⚙️ Workflow Summary

```
Raw Data → Preprocessing → EDA → Feature Encoding → Model Training → Evaluation → Export → Streamlit App
```

1. **Data Cleaning** — Dropped `Loan_ID`; handled missing values via mode/mean imputation
2. **Label Encoding** — Categorical variables converted to numeric codes
3. **EDA** — Loan status distribution, property area analysis, income vs loan amount, heatmap
4. **Modelling** — Trained 3 classification models; evaluated on Accuracy, Precision, Recall, F1
5. **Export** — Best model saved as `model.pkl` via `pickle`
6. **Deployment** — Interactive Streamlit app for real-time loan eligibility prediction

---

## 🤖 Models Trained

| Model                  | Description                                          |
|------------------------|------------------------------------------------------|
| Logistic Regression    | Linear baseline classifier; scaled with StandardScaler |
| Decision Tree          | Non-linear tree-based classifier                     |
| SVM (Linear Kernel)    | Support Vector Machine with linear boundary          |

> **Evaluation metrics used:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

The best-performing model (highest accuracy) is automatically selected and saved as `model.pkl`.

---

## 📈 Visualisations

- Loan status distribution (bar chart)
- Property area vs loan status (count plot)
- Education vs loan status (count plot)
- Applicant income vs loan amount (scatter plot)
- Feature correlation heatmap
- Model comparison — Accuracy, Precision, Recall, F1 (grouped bar chart)
- Confusion matrix (heatmap)

---

## 🖥️ Streamlit App

An interactive web dashboard (`app.py`) allows users to check loan eligibility instantly.

### Features

- Two-column input layout for applicant profile details
- Loan term selector (3 to 30 years) — automatically converted to months for the model
- Credit history toggle with a clear label (Good / Poor)
- Dark gradient UI with styled Approve / Reject result cards
- One-click **Predict Loan Status** button

### Run Locally

```bash
# Install dependencies
pip install streamlit numpy scikit-learn

# Run the app (model.pkl must be in the same directory)
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Deploy on Streamlit Cloud

1. Push all project files to a public GitHub repository
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **New app** → select repo → set `app.py` as the entry point
4. Click **Deploy** — your app goes live at a `*.streamlit.app` URL

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
https://github.com/VAISAKKG/Loan-Status-Prediction/edit/main/README.md
cd loan-status-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 3. Train the model (optional — `model.pkl` is already included)

```bash
python train_model.py
```

### 4. Run the notebook

Open `ML_Project_Loan_Status_Prediction.ipynb` in Jupyter or Google Colab and run all cells.

> **Note:** If using Google Colab, update the dataset path in the data loading cell:
> ```python
> loan_dataset = pd.read_csv('/content/Loan_predict.csv')
> ```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Web App:** Streamlit
- **Model Serialisation:** Pickle
- **Environment:** Jupyter Notebook / Google Colab / Streamlit Cloud

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Vaisak Gopalakrishnan**
Data Analyst | MSc Data Science
[LinkedIn](#) · [GitHub](#)
