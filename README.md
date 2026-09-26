# 💳 Credit Card Fraud Detection

A Machine Learning project that detects potentially fraudulent credit card transactions using classification algorithms and provides real-time predictions through a Streamlit web application.

The project demonstrates an end-to-end Machine Learning workflow, including data understanding, preprocessing, model training, evaluation, threshold tuning, model persistence, and deployment.

---

## 🚀 Live Demo

Try the deployed Streamlit application:

**[Credit Card Fraud Detection · Streamlit](https://credit-card-fraud-detection-2cgm6lmf4wappruutt7crgh.streamlit.app/)**

---

## 📌 Project Overview

Credit card fraud detection is a **binary classification** problem where a transaction is classified as either normal or fraudulent.

The objective of this project is to analyze transaction characteristics and predict whether a transaction is potentially fraudulent.

### Target Variable

```text
0 → Normal Transaction
1 → Fraudulent Transaction
```

The project also addresses the challenge of **imbalanced classification**, where fraudulent transactions represent a much smaller portion of the dataset.

---

## 🎯 Project Objectives

- Understand the transaction dataset
- Perform data preprocessing
- Handle an imbalanced classification problem
- Train multiple Machine Learning classification models
- Evaluate model performance using multiple metrics
- Compare different classification models
- Perform probability threshold tuning
- Analyze the Precision-Recall relationship
- Save the trained model
- Build a real-time prediction system
- Deploy the model using Streamlit

---

## 📊 Dataset

The dataset contains **5,000 synthetic credit card transactions** created for educational, learning, and portfolio practice purposes.

### Dataset Features

| Feature | Description |
|---|---|
| `transaction_id` | Unique transaction identifier |
| `amount` | Transaction amount |
| `hour` | Hour at which the transaction occurred |
| `distance_from_home_km` | Distance of the transaction from the customer's home |
| `transactions_last_24h` | Number of transactions made in the last 24 hours |
| `card_present` | Whether the physical card was present |
| `international_transaction` | Whether the transaction was international |
| `new_device` | Whether a new device was used |
| `failed_attempts` | Number of failed attempts |
| `fraud` | Target variable |

### Target Variable

```text
0 = Normal Transaction
1 = Fraudulent Transaction
```

---

## 📊 Class Distribution

The dataset contains:

| Class | Count | Percentage |
|---|---:|---:|
| Normal Transactions | 4,776 | 95.52% |
| Fraudulent Transactions | 224 | 4.48% |

Because fraudulent transactions represent a small portion of the dataset, this is an **imbalanced classification problem**.

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Threshold Tuning
     ↓
Precision-Recall Analysis
     ↓
Final Model
     ↓
Model Saving
     ↓
New Transaction Prediction
     ↓
Streamlit Web Application
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Inspected the dataset
- Checked dataset shape and columns
- Checked data types
- Checked for missing values
- Checked for duplicate records
- Removed `transaction_id` from the modeling features
- Separated features (`X`) and target (`y`)
- Used an 80/20 train-test split
- Used `stratify=y` to preserve the class distribution
- Used `class_weight="balanced"` during model training to give greater importance to the minority fraud class

### Train/Test Split

```text
Training Data = 4,000 rows
Testing Data  = 1,000 rows
```

### Class Distribution After Splitting

```text
Training Fraud Percentage = 4.475%
Testing Fraud Percentage  = 4.50%
```

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and compared.

### 1. Logistic Regression

Logistic Regression was used as a linear classification model for estimating the probability of fraud.

### 2. Decision Tree

Decision Tree was used to learn decision rules from transaction features and classify transactions.

### 3. Random Forest

Random Forest was used as an ensemble learning model consisting of multiple decision trees.

---

## 📈 Model Evaluation

The models were evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

Because the dataset is imbalanced, **Accuracy alone is not sufficient** for understanding fraud detection performance.

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 82.20% | 17.56% | 80.00% | 0.288 | 0.878 |
| Decision Tree | 93.80% | 20.69% | 13.33% | 0.162 | 0.555 |
| Random Forest | 95.20% | 42.11% | 17.78% | 0.250 | 0.842 |

These results are based on the held-out test dataset.

---

## 🧮 Logistic Regression Confusion Matrix

The Logistic Regression model produced the following confusion matrix:

```text
[[786 169]
 [  9  36]]
```

For the fraud class:

- **True Positives:** 36
- **False Negatives:** 9
- **False Positives:** 169
- **True Negatives:** 786

This means the model correctly identified 36 of the 45 actual fraudulent transactions at the default classification threshold.

---

## 🎚️ Threshold Tuning

The Logistic Regression model was further analyzed using different probability thresholds.

The default classification threshold was:

```text
0.50
```

Different thresholds were tested to study the trade-off between precision and recall.

The threshold that produced the highest F1-score in this experiment was approximately:

```text
0.75
```

### Results at Threshold 0.75

| Metric | Result |
|---|---:|
| Precision | 32.86% |
| Recall | 51.11% |
| F1-Score | 0.40 |

Threshold tuning demonstrated that changing the decision threshold can change the balance between detecting fraudulent transactions and generating false alerts.

---

## 📉 Precision-Recall Analysis

A **Precision-Recall Curve** was generated to analyze the relationship between precision and recall across different classification thresholds.

### Average Precision

```text
0.408
```

This analysis provides an additional view of model performance for the imbalanced fraud detection problem.

---

## 💾 Model Saving

The trained model and supporting information were saved using `joblib`.

### Saved Files

```text
fraud_detection_model.pkl
fraud_detection_features.pkl
fraud_detection_threshold.pkl
```

These files allow the application to load the trained model and make predictions without retraining it every time.

---

## 🔮 New Transaction Prediction

The trained model was tested with a new transaction containing the following values:

| Feature | Value |
|---|---:|
| Amount | 850 |
| Hour | 2 |
| Distance From Home | 450 km |
| Transactions Last 24h | 8 |
| Card Present | No |
| International Transaction | Yes |
| New Device | Yes |
| Failed Attempts | 3 |

The model produced a very high fraud probability and classified the transaction as:

```text
🚨 FRAUD
```

---

## 🌐 Streamlit Web Application

A Streamlit web application was developed to provide an interactive interface for the trained Machine Learning model.

### User Inputs

The application allows the user to enter:

- Transaction amount
- Transaction hour
- Distance from home
- Number of transactions in the last 24 hours
- Card presence
- International transaction status
- New device status
- Number of failed attempts

### Application Output

The application displays:

- Fraud probability
- Fraud/Normal prediction
- Decision threshold

---

## 🔮 Prediction Workflow

The deployed application follows this process:

```text
User Input
    ↓
Input Conversion
    ↓
Feature Alignment
    ↓
Saved Machine Learning Model
    ↓
Fraud Probability
    ↓
Decision Threshold
    ↓
FRAUD / NORMAL
```

---

## 🖥️ Run the Application Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📁 Project Structure

```text
Credit_Card_Fraud/
│
├── credit_card_fraud_data.csv
├── fraud_detection.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── fraud_detection_model.pkl
├── fraud_detection_features.pkl
└── fraud_detection_threshold.pkl
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn

### Visualization

- Matplotlib

### Model Persistence

- Joblib

### Deployment

- Streamlit

### Version Control

- Git
- GitHub

---

## 📚 Machine Learning Concepts Practiced

This project provided practical experience with:

- Binary Classification
- Imbalanced Classification
- Data Understanding
- Data Preprocessing
- Train/Test Split
- Stratified Sampling
- Class Weighting
- Logistic Regression
- Decision Tree
- Random Forest
- Model Comparison
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- Probability Prediction
- Decision Thresholds
- Threshold Tuning
- Precision-Recall Curve
- Average Precision
- Model Persistence
- Streamlit Deployment
- Git
- GitHub

---

## 📌 Key Project Highlights

- Built an end-to-end fraud detection Machine Learning pipeline
- Worked with an imbalanced classification dataset
- Compared three different classification algorithms
- Used class weighting to address class imbalance
- Evaluated models using multiple classification metrics
- Performed probability threshold tuning
- Analyzed the Precision-Recall relationship
- Saved the trained model using Joblib
- Built an interactive Streamlit application
- Tested real-time transaction predictions
- Deployed the application online
- Documented the complete project workflow

---

## ⚠️ Disclaimer

This project uses a **synthetic dataset** created for educational, learning, and portfolio practice purposes.

It is not a production-ready financial fraud detection system and should not be used for real financial decisions without proper validation, security testing, monitoring, real-world data, and domain-specific evaluation.

---

## 👩‍💻 Author

**Ariba Muskan**

Software Engineering Student | Machine Learning Enthusiast

---

## ⭐ Project Summary

This project demonstrates an end-to-end Machine Learning workflow for a real-world-style credit card fraud detection problem.

The project covers the complete journey from **raw transaction data to a deployed Machine Learning application**, including data preprocessing, model training, model evaluation, threshold optimization, model persistence, real-time prediction, and Streamlit deployment.

### 🔗 Project Resources

- **Live Demo:** [Credit Card Fraud Detection · Streamlit](https://credit-card-fraud-detection-2cgm6lmf4wappruutt7crgh.streamlit.app/)
- **Source Code:** [GitHub Repository](https://github.com/aribamuskan/credit-card-fraud-detection)