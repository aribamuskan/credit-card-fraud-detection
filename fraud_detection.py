import pandas as pd


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("credit_card_fraud_data.csv")

print("\n========== DATASET ==========")
print(df)


# ==========================================
# 2. DATASET SHAPE
# ==========================================

print("\n========== SHAPE ==========")
print(df.shape)


# ==========================================
# 3. COLUMN NAMES
# ==========================================

print("\n========== COLUMNS ==========")
print(df.columns)


# ==========================================
# 4. DATA TYPES
# ==========================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)


# ==========================================
# 5. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ==========================================
# 6. DUPLICATE ROWS
# ==========================================

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())


# ==========================================
# 7. FRAUD DISTRIBUTION
# ==========================================

print("\n========== FRAUD DISTRIBUTION ==========")
print(df["fraud"].value_counts())


# ==========================================
# 8. FRAUD PERCENTAGE
# ==========================================

print("\n========== FRAUD PERCENTAGE ==========")

fraud_percentage = df["fraud"].value_counts(normalize=True) * 100

print(fraud_percentage)


# ==========================================
# 9. BASIC STATISTICS
# ==========================================

print("\n========== STATISTICS ==========")
print(df.describe())
# ==========================================
# DATA PREPROCESSING
# ==========================================

print("\n========== DATA PREPROCESSING ==========")


# 1. Remove transaction_id
df = df.drop("transaction_id", axis=1)


# 2. Separate Features and Target

X = df.drop("fraud", axis=1)
y = df["fraud"]


print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)


# 3. Check Missing Values

print("\nMissing values after preprocessing:")
print(X.isnull().sum().sum())


# 4. Check Duplicate Rows

print("\nDuplicate rows:")
print(X.duplicated().sum())


# 5. Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 6. Check Train/Test Sizes

print("\n========== TRAIN/TEST SPLIT ==========")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:")
print(y_train.value_counts())

print("\nTesting target:")
print(y_test.value_counts())


# 7. Check Fraud Percentage

print("\n========== FRAUD DISTRIBUTION ==========")

print(
    "Training fraud percentage:",
    y_train.mean() * 100
)

print(
    "Testing fraud percentage:",
    y_test.mean() * 100
)


# ==========================================
# PREPROCESSING COMPLETED
# ==========================================

print("\n========================================")
print("DATA PREPROCESSING COMPLETED")
print("========================================")
# ==========================================
# MODEL TRAINING
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


print("\n========================================")
print("MODEL TRAINING")
print("========================================")


# ==========================================
# 1. LOGISTIC REGRESSION
# ==========================================

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

print("\nLogistic Regression training completed.")


# ==========================================
# 2. DECISION TREE
# ==========================================

decision_tree_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

decision_tree_model.fit(X_train, y_train)

print("Decision Tree training completed.")


# ==========================================
# 3. RANDOM FOREST
# ==========================================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

print("Random Forest training completed.")


# ==========================================
# 4. MAKE PREDICTIONS
# ==========================================

logistic_predictions = logistic_model.predict(X_test)

decision_tree_predictions = decision_tree_model.predict(X_test)

random_forest_predictions = random_forest_model.predict(X_test)


# ==========================================
# 5. PREDICTION PROBABILITIES
# ==========================================

logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

decision_tree_probabilities = decision_tree_model.predict_proba(X_test)[:, 1]

random_forest_probabilities = random_forest_model.predict_proba(X_test)[:, 1]


# ==========================================
# 6. BASIC PREDICTION COUNTS
# ==========================================

print("\n========================================")
print("PREDICTION COUNTS")
print("========================================")


print("\nLogistic Regression:")
print(pd.Series(logistic_predictions).value_counts())


print("\nDecision Tree:")
print(pd.Series(decision_tree_predictions).value_counts())


print("\nRandom Forest:")
print(pd.Series(random_forest_predictions).value_counts())


# ==========================================
# 7. MODEL TRAINING COMPLETED
# ==========================================

print("\n========================================")
print("MODEL TRAINING COMPLETED")
print("========================================")
# ==========================================
# MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("\n========================================")
print("MODEL EVALUATION")
print("========================================")


# ==========================================
# FUNCTION FOR MODEL EVALUATION
# ==========================================

def evaluate_model(
    model_name,
    y_true,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )


    print(f"\n========== {model_name} ==========")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("ROC-AUC:", roc_auc)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }


# ==========================================
# EVALUATE LOGISTIC REGRESSION
# ==========================================

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_probabilities
)


# ==========================================
# EVALUATE DECISION TREE
# ==========================================

decision_tree_results = evaluate_model(
    "Decision Tree",
    y_test,
    decision_tree_predictions,
    decision_tree_probabilities
)


# ==========================================
# EVALUATE RANDOM FOREST
# ==========================================

random_forest_results = evaluate_model(
    "Random Forest",
    y_test,
    random_forest_predictions,
    random_forest_probabilities
)


# ==========================================
# MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame([
    logistic_results,
    decision_tree_results,
    random_forest_results
])


print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    comparison.to_string(index=False)
)


# ==========================================
# BEST MODEL BASED ON F1
# ==========================================

best_model = comparison.loc[
    comparison["F1-Score"].idxmax()
]


print("\n========================================")
print("BEST MODEL BASED ON F1-SCORE")
print("========================================")

print(
    "Model:",
    best_model["Model"]
)

print(
    "F1-Score:",
    best_model["F1-Score"]
)


print("\n========================================")
print("MODEL EVALUATION COMPLETED")
print("========================================")
# ==========================================
# THRESHOLD TUNING + PRECISION-RECALL ANALYSIS
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score
)

print("\n========================================")
print("THRESHOLD TUNING")
print("========================================")

# Logistic Regression probabilities
# Hum Logistic Regression ko analyze kar rahe hain
# because isne fraud recall aur F1 mein best result diya tha.

thresholds = np.arange(0.10, 0.91, 0.05)

threshold_results = []

for threshold in thresholds:

    threshold_predictions = (
        logistic_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })


threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold Comparison:")
print(threshold_df.to_string(index=False))


# ==========================================
# BEST THRESHOLD BASED ON F1
# ==========================================

best_threshold_row = threshold_df.loc[
    threshold_df["F1-Score"].idxmax()
]

best_threshold = best_threshold_row["Threshold"]

print("\n========================================")
print("BEST THRESHOLD BASED ON F1-SCORE")
print("========================================")

print("Best Threshold:", best_threshold)
print("Precision:", best_threshold_row["Precision"])
print("Recall:", best_threshold_row["Recall"])
print("F1-Score:", best_threshold_row["F1-Score"])


# ==========================================
# PRECISION-RECALL CURVE
# ==========================================

precision_values, recall_values, pr_thresholds = precision_recall_curve(
    y_test,
    logistic_probabilities
)

average_precision = average_precision_score(
    y_test,
    logistic_probabilities
)

print("\n========================================")
print("PRECISION-RECALL ANALYSIS")
print("========================================")

print("Average Precision:", average_precision)


plt.figure(figsize=(8, 6))

plt.plot(
    recall_values,
    precision_values,
    label="Logistic Regression"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.legend()
plt.grid()

plt.show()


# ==========================================
# FINAL THRESHOLD PREDICTIONS
# ==========================================

final_predictions = (
    logistic_probabilities >= best_threshold
).astype(int)

final_precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)

print("\n========================================")
print("FINAL THRESHOLD RESULTS")
print("========================================")

print("Threshold:", best_threshold)
print("Precision:", final_precision)
print("Recall:", final_recall)
print("F1-Score:", final_f1)

print("\n========================================")
print("THRESHOLD TUNING COMPLETED")
print("========================================")
# ==========================================
# SAVE FINAL MODEL
# ==========================================

import joblib

print("\n========================================")
print("SAVING FINAL MODEL")
print("========================================")

# Save trained Logistic Regression model
joblib.dump(
    logistic_model,
    "fraud_detection_model.pkl"
)

# Save feature names
joblib.dump(
    X.columns.tolist(),
    "fraud_detection_features.pkl"
)

# Save the selected threshold
joblib.dump(
    best_threshold,
    "fraud_detection_threshold.pkl"
)

print("Model saved as: fraud_detection_model.pkl")
print("Features saved as: fraud_detection_features.pkl")
print("Threshold saved as: fraud_detection_threshold.pkl")

print("\n========================================")
print("MODEL SAVING COMPLETED")
print("========================================")


# ==========================================
# NEW TRANSACTION PREDICTION
# ==========================================

print("\n========================================")
print("NEW TRANSACTION PREDICTION")
print("========================================")

# Example new transaction
new_transaction = pd.DataFrame({
    "amount": [850.00],
    "hour": [2],
    "distance_from_home_km": [450.0],
    "transactions_last_24h": [8],
    "card_present": [0],
    "international_transaction": [1],
    "new_device": [1],
    "failed_attempts": [3]
})

# Make sure columns are in the same order
new_transaction = new_transaction[X.columns]

# Get fraud probability
fraud_probability = logistic_model.predict_proba(
    new_transaction
)[:, 1][0]

# Apply selected threshold
if fraud_probability >= best_threshold:
    prediction = 1
    result = "FRAUD"
else:
    prediction = 0
    result = "NORMAL"

print("\nNew Transaction:")
print(new_transaction)

print("\nFraud Probability:", fraud_probability)
print("Threshold:", best_threshold)
print("Prediction:", prediction)
print("Result:", result)

print("\n========================================")
print("PREDICTION COMPLETED")
print("========================================")