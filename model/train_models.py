"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset
(UCI Machine Learning Repository / also available via sklearn.datasets).

Dataset: 569 instances, 30 numeric features, binary target (malignant / benign)
-> Satisfies assignment minimums: >=12 features, >=500 instances.

Models trained:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbors Classifier
  4. Gaussian Naive Bayes
  5. Random Forest Classifier (Ensemble)

Outputs (saved into the model/ folder):
  - scaler.pkl
  - logistic_regression.pkl
  - decision_tree.pkl
  - knn.pkl
  - naive_bayes.pkl
  - random_forest.pkl
  - feature_names.pkl
  - comparison_table.csv   (metrics for all 5 models)

Also writes ../test_data.csv at project root (a held-out test split, features + true label)
for use in the Streamlit app.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORTS_DIR = os.path.join(HERE, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Consistent short "key" for each model name -> used in filenames everywhere
NAME_KEY = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest (Ensemble)": "random_forest",
}

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class distribution:\n{y.value_counts()}")

# ---------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------
# 3. Scale features (fit on train only)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []
roc_data = {}  # name -> (fpr, tpr, auc)
os.makedirs(HERE, exist_ok=True)

# ---------------------------------------------------------
# 5. Train, evaluate, save each model
# ---------------------------------------------------------
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # AUC needs probability scores
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    key = NAME_KEY[name]

    # Save model (explicit filename map keeps this in sync with app.py)
    filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest.pkl",
    }
    joblib.dump(model, os.path.join(HERE, filename_map[name]))

    # -------------------------------------------------------
    # Confusion Matrix (saved as PNG, one per model)
    # -------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, f"confusion_matrix_{key}.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------
    # Classification Report (saved as .txt, one per model)
    # -------------------------------------------------------
    report_str = classification_report(
        y_test, y_pred, target_names=["Malignant (0)", "Benign (1)"]
    )
    with open(os.path.join(REPORTS_DIR, f"classification_report_{key}.txt"), "w") as f:
        f.write(f"Classification Report — {name}\n")
        f.write("=" * 50 + "\n")
        f.write(report_str)

    # -------------------------------------------------------
    # Stash ROC curve data for plotting after the loop
    # -------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, metrics["AUC"])

# ---------------------------------------------------------
# 5b. ROC Curves
#     - one combined plot comparing all 5 models
#     - individual highlighted plots for Logistic Regression and Random Forest
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
for name, (fpr, tpr, auc_val) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — All Models")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS_DIR, "roc_curve_all_models.png"), dpi=150)
plt.close(fig)

for highlight_name in ["Logistic Regression", "Random Forest (Ensemble)"]:
    fpr, tpr, auc_val = roc_data[highlight_name]
    key = NAME_KEY[highlight_name]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="darkorange", linewidth=2.5,
            label=f"{highlight_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {highlight_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, f"roc_curve_{key}.png"), dpi=150)
    plt.close(fig)

print(f"\nSaved to {REPORTS_DIR}:")
print("  - confusion_matrix_<model>.png (5 files)")
print("  - classification_report_<model>.txt (5 files)")
print("  - roc_curve_all_models.png (combined comparison)")
print("  - roc_curve_logistic_regression.png / roc_curve_random_forest.png (highlighted)")

# ---------------------------------------------------------
# 6. Save scaler + feature names (needed by the Streamlit app)
# ---------------------------------------------------------
joblib.dump(scaler, os.path.join(HERE, "scaler.pkl"))
joblib.dump(list(X.columns), os.path.join(HERE, "feature_names.pkl"))

# ---------------------------------------------------------
# 7. Save comparison table
# ---------------------------------------------------------
comparison_df = pd.DataFrame(results)
comparison_df.to_csv(os.path.join(HERE, "comparison_table.csv"), index=False)
print("\nComparison table saved to model/comparison_table.csv")
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------
# 8. Save test_data.csv (features + true label) for the Streamlit app
#    This is the "test data used in experiments" required at repo root.
# ---------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"\ntest_data.csv saved with shape {test_df.shape} to project root.")
