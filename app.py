"""
Streamlit app for ML Assignment 2
----------------------------------
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI / sklearn
Task: Binary classification (malignant vs benign)

Features:
  a. CSV upload of test data
  b. Model selection dropdown (with descriptions)
  c. Dataset preview + stats
  d. Evaluation metrics display (styled cards)
  e. Confusion matrix + classification report
  f. Highlight best model in comparison table
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Classifier Dashboard", layout="wide")
st.title("🩺 Breast Cancer Classification Dashboard")
st.caption("Machine Learning Assignment 2 — Comparison of 5 classifiers on the same dataset")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# ---------------------------------------------------------
# Load saved artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
        "kNN": joblib.load(os.path.join(MODEL_DIR, "knn.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
        "Random Forest (Ensemble)": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
    }
    comparison_table = pd.read_csv(os.path.join(MODEL_DIR, "comparison_table.csv"))
    return scaler, feature_names, models, comparison_table

scaler, feature_names, models, comparison_table = load_artifacts()

# ---------------------------------------------------------
# Sidebar: Upload + model selection
# ---------------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV) — must include a 'target' column",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select a model",
    list(models.keys()),
    help="Choose a classifier: Logistic Regression (linear), Decision Tree (rule-based), KNN (neighbors), Naive Bayes (probabilistic), Random Forest (ensemble)"
)

st.sidebar.markdown("---")
st.sidebar.info("Steps:\n1. Upload test CSV\n2. Select model\n3. View metrics, confusion matrix, and report")

st.sidebar.markdown(
    "**Expected CSV format:** the 30 breast-cancer feature columns "
    " plus a `target` column "
    "(0 = malignant, 1 = benign)."
)

# ---------------------------------------------------------
# Full model comparison table (always visible)
# ---------------------------------------------------------
st.subheader("📊 Model Comparison")
st.caption("Performance metrics on held-out test split")

# Round numbers for readability
comparison_table = comparison_table.round(4)

# Identify best model by F1 score
best_model = comparison_table.loc[comparison_table['F1'].idxmax(), 'ML Model Name']

# Highlight best model row inline
styled_table = comparison_table.style.apply(
    lambda row: ['background-color: #d4edda' if row['ML Model Name'] == best_model else '' for _ in row],
    axis=1
)

st.dataframe(styled_table, width="stretch") 

# Compact banner above table
st.success(f"🏆 Best overall model (F1 Score): {best_model}")

# ---------------------------------------------------------
# Main logic: only runs once a file is uploaded
# ---------------------------------------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column with true labels.")
        st.stop()

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols[:5]}...")
        st.stop()

    # Dataset preview + stats
    st.subheader("📂 Uploaded Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Target Distribution:", df['target'].value_counts())

    X_uploaded = df[feature_names]
    y_true = df["target"]

    X_scaled = scaler.transform(X_uploaded)

    selected_model = models[model_name]
    y_pred = selected_model.predict(X_scaled)

    if hasattr(selected_model, "predict_proba"):
        y_proba = selected_model.predict_proba(X_scaled)[:, 1]
    else:
        y_proba = selected_model.decision_function(X_scaled)

    # -----------------------------------------------------
    # Metrics (styled cards)
    # -----------------------------------------------------
    st.subheader(f"📈 Evaluation Metrics — {model_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
    col5.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
    col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

    # -----------------------------------------------------
    # Confusion matrix + classification report
    # -----------------------------------------------------
    st.subheader("🔎 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,2.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📋 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

else:
    st.info("👈 Upload a test CSV file from the sidebar to see live predictions, "
             "the confusion matrix, and the classification report for the selected model.")
