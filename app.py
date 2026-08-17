"""
ML Assignment 2 - Streamlit Web Application
Wine Quality Classification - Binary Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Wine Quality Classification",
    page_icon="🍷",
    layout="wide"
)

# --- Title ---
st.title("🍷 Wine Quality Classification")
st.markdown("**ML Assignment 2** - Comparing Multiple Classification Models")
st.markdown("---")

# --- Load Models ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl'
}


@st.cache_resource
def load_models():
    """Load all trained models and scaler."""
    loaded_models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return loaded_models, scaler


models, scaler = load_models()

if not models:
    st.error("❌ No trained models found. Please run train_models.py first.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")

# Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    options=list(models.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.markdown("""
- **Source**: UCI Wine Quality (Red)
- **Features**: 11 physicochemical properties
- **Target**: Binary (Good/Not Good)
- **Threshold**: Quality ≥ 7 → Good
""")

# --- Main Content ---
# File upload
st.header("📁 Upload Test Data (CSV)")
st.markdown("Upload your test dataset in CSV format. The file should contain the same features as the training data.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload test data CSV with features and 'target' column"
)

# Load default test data if no upload
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Uploaded file loaded: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    default_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info(f"ℹ️ Using default test data: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        st.warning("⚠️ No test data available. Please upload a CSV file.")
        st.stop()

# Display dataset preview
with st.expander("📋 Dataset Preview", expanded=False):
    st.dataframe(df.head(10))

# Check for target column
if 'target' not in df.columns:
    st.error("❌ The uploaded CSV must contain a 'target' column for evaluation.")
    st.stop()

# Separate features and target
X_test = df.drop('target', axis=1)
y_test = df['target']

# Scale features
if scaler is not None:
    X_test_scaled = scaler.transform(X_test)
else:
    X_test_scaled = X_test.values

st.markdown("---")

# --- Evaluation for Selected Model ---
st.header(f"📈 Results: {selected_model_name}")

model = models[selected_model_name]
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", f"{acc:.4f}")
    st.metric("AUC Score", f"{auc:.4f}")
with col2:
    st.metric("Precision", f"{prec:.4f}")
    st.metric("Recall", f"{rec:.4f}")
with col3:
    st.metric("F1 Score", f"{f1:.4f}")
    st.metric("MCC", f"{mcc:.4f}")

st.markdown("---")

# --- Confusion Matrix ---
st.subheader("🔢 Confusion Matrix")

col_cm, col_cr = st.columns(2)

with col_cm:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Not Good (0)', 'Good (1)'],
        yticklabels=['Not Good (0)', 'Good (1)'],
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {selected_model_name}')
    st.pyplot(fig)
    plt.close()

with col_cr:
    st.subheader("📝 Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=['Not Good (0)', 'Good (1)'],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"))

st.markdown("---")

# --- All Models Comparison ---
st.header("📊 All Models Comparison")

all_results = {}
for name, mdl in models.items():
    pred = mdl.predict(X_test_scaled)
    proba = mdl.predict_proba(X_test_scaled)[:, 1]
    all_results[name] = {
        'Accuracy': round(accuracy_score(y_test, pred), 4),
        'AUC': round(roc_auc_score(y_test, proba), 4),
        'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, pred, zero_division=0), 4),
        'F1': round(f1_score(y_test, pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, pred), 4)
    }

comparison_df = pd.DataFrame(all_results).T
comparison_df.index.name = 'Model'
st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))

# Bar chart comparison
st.subheader("📉 Visual Comparison")
fig2, ax2 = plt.subplots(figsize=(12, 5))
comparison_df.plot(kind='bar', ax=ax2)
ax2.set_title('Model Performance Comparison')
ax2.set_xlabel('Model')
ax2.set_ylabel('Score')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.legend(loc='lower right')
ax2.set_ylim(0, 1.05)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# Best model
best_model_name = comparison_df['F1'].idxmax()
st.success(f"🏆 **Best Model (by F1 Score): {best_model_name}** with F1 = {comparison_df.loc[best_model_name, 'F1']:.4f}")

st.markdown("---")
st.markdown("*ML Assignment 2 - BITS Pilani WILP | M.Tech AIML/DSE*")
