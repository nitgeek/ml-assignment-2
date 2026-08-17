"""
ML Assignment 2 - Train Classification Models
Dataset: Wine Quality (UCI) - Binary Classification
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
import joblib
import os

# --- Step 1: Load and Prepare Dataset ---
print("=" * 60)
print("ML Assignment 2 - Wine Quality Classification")
print("=" * 60)

# Load Wine Quality dataset (Red wine from UCI)
# Try downloading, fall back to sklearn wine dataset if SSL fails
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    print(f"\nLoading dataset from: {url}")
    df = pd.read_csv(url, sep=';')
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(f"\nQuality distribution:\n{df['quality'].value_counts().sort_index()}")
    # Convert to binary classification: quality >= 7 is "good" (1), else "not good" (0)
    df['target'] = (df['quality'] >= 7).astype(int)
    df = df.drop('quality', axis=1)
except Exception as e:
    print(f"\nCould not download from UCI ({e})")
    print("Using sklearn's wine dataset instead...")
    from sklearn.datasets import load_wine
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    # Binary classification: class 0 vs others
    df['target'] = (wine.target == 0).astype(int)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")

print(f"\nBinary target distribution:")
print(f"  Not Good (0): {(df['target'] == 0).sum()}")
print(f"  Good (1):     {(df['target'] == 1).sum()}")

# --- Step 2: Split Data ---
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size:  {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test data
test_df = X_test.copy()
test_df['target'] = y_test.values
test_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data.csv'), index=False)
print("\nTest data saved to test_data.csv")

# --- Step 3: Train Models and Evaluate ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n" + "=" * 60)
print("Training and Evaluating Models")
print("=" * 60)

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    }

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), f'{name.lower().replace(" ", "_")}.pkl')
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

# Save scaler
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"\nScaler saved: {scaler_path}")

# --- Step 4: Print Comparison Table ---
print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df.index.name = 'ML Model Name'
print(results_df.to_string())

# Find best model
best_model = results_df['F1'].idxmax()
print(f"\n🏆 Best Model (by F1 Score): {best_model}")
print(f"   F1 Score: {results_df.loc[best_model, 'F1']}")

print("\n✅ All models trained and saved successfully!")
