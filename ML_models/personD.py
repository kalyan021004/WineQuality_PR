"""
Person D: KNN with Polynomial Features + PCA (optional) for 3-Class Wine Quality
Feature Engineering:
 - acid_balance = (fixed acidity + citric acid) / volatile acidity
 - wine_type encoding
 - PolynomialFeatures(degree=2) to capture non-linear interactions
Dimensionality Reduction:
 - Optional PCA(n_components=10) to reduce feature size
Classifier:
 - KNeighborsClassifier (k=7)
Saves model to: model_personD_multiclass.joblib
"""

import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path

# -------------------- LOAD DATA --------------------
def load_data():
    file_path = Path("winequality_combined.csv")
    if not file_path.exists():
        print("Error: Place 'winequality_combined.csv' in this folder.")
        sys.exit(1)

    df = pd.read_csv(file_path)

    if 'wine_type' not in df.columns:
        print("Error: Dataset must contain 'wine_type' column ('red'/'white').")
        sys.exit(1)

    return df

# -------------------- FEATURE ENGINEERING --------------------
def feature_engineer(df):
    X = df.copy()

    # Domain feature
    X['acid_balance'] = (X['fixed acidity'] + X['citric acid']) / (X['volatile acidity'] + 1e-6)

    # Encode wine type
    X['wine_type_code'] = (X['wine_type'] == 'red').astype(int)

    # Convert quality to 3 classes
    y = X['quality']
    y = np.where(y <= 5, 0, np.where(y == 6, 1, 2))

    # Drop columns not needed
    X = X.drop(columns=['quality', 'wine_type'])

    # Fix missing values
    X = X.fillna(X.mean())

    return X, y

# -------------------- MAIN PIPELINE --------------------
def run_pipeline():
    df = load_data()
    X, y = feature_engineer(df)

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(Xs)

    # PCA (optional, reduces dimension & speeds KNN)
    pca = PCA(n_components=10)
    try:
        Xpc = pca.fit_transform(Xp)
    except Exception as e:
        print("PCA failed, using poly features directly:", e)
        Xpc = Xp

    # KNN Model
    clf = KNeighborsClassifier(n_neighbors=7)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(clf, Xpc, y, cv=cv)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')

    print("\n[Person D] KNN + PolynomialFeatures + PCA (3-Class Quality)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}\n")

    clf.fit(Xpc, y)

    # Save model
    save_path = "model_personD_multiclass.joblib"
    joblib.dump({'model': clf, 'scaler': scaler, 'poly': poly, 'pca': pca}, save_path)
    print(f"✅ Model saved at: {save_path}")

# -------------------- RUN --------------------
if __name__ == '__main__':
    run_pipeline()
