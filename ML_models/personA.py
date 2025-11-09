"""
Person A: SVM with PCA + Domain-derived features
3-Class Classification Version
Mapping:
  quality <= 5 -> 0 (Low)
  quality == 6 -> 1 (Medium)
  quality >= 7 -> 2 (High)
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path

# -------------------- LOAD DATA --------------------
def load_data():
    file_path = Path("winequality_combined.csv")  # Change name if required
    if not file_path.exists():
        print("Error: Place 'winequality_combined.csv' in this folder.")
        sys.exit(1)

    df = pd.read_csv(file_path)

    if 'wine_type' not in df.columns:
        print("Error: Dataset must contain 'wine_type' column (red/white).")
        sys.exit(1)

    return df

# -------------------- FEATURE ENGINEERING --------------------
def feature_engineer(df):
    X = df.copy()

    # Domain-based features
    X['acidity_ratio'] = X['fixed acidity'] / (X['volatile acidity'] + 1e-6)
    X['sulfur_ratio'] = X['total sulfur dioxide'] / (X['free sulfur dioxide'] + 1e-6)
    X['sweetness_index'] = X['residual sugar'] * X['density']

    # Encode wine type (red=1, white=0)
    X['wine_type_code'] = (X['wine_type'] == 'red').astype(int)

    # 3-Class Target Mapping
    y = X['quality']
    y = np.where(y <= 5, 0, np.where(y == 6, 1, 2))

    # Drop text + original target
    X = X.drop(columns=['quality', 'wine_type'])

    # Fix missing values
    X = X.fillna(X.mean())

    return X, y

# -------------------- MAIN PIPELINE --------------------
def run_pipeline():
    df = load_data()
    X, y = feature_engineer(df)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=8)
    Xp = pca.fit_transform(Xs)

    # Better SVM settings for multi-class
    clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(clf, Xp, y, cv=cv)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')

    print("\n[Person A] SVM + PCA (3-Class Quality)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}\n")

    clf.fit(Xp, y)

    save_path = "model_personA_multiclass.joblib"
    joblib.dump({'model': clf, 'scaler': scaler, 'pca': pca}, save_path)
    print(f"✅ Model saved at: {save_path}")

# -------------------- RUN --------------------
if __name__ == '__main__':
    run_pipeline()
