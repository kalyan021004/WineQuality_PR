"""
Person B: Random Forest with Interaction Features + SelectKBest (3-Class Wine Quality)
Feature extraction:
 - Pairwise interaction terms for selected chemical features
 - SelectKBest (ANOVA F-test) feature selection
Classifier: RandomForestClassifier
Saves model to: model_personB_multiclass.joblib
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path

# -------------------- LOAD DATA --------------------
def load_data():
    file_path = Path("winequality_combined.csv")  # Change filename if needed
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

    # Pairwise interaction features
    base_cols = ['alcohol', 'sulphates', 'density', 'pH', 'residual sugar']
    for i in range(len(base_cols)):
        for j in range(i+1, len(base_cols)):
            a = base_cols[i]; b = base_cols[j]
            X[f"{a}_x_{b}"] = X[a] * X[b]

    # Encode wine type
    X['wine_type_code'] = (X['wine_type'] == 'red').astype(int)

    # Convert quality to 3 classes
    y = X['quality']
    y = np.where(y <= 5, 0, np.where(y == 6, 1, 2))

    # Drop original columns
    X = X.drop(columns=['quality', 'wine_type'])

    # Fix NaNs
    X = X.fillna(X.mean())

    return X, y

# -------------------- MAIN PIPELINE --------------------
def run_pipeline():
    df = load_data()
    X, y = feature_engineer(df)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Select best 12 features
    selector = SelectKBest(score_func=f_classif, k=12)
    Xk = selector.fit_transform(Xs, y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(clf, Xk, y, cv=cv)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')

    print("\n[Person B] Random Forest + SelectKBest (3-Class Quality)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}\n")

    clf.fit(Xk, y)

    save_path = "model_personB_multiclass.joblib"
    joblib.dump({'model': clf, 'scaler': scaler, 'selector': selector}, save_path)
    print(f"✅ Model saved at: {save_path}")

# -------------------- RUN --------------------
if __name__ == '__main__':
    run_pipeline()
