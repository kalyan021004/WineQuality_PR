"""
Person C: Gradient Boosting + LDA + Statistical Feature Engineering (3-Class Wine Quality)
Feature Engineering:
 - alcohol/density ratio
 - log residual sugar
 - normalized sulphates
 - sulfur balance (total - free)
 - wine type encoding
Dimensionality Reduction:
 - Linear Discriminant Analysis (LDA), n_components = min(3-1, 4) = 2
Classifier:
 - GradientBoostingClassifier
Saves model to: model_personC_multiclass.joblib
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path

# -------------------- LOAD DATA --------------------
def load_data():
    file_path = Path("winequality_combined.csv")  # Change name if needed
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

    # Statistical feature engineering
    X['alc_over_density'] = X['alcohol'] / (X['density'] + 1e-6)
    X['log_res_sugar'] = np.log1p(X['residual sugar'])
    X['norm_sulphates'] = (X['sulphates'] - X['sulphates'].mean()) / (X['sulphates'].std() + 1e-6)
    X['sulfur_balance'] = X['total sulfur dioxide'] - X['free sulfur dioxide']

    # Encode wine type
    X['wine_type_code'] = (X['wine_type'] == 'red').astype(int)

    # 3-Class quality conversion
    y = X['quality']
    y = np.where(y <= 5, 0, np.where(y == 6, 1, 2))

    # Drop columns not needed
    X = X.drop(columns=['quality', 'wine_type'])

    # Fill any NaNs
    X = X.fillna(X.mean())

    return X, y

# -------------------- MAIN PIPELINE --------------------
def run_pipeline():
    df = load_data()
    X, y = feature_engineer(df)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # LDA (max possible components = n_classes - 1 = 2)
    lda = LinearDiscriminantAnalysis(n_components=2)
    try:
        X_lda = lda.fit_transform(Xs, y)
    except Exception as e:
        print("LDA failed, using scaled data instead:", e)
        X_lda = Xs

    clf = GradientBoostingClassifier(n_estimators=200, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(clf, X_lda, y, cv=cv)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')

    print("\n[Person C] Gradient Boosting + LDA (3-Class Quality)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}\n")

    clf.fit(X_lda, y)

    save_path = "model_personC_multiclass.joblib"
    joblib.dump({'model': clf, 'scaler': scaler, 'lda': lda}, save_path)
    print(f"✅ Model saved at: {save_path}")

# -------------------- RUN --------------------
if __name__ == '__main__':
    run_pipeline()
