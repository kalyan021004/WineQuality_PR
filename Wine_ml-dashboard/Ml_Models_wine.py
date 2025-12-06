# ============================================================
# 🍷 WINE QUALITY CLASSIFICATION — 6 MODELS (VS CODE VERSION)
# CatBoost, XGBoost, LightGBM, RandomForest, SVM, GradientBoosting
# ------------------------------------------------------------
# Saves:
# - results/<model>_cm.png
# - results/<model>_roc.png
# - models/<model>_model.pkl
# - models/<model>_metrics.json
# ============================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

# ============= INSTALL MISSING LIBRARIES ==============
try:
    from catboost import CatBoostClassifier
except:
    os.system("pip install catboost")
    from catboost import CatBoostClassifier

try:
    from lightgbm import LGBMClassifier
except:
    os.system("pip install lightgbm")
    from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier
except:
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

# ============= STANDARD LIBRARIES ======================
# ============= STANDARD LIBRARIES ======================
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# IMPORTANT — YOU MISSED THIS EARLIER
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.svm import SVC


# Create folders
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================
df = pd.read_csv("wine.csv")

df = df.dropna(subset=["quality"])
df.fillna(df.median(numeric_only=True), inplace=True)

if "wine_type" in df.columns:
    df["wine_type"] = LabelEncoder().fit_transform(df["wine_type"].astype(str))

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
df["sulfur_ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"].replace(0, np.nan)
df["sugar_per_acid"] = df["residual sugar"] / (df["total_acidity"] + 1e-5)
df["acid_sugar_ratio"] = df["total_acidity"] / (df["residual sugar"] + 1e-5)
df["density_alcohol_ratio"] = df["density"] / (df["alcohol"] + 1e-5)
df["is_high_alcohol"] = (df["alcohol"] > df["alcohol"].median()).astype(int)
df["is_high_sugar"] = (df["residual sugar"] > df["residual sugar"].median()).astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# ============================================================
# 3. LABEL CREATION
# ============================================================
def encode_quality(q):
    return 0 if q <= 5 else (1 if q == 6 else 2)

df["quality_class"] = df["quality"].apply(encode_quality)

X = df.drop(columns=["quality", "quality_class"])
y = df["quality_class"]

# Scaling
X = StandardScaler().fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================
def evaluate_model(name, model, y_pred, y_prob):
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")

    report = classification_report(
        y_test, y_pred, target_names=["Low", "Medium", "High"], output_dict=True
    )

    # Save Metrics JSON
    with open(f"models/{name}_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=4)

    # Save Model
    joblib.dump(model, f"models/{name}_model.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"])
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"results/{name}_cm.png")
    plt.close()

    # ROC Curve
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(["Low", "Medium", "High"]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc_score:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.savefig(f"results/{name}_roc.png")
    plt.close()

    return acc

# ============================================================
# 5. TRAIN SIX MODELS
# ============================================================

models = {
    "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8, verbose=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=400, learning_rate=0.05),
    "LightGBM": LGBMClassifier(n_estimators=600, num_leaves=35),
    "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=14),
    "SVM": SVC(kernel="rbf", C=2, probability=True),
    "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.05)
}

accuracies = {}

# Train all models
for name, model in models.items():
    print(f"\n================ TRAINING {name} ================")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracies[name] = evaluate_model(name, model, y_pred, y_prob)

# Print summary
print("\n\n=================== SUMMARY ===================")
for k, v in accuracies.items():
    print(f"{k}: {v:.4f}")

best = max(accuracies, key=accuracies.get)
print(f"\nBEST MODEL: {best} (Accuracy = {accuracies[best]:.4f})")
