# ============================================================
# 🍷 Wine Quality - LightGBM
# Saves:
# - models/lgbm_model.pkl
# - models/lgbm_metrics.json
# - results/lgbm_confusion_matrix.png
# - results/lgbm_roc_curve.png
# ============================================================

import os
import json
import joblib
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Create folders
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------- LOAD DATA --------------------
df = pd.read_csv("wine.csv")

df = df.dropna(subset=["quality"])
df.fillna(df.median(numeric_only=True), inplace=True)

if "wine_type" in df.columns:
    df["wine_type"] = LabelEncoder().fit_transform(df["wine_type"].astype(str))

# ---------------- FEATURE ENGINEERING ----------------
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
df["sulfur_ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"].replace(0, np.nan)
df["sugar_per_acid"] = df["residual sugar"] / (df["total_acidity"] + 1e-5)
df["acid_sugar_ratio"] = df["total_acidity"] / (df["residual sugar"] + 1e-5)
df["density_alcohol_ratio"] = df["density"] / (df["alcohol"] + 1e-5)
df["is_high_alcohol"] = (df["alcohol"] > df["alcohol"].median()).astype(int)
df["is_high_sugar"] = (df["residual sugar"] > df["residual sugar"].median()).astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# ---------------- LABELS ----------------
def label(q): return 0 if q <= 5 else (1 if q == 6 else 2)
df["quality_class"] = df["quality"].apply(label)

X = df.drop(columns=["quality", "quality_class"])
y = df["quality_class"]

# ---------------- SCALING ----------------
X = StandardScaler().fit_transform(X)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# ---------------- MODEL ----------------
model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=35,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ---------------- METRICS ----------------
acc = round(accuracy_score(y_test, y_pred), 4)
print(f"LightGBM Accuracy: {acc}")

cls_report = classification_report(
    y_test, y_pred, target_names=["Low", "Medium", "High"], output_dict=True
)

metrics = {
    "accuracy": acc,
    "classification_report": cls_report
}

with open("models/lgbm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("✓ Saved: models/lgbm_metrics.json")

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "models/lgbm_model.pkl")
print("✓ Saved: models/lgbm_model.pkl")

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)
labels = ["Low", "Medium", "High"]

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - LightGBM", fontsize=14)
plt.tight_layout()
plt.savefig("results/lgbm_confusion_matrix.png", dpi=300)
plt.close()
print("✓ Saved: results/lgbm_confusion_matrix.png")

# ---------------- ROC CURVE ----------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))
for i, cls in enumerate(["Low", "Medium", "High"]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{cls} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LightGBM (3-Class)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/lgbm_roc_curve.png", dpi=300)
plt.close()
print("✓ Saved: results/lgbm_roc_curve.png")
