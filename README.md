# Wine Quality Classification – Complete Project 

This README contains **everything required for your project submission**, including:

✔ Project explanation  
✔ How to run  
✔ Dataset info  
✔ Feature engineering  
✔ Model descriptions  
✔ Full Python code (all 6 ML models in one script)  
✔ Output description  

The instructor needs **ONLY this file** + `wine.csv`.

---

# 📌 1. Project Overview

Wine quality prediction is essential for evaluating wine taste and chemical balance.  
This project uses **6 supervised ML models** to classify wine quality into:

- **0 = Low Quality (≤ 5)**  
- **1 = Medium Quality (= 6)**  
- **2 = High Quality (≥ 7)**  

The file `Ml_Models_Wine_AllInOne.py` trains 6 models:

1. CatBoost  
2. Gradient Boosting  
3. LightGBM  
4. Random Forest  
5. Support Vector Machine (SVM)  
6. XGBoost  

All models are trained, evaluated, and saved automatically.

---

# 📌 2. How to Run (VS Code / Terminal)

### Step 1 — Install Dependencies

```bash
pip install numpy pandas seaborn matplotlib scikit-learn joblib catboost lightgbm xgboost

ML_Algorithms/
│── Ml_Models_Wine_AllInOne.py
│── wine.csv
│── README.md

python Ml_Models_wine.py

