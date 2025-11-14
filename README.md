# Wine Quality Prediction using Machine Learning

A comprehensive comparative analysis of six advanced machine learning algorithms for multi-class wine quality prediction using physicochemical properties.

## 📋 Table of Contents
- [Overview](#overview)
- [Authors](#authors)
- [Dataset](#dataset)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

This project presents a comprehensive comparative analysis of six state-of-the-art machine learning algorithms for predicting wine quality based on physicochemical properties. The study evaluates performance across multiple metrics and provides insights into computational efficiency trade-offs for practical deployment scenarios.

**Key Highlights:**
- 📊 Analysis of 6,497 wine samples (1,599 red + 4,898 white wines)
- 🎯 3-class quality categorization (Low/Medium/High)
- 🏆 XGBoost achieves highest accuracy: 72.62%
- ⚡ Comprehensive performance benchmarking
- 📈 Feature engineering with domain-specific interactions

## 👥 Authors

**Venkata Kalyan Chittiboina** 
**Soorneedi Poorna Naga Sujit**  
**Vatam Rohith Reddy** 
**Yerukali Punarvitha** 

*Department of Electronics and Communication Engineering*  
*Indian Institute of Information Technology, Sri City*

## 📊 Dataset

### Wine Quality Dataset
- **Source:** UCI Machine Learning Repository
- **Samples:** 6,497 total (1,599 red + 4,898 white wines)
- **Features:** 11 physicochemical properties
- **Target:** Wine quality scores (3-8)

### Physicochemical Features
1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

### Engineered Features
- **Total Acidity:** Sum of fixed, volatile acidity and citric acid
- **Sulfur Ratio:** Free SO₂ / Total SO₂
- **Sugar per Acid:** Residual Sugar / Total Acidity
- **Acid-Sugar Ratio:** Inverse ratio
- **Density-Alcohol Ratio:** Interaction feature
- **Binary Indicators:** High alcohol and high sugar flags

### Quality Categorization
- **Low Quality:** Scores ≤ 5
- **Medium Quality:** Score = 6
- **High Quality:** Scores ≥ 7

## 🤖 Algorithms Implemented

### 1. Support Vector Machine (SVM)
- **Kernel:** RBF (Radial Basis Function)
- **Parameters:** C=1.0, gamma='scale'
- **Accuracy:** 60.98%

### 2. Random Forest (RF)
- **Estimators:** 500
- **Max Depth:** 15
- **Min Samples Split:** 10
- **Accuracy:** 70.46%

### 3. Gradient Boosting (GB)
- **Estimators:** 400
- **Learning Rate:** 0.05
- **Max Depth:** 5
- **Accuracy:** 64.06%

### 4. XGBoost
- **Estimators:** 500
- **Learning Rate:** 0.05
- **Max Depth:** 6
- **Subsample:** 0.8
- **Accuracy:** 72.62% 🏆

### 5. LightGBM
- **Estimators:** 600
- **Learning Rate:** 0.05
- **Num Leaves:** 35
- **Subsample:** 0.9
- **Accuracy:** 71.32%

### 6. CatBoost
- **Iterations:** 500
- **Learning Rate:** 0.05
- **Depth:** 6
- **Accuracy:** 69.00%

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
```

### Required Libraries
```bash
pip install numpy pandas scikit-learn
pip install xgboost lightgbm catboost
pip install matplotlib seaborn
pip install jupyter notebook
```

### Clone Repository
```bash
git clone https://github.com/kalyan021004/WineQuality_PR.git
cd WineQuality_PR
```

## 📁 Project Structure

```
WineQuality_PR/
│
├── Ml_models/
│   ├── wine_catboost.py
│   └── wine_gradientboost.py
|   ├── wine_lightgbm.py
│   ├── wine_randomforest.py
│   ├── wine_svm.py
│   └── wine_xgboost.py
|    
|
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Usage

### 1. Data Preprocessing
```python
from src.data_preprocessing import load_and_preprocess_data

# Load and preprocess wine quality data
X_train, X_test, y_train, y_test = load_and_preprocess_data()
```

### 2. Feature Engineering
```python
from src.feature_engineering import engineer_features

# Create engineered features
X_train_engineered = engineer_features(X_train)
X_test_engineered = engineer_features(X_test)
```

### 3. Train Models
```python
from src.model_training import train_all_models

# Train all six algorithms
models = train_all_models(X_train, y_train)
```

### 4. Evaluate Performance
```python
from src.model_evaluation import evaluate_models

# Evaluate and compare all models
results = evaluate_models(models, X_test, y_test)
print(results)
```

### 5. Generate Visualizations
```python
from src.visualization import plot_confusion_matrix, plot_roc_curves

# Generate confusion matrices and ROC curves
for model_name, model in models.items():
    plot_confusion_matrix(model, X_test, y_test, model_name)
    plot_roc_curves(model, X_test, y_test, model_name)
```

### Running Complete Pipeline
```bash
# Run the complete analysis
python main.py
```

## 📈 Results

### Performance Comparison

| Algorithm | Precision | Recall | F1-Score | Accuracy |
|-----------|-----------|--------|----------|----------|
| **XGBoost** | **0.73** | **0.73** | **0.73** | **72.62%** |
| LightGBM | 0.72 | 0.71 | 0.71 | 71.32% |
| Random Forest | 0.71 | 0.70 | 0.70 | 70.46% |
| CatBoost | 0.70 | 0.69 | 0.69 | 69.00% |
| Gradient Boosting | 0.65 | 0.64 | 0.64 | 64.06% |
| SVM | 0.61 | 0.61 | 0.61 | 60.98% |

### Performance Ranking
**XGBoost (72.6%) > LightGBM (71.3%) > RF (70.5%) > CatBoost (69.0%) > GB (64.1%) > SVM (61.0%)**

### Key Findings
- ✅ XGBoost achieves the highest accuracy at 72.62%
- ✅ LightGBM offers excellent balance between accuracy and speed
- ✅ Random Forest provides good interpretability with competitive performance
- ✅ Traditional Gradient Boosting underperforms compared to modern variants
- ✅ SVM struggles with the multi-class imbalanced dataset

### Deployment Recommendations

| Use Case | Recommended Algorithm | Reason |
|----------|----------------------|---------|
| **Maximum Accuracy** | XGBoost | Highest F1-score (0.73) |
| **Production Balance** | LightGBM | Fast training, competitive accuracy |
| **Interpretability** | Random Forest | Feature importance analysis |

## 📄 Research Paper

This work has been documented in a comprehensive research paper:

**Title:** *A Comparative Analysis of Advanced Machine Learning Algorithms for Multi-Class Predictive Modeling of Wine Quality*

**Abstract:** This paper presents a comprehensive comparative analysis of six advanced machine learning algorithms for multi-class wine quality prediction using physicochemical properties...

[Read Full Paper](./paper/Wine_Quality_Paper.pdf)

## 🔬 Methodology

### Data Pipeline
1. **Data Collection:** Load red and white wine datasets
2. **Preprocessing:** Handle missing values, encode wine type
3. **Feature Engineering:** Create domain-specific interaction features
4. **Normalization:** StandardScaler (zero mean, unit variance)
5. **Train-Test Split:** Stratified 75-25 split (random_state=42)
6. **Model Training:** Train six algorithms with optimized hyperparameters
7. **Evaluation:** Comprehensive metrics including confusion matrices and ROC curves

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or collaboration opportunities, please contact:

- **Venkata Kalyan Chittiboina** - [GitHub Profile](https://github.com/kalyan021004)
- **Project Link:** [https://github.com/kalyan021004/WineQuality_PR](https://github.com/kalyan021004/WineQuality_PR)

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{chittiboina2024wine,
  title={A Comparative Analysis of Advanced Machine Learning Algorithms for Multi-Class Predictive Modeling of Wine Quality},
  author={Chittiboina, Venkata Kalyan and Sujit, Soorneedi Poorna Naga and Reddy, Vatam Rohith and Punarvitha, Yerukali},
  journal={IEEE Conference},
  year={2024},
  institution={Indian Institute of Information Technology, Sri City}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the Wine Quality Dataset
- Indian Institute of Information Technology, Sri City for research support
- Open-source community for the excellent ML libraries (scikit-learn, XGBoost, LightGBM, CatBoost)

---

**⭐ If you find this project helpful, please consider giving it a star!**

**Made with ❤️ by IIIT Sri City ECE Students**
