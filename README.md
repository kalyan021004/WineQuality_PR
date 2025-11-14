# 🍷 Wine Quality Prediction Dashboard

<div align="center">

![Wine ML Banner](https://img.shields.io/badge/Machine%20Learning-Wine%20Quality-red?style=for-the-badge&logo=python)
![Node.js](https://img.shields.io/badge/Node.js-43853D?style=for-the-badge&logo=node.js&logoColor=white)
![Express.js](https://img.shields.io/badge/Express.js-404D59?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

**A comprehensive full-stack web application for visualizing and comparing machine learning model performance in wine quality classification**

[🚀 Live Demo](https://winequalityclassification.onrender.com/) 

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Machine Learning Models](#-machine-learning-models)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [API Documentation](#-api-documentation)
- [Screenshots](#-screenshots)
- [Performance Metrics](#-performance-metrics)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#-authors)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project presents a **comparative analysis dashboard** for six advanced machine learning algorithms applied to multi-class wine quality prediction. Using physicochemical properties of wines, the system classifies wine samples into three quality tiers: **Low**, **Medium**, and **High**.

### 🎓 Academic Context

This project is part of a Pattern Recognition course at **Indian Institute of Information Technology, Sri City** and demonstrates practical applications of ensemble learning, gradient boosting, and support vector machines in real-world classification tasks.

### 🔬 Research Highlights

- Evaluated **6 state-of-the-art ML algorithms** on 6,497 wine samples
- Achieved **72.62% accuracy** with XGBoost (best performer)
- Comprehensive analysis of class imbalance challenges
- Production-ready deployment with interactive visualizations

---

## ✨ Features

### 🤖 Machine Learning Capabilities

- **Six Pre-trained Models**: XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting, SVM
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score (per class and weighted)
- **Visual Analytics**: Confusion matrices, ROC curves with AUC scores
- **Feature Engineering**: 8 engineered features from 11 physicochemical properties



### ⚡ Performance

- **Fast Loading**: Optimized static asset delivery
- **Real-time Rendering**: Dynamic EJS templating
- **Scalable Architecture**: RESTful API design

---

## 🛠 Tech Stack

### Backend
![Node.js](https://img.shields.io/badge/Node.js-v18+-339933?style=flat&logo=node.js&logoColor=white)
![Express](https://img.shields.io/badge/Express-v4.18+-000000?style=flat&logo=express&logoColor=white)
![EJS](https://img.shields.io/badge/EJS-Template%20Engine-B4CA65?style=flat)

### Frontend
![Bootstrap](https://img.shields.io/badge/Bootstrap-v5.3-7952B3?style=flat&logo=bootstrap&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

### Machine Learning
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EA4C89?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)

### Deployment
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)

---

## 🧠 Machine Learning Models

### Algorithms Implemented

| Algorithm | Accuracy | F1-Score | Training Time | Status |
|-----------|----------|----------|---------------|--------|
| **XGBoost** | **72.62%** | **0.73** | ~12s | ✅ Best |
| **LightGBM** | 71.32% | 0.71 | ~8s | ✅ Fast |
| **Random Forest** | 70.46% | 0.70 | ~15s | ✅ Stable |
| **CatBoost** | 69.00% | 0.69 | ~10s | ✅ Good |
| **Gradient Boosting** | 64.06% | 0.64 | ~18s | ⚠️ Baseline |
| **SVM (RBF)** | 60.98% | 0.61 | ~25s | ⚠️ Baseline |

### Model Artifacts

Each model includes:
- 📦 **Serialized Model**: `.pkl` file for inference
- 📊 **Performance Metrics**: JSON with precision, recall, F1-score
- 🎨 **Confusion Matrix**: High-resolution PNG visualization
- 📈 **ROC Curve**: Multi-class ROC with AUC scores

### Dataset Information

- **Total Samples**: 6,497 (1,599 red + 4,898 white wines)
- **Features**: 11 physicochemical + 8 engineered = 19 total
- **Classes**: 3 (Low ≤5, Medium =6, High ≥7)
- **Split**: 75% training, 25% testing (stratified)
- **Source**: UCI Machine Learning Repository

### Feature Engineering

```python
# Core Features (11)
fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
density, pH, sulphates, alcohol

# Engineered Features (8)
total_acidity = fixed + volatile + citric
sulfur_ratio = free_sulfur / total_sulfur
sugar_per_acid = residual_sugar / total_acidity
acid_sugar_ratio = total_acidity / residual_sugar
density_alcohol_ratio = density / alcohol
high_alcohol = (alcohol > median) ? 1 : 0
high_sugar = (sugar > median) ? 1 : 0
```

---

## 📁 Project Structure

```
WineQuality_PR/
│
├── 📄 server.js                 # Express.js backend server
├── 📄 package.json              # Node.js dependencies
├── 📄 render.yaml               # Render deployment config
├── 📄 README.md                 # This file
│
├── 📂 models/                   # ML models and metrics
│   ├── xgb_model.pkl           # XGBoost trained model
│   ├── xgb_metrics.json        # XGBoost performance metrics
│   ├── lgbm_model.pkl          # LightGBM trained model
│   ├── lgbm_metrics.json       # LightGBM performance metrics
│   ├── catboost_model.pkl      # CatBoost trained model
│   ├── catboost_metrics.json   # CatBoost performance metrics
│   ├── rf_model.pkl            # Random Forest trained model
│   ├── rf_metrics.json         # Random Forest metrics
│   ├── gb_model.pkl            # Gradient Boosting model
│   ├── gb_metrics.json         # Gradient Boosting metrics
│   ├── svm_model.pkl           # SVM trained model
│   └── svm_metrics.json        # SVM performance metrics
│
├── 📂 results/                  # Visualization outputs
│   ├── xgb_confusion_matrix.png
│   ├── xgb_roc_curve.png
│   ├── lgbm_confusion_matrix.png
│   ├── lgbm_roc_curve.png
│   ├── catboost_confusion_matrix.png
│   ├── catboost_roc_curve.png
│   ├── rf_confusion_matrix.png
│   ├── rf_roc_curve.png
│   ├── gb_confusion_matrix.png
│   ├── gb_roc_curve.png
│   ├── svm_confusion_matrix.png
│   ├── svm_roc_curve.png
│   ├── performance_comparison.png
│   └── accuracy_vs_speed.png
│
├── 📂 ML_models/                # Python training scripts
│   ├── wine_xgboost.py
│   ├── wine_lightgbm.py
│   ├── wine_catboost.py
│   ├── wine_randomforest.py
│   ├── wine_gradientboost.py
│   ├── wine_svm.py
│   └── data_preprocessing.py
│
├── 📂 views/                    # EJS templates
│   ├── index.ejs               # Home page / Algorithm list
│   ├── algorithm.ejs           # Individual model details
│   └── partials/
│       ├── header.ejs
│       └── footer.ejs
│
├── 📂 public/                   # Static assets
│   ├── css/
│   │   └── style.css           # Custom styles
│   ├── js/
│   │   └── main.js             # Client-side scripts
│   └── images/
│       └── logo.png
│
└── 📂 docs/                     # Documentation
    ├── research_paper.tex      # LaTeX research paper
    ├── methodology.md
    └── api_docs.md
```

---

## 🚀 Installation

### Prerequisites

- **Node.js** (v18 or higher)
- **npm** (v9 or higher)
- **Python** 3.8+ (for ML training scripts)
- **Git**

### Local Development Setup

#### 1️⃣ Clone Repository

```bash
git clone https://github.com/kalyan021004/WineQuality_PR.git
cd WineQuality_PR
```

#### 2️⃣ Install Node Dependencies

```bash
npm install
```

#### 3️⃣ Verify ML Models & Results

Ensure the following directories exist with files:
```bash
ls models/     # Should contain *.pkl and *.json files
ls results/    # Should contain *.png files
```

#### 4️⃣ Start Development Server

```bash
npm start
```

Or with auto-reload:
```bash
npm run dev
```

#### 5️⃣ Open Browser

Navigate to: **http://localhost:3000**

---

## 🎮 Usage

### Viewing Dashboard

1. **Home Page**: Displays all 6 algorithms with overview cards
2. **Click Algorithm Card**: Navigate to detailed model page
3. **View Metrics**: Confusion matrix, ROC curve, and performance scores
4. **Compare Models**: Return to home to compare different algorithms

### Training New Models (Optional)

If you want to retrain models:

```bash
cd ML_models

# Train individual models
python wine_xgboost.py
python wine_lightgbm.py
python wine_catboost.py
python wine_randomforest.py
python wine_gradientboost.py
python wine_svm.py
```

Models and visualizations will be saved to `models/` and `results/` directories.

---

## 🌐 Deployment

### Deploy to Render

#### Method 1: Automatic Deployment (Recommended)

1. **Push to GitHub**:
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`

3. **Deploy**:
   - Click **"Create Web Service"**
   - Wait for build to complete (~2-3 minutes)
   - Access your live app!

#### Method 2: Manual Configuration

If you prefer manual setup:

```yaml
# render.yaml
services:
  - type: web
    name: wine-ml-dashboard
    env: node
    buildCommand: npm install
    startCommand: node server.js
    envVars:
      - key: NODE_ENV
        value: production
```

**Render Settings**:
- **Build Command**: `npm install`
- **Start Command**: `node server.js`
- **Environment**: Node
- **Plan**: Free (or upgrade for better performance)

### Deploy to Other Platforms

<details>
<summary>Heroku Deployment</summary>

```bash
heroku login
heroku create wine-ml-dashboard
git push heroku main
heroku open
```
</details>

<details>
<summary>Vercel Deployment</summary>

```bash
npm install -g vercel
vercel
```
</details>

<details>
<summary>AWS EC2 / DigitalOcean</summary>

```bash
# SSH into server
ssh user@your-server-ip

# Clone and setup
git clone https://github.com/kalyan021004/WineQuality_PR.git
cd WineQuality_PR
npm install

# Use PM2 for process management
npm install -g pm2
pm2 start server.js --name wine-dashboard
pm2 save
pm2 startup
```
</details>

---

## 📚 API Documentation

### Endpoints

#### `GET /`
**Description**: Home page with algorithm list  
**Response**: HTML page with all models

#### `GET /algorithm/:name`
**Description**: Detailed view of specific algorithm  
**Parameters**: 
- `name` (string): Algorithm identifier (xgb, lgbm, catboost, rf, gb, svm)

**Response**: HTML page with model details

#### `GET /api/metrics`
**Description**: Get all model metrics (JSON API)  
**Response**:
```json
{
  "xgb": {
    "accuracy": 0.7262,
    "precision": 0.73,
    "recall": 0.73,
    "f1_score": 0.73,
    "class_metrics": {...}
  },
  ...
}
```

---

## 📸 Screenshots

### Home Dashboard
![Dashboard Home](https://via.placeholder.com/800x400/4A90E2/ffffff?text=Algorithm+Comparison+Dashboard)

### XGBoost Model Details
![XGBoost Details](https://via.placeholder.com/800x400/50C878/ffffff?text=XGBoost+Performance+Metrics)

### Confusion Matrix
![Confusion Matrix](https://via.placeholder.com/400x400/FF6B6B/ffffff?text=Confusion+Matrix)

### ROC Curve
![ROC Curve](https://via.placeholder.com/400x400/4ECDC4/ffffff?text=Multi-Class+ROC+Curve)

---

## 📊 Performance Metrics

### Overall Comparison

```
┌─────────────────────┬──────────┬───────────┬────────┬──────────┐
│ Algorithm           │ Accuracy │ Precision │ Recall │ F1-Score │
├─────────────────────┼──────────┼───────────┼────────┼──────────┤
│ XGBoost            │ 72.62%   │ 0.73      │ 0.73   │ 0.73     │
│ LightGBM           │ 71.32%   │ 0.72      │ 0.71   │ 0.71     │
│ Random Forest      │ 70.46%   │ 0.71      │ 0.70   │ 0.70     │
│ CatBoost           │ 69.00%   │ 0.70      │ 0.69   │ 0.69     │
│ Gradient Boosting  │ 64.06%   │ 0.65      │ 0.64   │ 0.64     │
│ SVM (RBF)          │ 60.98%   │ 0.61      │ 0.61   │ 0.61     │
└─────────────────────┴──────────┴───────────┴────────┴──────────┘
```

### Per-Class Performance (XGBoost - Best Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low (0) | 0.68 | 0.82 | 0.74 | 320 |
| Medium (1) | 0.76 | 0.72 | 0.74 | 875 |
| High (2) | 0.78 | 0.58 | 0.67 | 429 |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**:
```bash
git checkout -b feature/AmazingFeature
```
3. **Commit changes**:
```bash
git commit -m 'Add some AmazingFeature'
```
4. **Push to branch**:
```bash
git push origin feature/AmazingFeature
```
5. **Open Pull Request**

### Contribution Ideas

- 🎨 UI/UX improvements
- 📊 Additional visualization types
- 🧠 New ML algorithms (Neural Networks, Stacking, etc.)
- 📱 Mobile app version
- 🔍 Feature importance analysis (SHAP values)
- 📈 Real-time prediction API
- 🌍 Multi-language support

### Code Style

- Follow JavaScript Standard Style
- Use meaningful variable names
- Comment complex logic
- Write descriptive commit messages

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Venkata Kalyan Chittiboina

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 👥 Authors

### Research & Development Team

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/kalyan021004.png" width="100px;" alt="Kalyan"/><br />
      <sub><b>Venkata Kalyan Chittiboina</b></sub><br />
      <sub>Lead Developer & ML Engineer</sub><br />
      <a href="https://github.com/kalyan021004">GitHub</a>
    </td>
    <td align="center">
      <sub><b>Soorneedi Poorna Naga Sujit</b></sub><br />
      <sub>Data Engineer & ML Specialist</sub><br />
      <sub>S20230020351</sub>
    </td>
    <td align="center">
      <sub><b>Vatam Rohith Reddy</b></sub><br />
      <sub>Backend Developer & Analyst</sub><br />
      <sub>S20230020357</sub>
    </td>
    <td align="center">
      <sub><b>Yerukali Punarvitha</b></sub><br />
      <sub>Frontend Developer & QA</sub><br />
      <sub>S20230020361</sub>
    </td>
  </tr>
</table>

### Institutional Affiliation

**Indian Institute of Information Technology, Sri City**  
Department of Electronics and Communication Engineering  
Pattern Recognition Course Project (2024)

---

## 🙏 Acknowledgments

- **Dataset**: [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Libraries**: XGBoost, LightGBM, CatBoost, scikit-learn, pandas, matplotlib, seaborn
- **Frameworks**: Express.js, Bootstrap, EJS
- **Inspiration**: Research papers on ensemble learning and gradient boosting
- **Course Instructor**: [Instructor Name], Pattern Recognition Course
- **IIIT Sri City** for providing computational resources

---

## 📞 Contact & Support

- 📧 **Email**: [s20230020358@iiits.in](mailto:s20230020358@iiits.in)
- 🐛 **Issues**: [GitHub Issues](https://github.com/kalyan021004/WineQuality_PR/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/kalyan021004/WineQuality_PR/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/kalyan021004/WineQuality_PR/wiki)

---

## 🌟 Show Your Support

If you find this project helpful, please consider:

- ⭐ **Starring** the repository
- 🍴 **Forking** for your own use
- 📢 **Sharing** with others
- 🐛 **Reporting** bugs and issues
- 💡 **Suggesting** new features

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/kalyan021004/WineQuality_PR?style=social)](https://github.com/kalyan021004/WineQuality_PR/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/kalyan021004/WineQuality_PR?style=social)](https://github.com/kalyan021004/WineQuality_PR/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/kalyan021004/WineQuality_PR?style=social)](https://github.com/kalyan021004/WineQuality_PR/watchers)

**Made with ❤️ by the IIIT Sri City ML Team**

</div>

---

## 📈 Project Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/kalyan021004/WineQuality_PR)
![GitHub code size](https://img.shields.io/github/languages/code-size/kalyan021004/WineQuality_PR)
![Lines of code](https://img.shields.io/tokei/lines/github/kalyan021004/WineQuality_PR)
![GitHub last commit](https://img.shields.io/github/last-commit/kalyan021004/WineQuality_PR)

---

<div align="center">

**🍷 Enjoy Exploring Machine Learning for Wine Quality Prediction! 🍷**

[⬆ Back to Top](#-wine-quality-prediction-dashboard)

</div>