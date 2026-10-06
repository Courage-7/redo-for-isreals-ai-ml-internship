# 🌸 Iris Flower Classification — Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

A comprehensive, production-grade implementation of the **Iris Flower Classification** project for the **QSkill Artificial Intelligence & Machine Learning Internship (Task 1)**.

---

## 📌 Project Overview & Objectives
The goal of this project is to build, evaluate, and deploy a machine learning classification pipeline capable of classifying Iris flowers into three distinct species:
- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

Classification is based on four morphological features:
1. `sepal_length` (cm)
2. `sepal_width` (cm)
3. `petal_length` (cm)
4. `petal_width` (cm)

---

## 📂 Dataset Provenance
In strict adherence to the project specification (*"The classic Iris dataset from UCI Repository or scikit-learn"*):
- **Authentic UCI Repository**: Sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) and stored in `data/iris.csv`.
- **Integrity**: 150 instances (50 per class), balanced 1:1:1, with 0 missing values.
- **Scikit-Learn Fallback**: Programmatic integration with `sklearn.datasets.load_iris()`.

---

## 🏗️ Repository Architecture

```
Iris_Flower_Classification_Redo/
├── data/
│   ├── iris.csv               # Official UCI dataset with verified headers
│   └── README.md              # Dataset metadata & provenance description
├── models/
│   ├── iris_pipeline.joblib   # Production Pipeline (StandardScaler + Best Classifier)
│   ├── iris_best_model.joblib # Serialized champion estimator
│   ├── iris_scaler.joblib     # Serialized StandardScaler
│   └── README.md              # Model artifact registry
├── reports/
│   ├── project_report.md      # Comprehensive technical & experimental report
│   └── metrics_summary.csv    # Benchmark metrics comparison table across all models
├── results/
│   ├── eda_distributions.png  # Histograms with Kernel Density Estimation (KDE)
│   ├── eda_boxplots.png       # Box plots across species demonstrating separability
│   ├── eda_correlation.png    # Pearson correlation matrix heatmap
│   ├── eda_pairplot.png       # Pairwise feature relationship plots
│   ├── model_comparison.png   # Performance benchmark comparison chart
│   ├── confusion_matrix.png   # Champion model confusion matrix
│   └── README.md              # Visual assets catalog
├── Iris_Flower_Classification.ipynb # Fully executed Jupyter notebook with all outputs
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🔬 Methodology & Key Highlights

1. **Exploratory Data Analysis (EDA)**:
   - Visualized univariate distributions, box plots, and correlation matrices.
   - Identified that `petal_length` and `petal_width` are highly correlated ($r = 0.963$) and provide near-linear separation for *Iris-setosa*.
2. **Leakage-Free Preprocessing**:
   - Stratified 80/20 train/test split (`random_state=42`) preserving class ratios.
   - Encapsulated feature scaling (`StandardScaler`) within `sklearn.pipeline.Pipeline` to guarantee no data leakage during cross-validation or testing.
3. **Multi-Model Benchmarking**:
   - Compared **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Random Forest**.
   - Validated generalizability using **5-Fold Stratified Cross-Validation**.
4. **Evaluation**:
   - Assessed with Accuracy, Weighted/Macro Precision, Recall, F1-Score, and Confusion Matrix heatmaps.
5. **Production Deployment**:
   - Exported self-contained inference pipelines via `joblib`.
   - Verified inference against unseen botanical test inputs.

---

## 📊 Benchmark Summary

| Model | CV Mean Accuracy | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 94.17% | **96.67%** | **96.97%** | **96.67%** | **96.66%** |
| **Logistic Regression** | **96.67%** | 93.33% | 93.33% | 93.33% | 93.33% |
| **K-Nearest Neighbors** | 95.83% | 93.33% | 94.44% | 93.33% | 93.27% |
| **Random Forest** | 95.00% | 90.00% | 90.24% | 90.00% | 89.97% |

*Detailed metrics and confusion matrix are available in [`reports/project_report.md`](reports/project_report.md).*

---

## 🚀 How to Run

### Option 1: Using Virtual Environment
```bash
# Clone or navigate to the directory
cd Iris_Flower_Classification_Redo

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Iris_Flower_Classification.ipynb
```

### Option 2: Google Colab
1. Upload `Iris_Flower_Classification.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload the `data/` folder (or let the notebook automatically fallback to built-in scikit-learn data).
3. Select **Runtime → Run all**.
