# Iris Flower Classification Pipeline

Implementation of the Iris Flower Classification project for Task 1 of the AI and Machine Learning internship.

### Project Overview and Objectives
The goal of this project is to build, evaluate, and serialize a machine learning classification pipeline capable of classifying Iris flowers into three distinct species:
* Iris setosa
* Iris versicolor
* Iris virginica

Classification is based on four morphological features:
1. sepal_length (cm)
2. sepal_width (cm)
3. petal_length (cm)
4. petal_width (cm)

### Dataset Provenance
* Source: Official UCI Machine Learning Repository (Dataset ID: 53) retrieved via `ucimlrepo` client API in memory.
* Volume: 150 instances (50 per class), balanced 1:1:1, with 0 missing values.
* Local Reference: A reference copy is preserved under `data/iris.csv` for offline review.

### Directory Architecture

```
Task_1_Iris_Flower_Classification/
├── data/
│   ├── iris.csv               # Official UCI dataset
│   └── README.md              # Dataset metadata description
├── models/
│   ├── iris_pipeline.joblib   # Serialized Pipeline (StandardScaler + Best Classifier)
│   ├── iris_best_model.joblib # Serialized champion estimator
│   ├── iris_scaler.joblib     # Serialized StandardScaler
│   └── README.md              # Model artifact registry
├── reports/
│   ├── project_report.md      # Technical project report
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

### Methodology

1. Exploratory Data Analysis (EDA):
   * Visualized univariate distributions, box plots, and correlation matrices.
   * Identified that petal_length and petal_width are strongly correlated (r = 0.963) and provide near-linear separation for Iris setosa.
2. Preprocessing:
   * Stratified 80/20 train/test split (random_state=42) preserving class balance.
   * Encapsulated feature scaling (StandardScaler) within `sklearn.pipeline.Pipeline` to prevent data leakage during cross-validation and testing.
3. Multi-Model Benchmarking:
   * Compared Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Classifier.
   * Validated performance using 5-Fold Stratified Cross-Validation.
4. Evaluation:
   * Assessed using Accuracy, Macro/Weighted Precision, Recall, F1-Score, and Confusion Matrices.
5. Model Serialization:
   * Exported trained inference pipelines using `joblib`.
   * Verified predictions against unseen inputs.

### Benchmark Summary

| Model | Cross Val Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Support Vector Classifier | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| K Nearest Neighbors | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9583 | 0.9667 | 0.9697 | 0.9667 | 0.9666 |
| Decision Tree | 0.9417 | 0.9667 | 0.9697 | 0.9667 | 0.9666 |

### Setup and Execution

To run the notebook locally:

```bash
cd Task_1_Iris_Flower_Classification
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Iris_Flower_Classification.ipynb
```
