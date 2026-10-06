# Redo for Isreal's AI/ML Internship

This repository provides the complete, production grade implementations of two applied machine learning tasks for the AI and Machine Learning internship program. Both solutions have been re engineered from the ground up to follow strict modular architecture, automated remote data ingestion, leak free preprocessing pipelines, comprehensive model benchmarks, and serialized deployment artifacts.

### Repository Architecture

```
redo_for_isreals_ai_ml_internship/
├── .gitignore
├── README.md
├── requirements.txt
│
├── Task_1_Iris_Flower_Classification/
│   ├── Iris_Flower_Classification.ipynb
│   ├── requirements.txt
│   ├── README.md
│   ├── data/
│   │   ├── iris.csv
│   │   └── README.md
│   ├── models/
│   │   ├── iris_pipeline.joblib
│   │   ├── iris_best_model.joblib
│   │   ├── iris_scaler.joblib
│   │   └── README.md
│   ├── reports/
│   │   ├── project_report.md
│   │   └── metrics_summary.csv
│   └── results/
│       ├── eda_distributions.png
│       ├── eda_boxplots.png
│       ├── eda_correlation.png
│       ├── eda_pairplot.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       └── README.md
│
└── Task_2_Spam_Mail_Detector/
    ├── Spam_Mail_Detector.ipynb
    ├── requirements.txt
    ├── README.md
    ├── data/
    │   ├── SMSSpamCollection
    │   ├── SMSSpamCollection.csv
    │   ├── uci_readme.txt
    │   └── README.md
    ├── models/
    │   ├── spam_pipeline.joblib
    │   ├── spam_best_model.joblib
    │   ├── tfidf_vectorizer.joblib
    │   └── README.md
    ├── reports/
    │   ├── project_report.md
    │   └── model_metrics.csv
    └── results/
        ├── class_distribution.png
        ├── message_length_distribution.png
        ├── top_spam_words.png
        ├── model_comparison.png
        ├── confusion_matrix_comparison.png
        └── README.md
```

### Executive Summary of Completed Tasks

#### Task 1: Iris Flower Classification
* Objective: Classify iris specimens into one of three species (Iris setosa, Iris versicolor, Iris virginica) based on sepal and petal measurements.
* Dataset: UCI Machine Learning Repository (Dataset ID: 53), 150 instances, 4 numeric continuous features, 0 missing values.
* Data Ingestion: Automated live retrieval via the official `ucimlrepo` client API in memory.
* Feature Engineering and Scaling: StandardScaler fitted strictly on the training partition inside an isolated pipeline.
* Models Evaluated: Logistic Regression, Support Vector Classifier (RBF kernel), Random Forest Classifier, K Nearest Neighbors, Decision Tree Classifier.
* Cross Validation: Stratified 5 fold cross validation to preserve class balance across all folds.
* Champion Model: Support Vector Classifier achieved 100% test accuracy, 100% precision, 100% recall, and 1.0000 macro F1 score on the held out test set.
* Artifacts: End to end inference pipeline serialized via `joblib` into `models/iris_pipeline.joblib`.

#### Task 2: Spam Mail Detector
* Objective: Detect and classify unsolicited spam communications versus legitimate ham messages.
* Dataset: UCI Machine Learning Repository SMS Spam Collection (Dataset ID: 228), 5,572 messages (86.6% Ham, 13.4% Spam).
* Data Ingestion: Direct in memory HTTP streaming and extraction from the official UCI static archive repository.
* Text Processing: Regular expression cleaning, lowercase normalization, tokenization, English stopword elimination, and sublinear TF IDF feature extraction with 3,000 maximum features and n gram range (1, 2).
* Models Evaluated: Multinomial Naive Bayes, Logistic Regression, Linear Support Vector Classifier, Random Forest Classifier.
* Primary Evaluation Metric: Precision on Spam class and F1 score, prioritizing zero false positives (ensuring legitimate communications are never misclassified as spam).
* Champion Model: Multinomial Naive Bayes achieved 98.48% accuracy, 100% Spam precision (0 false positives), and 0.9412 macro F1 score on the held out test set.
* Artifacts: End to end text vectorization and classification pipeline serialized via `joblib` into `models/spam_pipeline.joblib`.

### Performance Benchmarks

#### Task 1 Benchmark Summary
| Model | Cross Val Accuracy | Test Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Support Vector Classifier | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| K Nearest Neighbors | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9583 | 0.9667 | 0.9697 | 0.9667 | 0.9666 |
| Decision Tree | 0.9417 | 0.9667 | 0.9697 | 0.9667 | 0.9666 |

#### Task 2 Benchmark Summary
| Model | Test Accuracy | Spam Precision | Spam Recall | Spam F1 | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Multinomial Naive Bayes | 0.9848 | 1.0000 | 0.8867 | 0.9399 | 0.9655 |
| Linear SVC | 0.9839 | 0.9660 | 0.9133 | 0.9384 | 0.9644 |
| Logistic Regression | 0.9686 | 0.9837 | 0.8067 | 0.8864 | 0.9343 |
| Random Forest | 0.9785 | 0.9922 | 0.8467 | 0.9137 | 0.9507 |

### Engineering Standards Applied

1. Zero Local Path Hardcoding: Datasets are retrieved live from official UCI endpoints during execution. Reference copies in `data/` serve solely as offline fallbacks.
2. Leak Free Preprocessing: Scalers and vectorizers are fitted strictly inside `sklearn.pipeline.Pipeline` objects on the training sets.
3. Pre Computed Notebook Execution: All Jupyter notebooks have been pre executed with 100% completed outputs, zero runtime warnings, and clean matplotlib figures.
4. Serialized Inference Pipelines: Deployable pipelines are saved in `.joblib` format, allowing immediate inference without manual preprocessing code.
5. Strict Clean Formatting: Code and markdown adhere strictly to clean standards with zero emojis, no hyphen dashes, and academic typography.

### Installation and Usage

Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/Courage-7/redo-for-isreals-ai-ml-internship.git
cd redo-for-isreals-ai-ml-internship
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run individual tasks, launch Jupyter Lab or Notebook:

```bash
jupyter notebook Task_1_Iris_Flower_Classification/Iris_Flower_Classification.ipynb
jupyter notebook Task_2_Spam_Mail_Detector/Spam_Mail_Detector.ipynb
```
