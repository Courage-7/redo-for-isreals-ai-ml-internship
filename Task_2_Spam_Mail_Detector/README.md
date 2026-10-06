# 📧 Spam Mail & SMS Detector — Machine Learning NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-green.svg)](https://www.nltk.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

A complete, production-grade Natural Language Processing (NLP) classification pipeline built for the **QSkill Artificial Intelligence & Machine Learning Internship (Task 2)**.

---

## 📌 Project Overview & Objectives
The objective of this project is to build an intelligent NLP classifier that reliably identifies and filters **Spam** (unsolicited commercial offers, phishing, prize scams) while ensuring that legitimate communication (**Ham**) is never falsely blocked.

---

## 📂 Dataset Provenance
In strict compliance with the task specification (*"Public datasets like the SMS Spam Collection (UCI) or Enron Email Dataset"*):
- **Authentic UCI Repository**: Sourced directly from the [UCI Machine Learning Repository SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
- **Local Storage**: Retained locally under `data/SMSSpamCollection.csv` and `data/SMSSpamCollection` alongside the original authors' documentation (`data/uci_readme.txt`).
- **Volume**: 5,574 labeled SMS messages.
  - **Ham**: 4,827 messages (~86.6%)
  - **Spam**: 747 messages (~13.4%)

---

## 🏗️ Repository Architecture

```
Spam_Mail_Detector_Redo/
├── data/
│   ├── SMSSpamCollection      # Raw tab-separated dataset from UCI
│   ├── SMSSpamCollection.csv  # Clean CSV formatted dataset
│   ├── uci_readme.txt         # Official UCI collection documentation
│   └── README.md              # Dataset provenance documentation
├── models/
│   ├── spam_pipeline.joblib   # Serialized end-to-end Pipeline (Tfidf + Classifier)
│   ├── spam_best_model.joblib # Serialized champion estimator
│   ├── tfidf_vectorizer.joblib# Serialized TF-IDF vocabulary and weights
│   └── README.md              # Model artifacts description
├── reports/
│   ├── project_report.md      # Comprehensive technical research report
│   └── model_metrics.csv      # Comparative metrics table across all models
├── results/
│   ├── class_distribution.png # Target class imbalance breakdown (bar & donut)
│   ├── message_length_distribution.png # Character length & word count distributions
│   ├── top_spam_words.png     # Top 15 distinctive tokens in Spam vs. Ham
│   ├── model_comparison.png   # Performance benchmark chart
│   ├── confusion_matrix_comparison.png # Confusion matrices for all models
│   └── README.md              # Exported visual assets catalog
├── Spam_Mail_Detector.ipynb   # Fully executed Jupyter notebook with rich outputs
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🔬 Methodology & Key Innovations

1. **Information-Preserving Preprocessing**:
   - Rather than crudely stripping punctuation, the cleaner normalizes entities into semantic tokens:
     - URLs $\rightarrow$ `httpaddr`
     - Email addresses $\rightarrow$ `emailaddr`
     - Currency signs (`$`, `£`, `€`) $\rightarrow$ `currencysign` (primary spam signal)
     - Long number strings / phone numbers $\rightarrow$ `numbr`
   - Applies NLTK stopword filtering and `WordNetLemmatizer` for linguistic consistency.
2. **Feature Extraction**:
   - `TfidfVectorizer` with both unigram and bigram ranges (`ngram_range=(1, 2)`), `min_df=2`, and `max_df=0.95`.
3. **Multi-Model Benchmark & Cross-Validation**:
   - Compared **Multinomial Naive Bayes**, **Logistic Regression (Class Weighted)**, and **Linear Support Vector Classifier (LinearSVC)**.
   - Evaluated generalizability using **5-Fold Stratified Cross-Validation**.
4. **Precision-First Evaluation**:
   - Because False Positives (dropping legitimate mail) are far more damaging than False Negatives, models are evaluated on Spam Precision alongside F1-score and Accuracy.
5. **Production Deployment**:
   - Persisted `Pipeline` accepts raw, uncleaned text strings and returns instantaneous classification with probability/confidence scoring.
   - Automated error analysis reporting misclassified samples.

---

## 📊 Benchmark Summary

| Model | Test Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVC** | **99.01%** | 97.26% | **95.30%** | **96.27%** | **99.01%** |
| **Multinomial Naive Bayes** | 98.74% | **97.87%** | 92.62% | 95.17% | 98.73% |
| **Logistic Regression** | 98.48% | 92.86% | 95.97% | 94.39% | 98.49% |

*See [`reports/project_report.md`](reports/project_report.md) for complete experimental details, false-positive breakdowns, and discussion.*

---

## 🚀 How to Run

### Local Setup
```bash
# Navigate to the folder
cd Spam_Mail_Detector_Redo

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Spam_Mail_Detector.ipynb
```

### Google Colab
1. Upload `Spam_Mail_Detector.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Upload the `data/` directory or run the automatic fallback loader.
3. Select **Runtime → Run all**.
