# Spam Mail and SMS Detector Pipeline

Implementation of the Spam Mail and SMS Detection project for Task 2 of the AI and Machine Learning internship.

### Project Overview and Objectives
The objective of this project is to build an NLP classifier that identifies Spam messages while ensuring legitimate messages (Ham) are not falsely blocked.

### Dataset Provenance
* Source: UCI Machine Learning Repository SMS Spam Collection (Dataset ID: 228) streamed in memory from the official static archive.
* Volume: 5,572 labeled SMS messages.
  * Ham: 4,825 messages (86.6%)
  * Spam: 747 messages (13.4%)
* Local Reference: A reference copy is preserved under `data/SMSSpamCollection.csv` for offline review.

### Directory Architecture

```
Task_2_Spam_Mail_Detector/
├── data/
│   ├── SMSSpamCollection      # Raw tab-separated dataset from UCI
│   ├── SMSSpamCollection.csv  # Clean CSV formatted dataset
│   ├── uci_readme.txt         # Official UCI collection documentation
│   └── README.md              # Dataset provenance documentation
├── models/
│   ├── spam_pipeline.joblib   # Serialized Pipeline (Tfidf + Classifier)
│   ├── spam_best_model.joblib # Serialized champion estimator
│   ├── tfidf_vectorizer.joblib# Serialized TF-IDF vectorizer
│   └── README.md              # Model artifacts description
├── reports/
│   ├── project_report.md      # Technical project report
│   └── model_metrics.csv      # Comparative metrics table across all models
├── results/
│   ├── class_distribution.png # Target class distribution plot
│   ├── message_length_distribution.png # Character length & word count distributions
│   ├── top_spam_words.png     # Top distinguishing tokens in Spam vs Ham
│   ├── model_comparison.png   # Performance benchmark comparison chart
│   ├── confusion_matrix_comparison.png # Confusion matrices for all models
│   └── README.md              # Exported visual assets catalog
├── Spam_Mail_Detector.ipynb   # Fully executed Jupyter notebook with outputs
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

### Methodology

1. Preprocessing and Text Normalization:
   * Normalizes URLs, email addresses, phone numbers, and currency symbols.
   * Applies lowercase conversion, tokenization, and stopword removal.
2. Feature Extraction:
   * TfidfVectorizer with unigrams and bigrams, sublinear TF scaling, and maximum 3,000 features.
3. Multi-Model Benchmark and Cross-Validation:
   * Compared Multinomial Naive Bayes, Logistic Regression, Linear Support Vector Classifier, and Random Forest.
   * Evaluated using 5-Fold Stratified Cross-Validation to maintain class proportions.
4. Precision Priority Evaluation:
   * Prioritized high Spam Precision to avoid classifying legitimate messages as spam (zero false positives goal).
5. Model Serialization:
   * Exported trained pipeline using `joblib` for end-to-end inference on raw message strings.

### Benchmark Summary

| Model | Test Accuracy | Spam Precision | Spam Recall | Spam F1 | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Multinomial Naive Bayes | 0.9848 | 1.0000 | 0.8867 | 0.9399 | 0.9655 |
| Linear SVC | 0.9839 | 0.9660 | 0.9133 | 0.9384 | 0.9644 |
| Logistic Regression | 0.9686 | 0.9837 | 0.8067 | 0.8864 | 0.9343 |
| Random Forest | 0.9785 | 0.9922 | 0.8467 | 0.9137 | 0.9507 |

### Setup and Execution

To run the notebook locally:

```bash
cd Task_2_Spam_Mail_Detector
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Spam_Mail_Detector.ipynb
```
