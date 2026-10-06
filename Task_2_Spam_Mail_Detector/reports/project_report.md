# Spam Mail & SMS Detector: Technical Project Report

**Author:** AI & ML Intern  
**Program:** QSkill Internship  
**Domain:** Artificial Intelligence & Machine Learning  
**Task:** Task 2 (Spam Mail Detector)  
**Date:** October 2026  

### 1. Executive Summary
This project develops an advanced Natural Language Processing (NLP) classification pipeline to classify messages into **Ham** (legitimate) and **Spam**. Ingested live from the official UCI SMS Spam Collection repository in memory, the solution incorporates semantic entity normalization, bigram TF-IDF feature extraction, and a comparative study of **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Classifiers (LinearSVC)**.

The champion model (**Linear SVC**) achieved **99.01% test accuracy** and a **96.27% Spam F1-Score**, with **Multinomial Naive Bayes** achieving a remarkable **97.87% Spam Precision** (minimizing costly false positives). The entire pipeline was serialized for real-time inference.

### 2. Dataset Exploration & Statistical Analysis

#### 2.1 Provenance
* Source: Official UCI Machine Learning Repository (SMS Spam Collection, Dataset ID: 228) streamed live in memory
* Volume: 5,574 messages
* Class Imbalance:
  1. Ham: 4,827 messages (86.60%)
  2. Spam: 747 messages (13.40%)
* Quality: Zero null entries, natural duplicate patterns preserved for realistic training distributions

#### 2.2 Key Insights from Exploratory Analysis
1. Message Length Disparity:
   * Ham: Mean length of 71.02 characters (median: 52 characters, 14.2 words).
   * Spam: Mean length of 138.87 characters (median: 149 characters, 23.8 words). Spam messages systematically utilize almost the entire 160-character SMS payload limit.
2. Syntactic Markers:
   * Spam contains 15.4× more currency symbols (`$`, `£`, `€`) than legitimate messages.
   * Spam contains 4.6× more digit characters (phone numbers, account references, prize amounts).
   * Spam contains 5.2× more uppercase letters (urgent call-to-actions, e.g., `FREE`, `URGENT`, `CLAIM`).
3. Lexical Divergence:
   * Top spam tokens: `call`, `free`, `text`, `claim`, `mobile`, `prize`, `won`, `urgent`, `customer`.
   * Top ham tokens: `good`, `come`, `like`, `time`, `know`, `love`, `going`, `today`.

### 3. NLP Preprocessing Architecture

Standard text cleaners often delete all punctuation and numbers, destroying the strongest predictive signals in spam detection. Our custom preprocessor addresses this via **Entity Preservation**:
1. URL Normalization: URLs mapped to `httpaddr`.
2. Email Normalization: Emails mapped to `emailaddr`.
3. Currency Normalization: `$`, `£`, `€`, `¥` mapped to `currencysign`.
4. Number Normalization: Digit sequences mapped to `numbr`.
5. Morphological Normalization: Lowercasing, stopword elimination (NLTK English corpus), and `WordNetLemmatizer` lemmatization.

#### Feature Extraction:
* `TfidfVectorizer` configured with:
  1. N-gram range: $(1, 2)$ (capturing phrase structures like `claim prize`, `free entry`, `call now`).
  2. `min_df=2` (eliminating noise and rare typos).
  3. `max_df=0.95` (filtering ultra-frequent terms).
  4. Resulting vocabulary: **8,879 features**.

### 4. Model Benchmarking & Evaluation

#### 4.1 Cross-Validation (5-Fold Stratified on Training Set)
* Linear SVC: Macro F1: $0.9754 \pm 0.008$ | Accuracy: $98.92\% \pm 0.35\%$
* Multinomial Naive Bayes: Macro F1: $0.9632 \pm 0.010$ | Accuracy: $98.41\% \pm 0.42\%$
* Logistic Regression: Macro F1: $0.9579 \pm 0.011$ | Accuracy: $98.27\% \pm 0.48\%$

#### 4.2 Held-Out Test Set Performance (1,115 Unseen Messages)

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVC** | **99.01%** | 97.26% | **95.30%** | **96.27%** | **99.01%** |
| **Multinomial Naive Bayes** | 98.74% | **97.87%** | 92.62% | 95.17% | 98.73% |
| **Logistic Regression** | 98.48% | 92.86% | 95.97% | 94.39% | 98.49% |

#### 4.3 Detailed Confusion Matrix Breakdown (Test Set)
1. Linear SVC:
   * Correct Ham: 962 / 966 (False Positives: 4)
   * Correct Spam: 142 / 149 (False Negatives: 7)
2. Multinomial Naive Bayes:
   * Correct Ham: 963 / 966 (False Positives: 3)
   * Correct Spam: 138 / 149 (False Negatives: 11)

### 5. Error Analysis & Edge Cases

Examining the few misclassified messages reveals actionable insights:
1. False Positives (Ham flagged as Spam): Messages containing words like `free`, `credit`, or `gift` in legitimate contexts (e.g., *"I have extra credit for our project"*).
2. False Negatives (Spam flagged as Ham): Very short spam messages lacking overt commercial vocabulary (e.g., *"Call me at 078..."*).

### 6. Production Deployment & API

The full pipeline (`TfidfVectorizer` + `Classifier`) was serialized to `models/spam_pipeline.joblib`.

#### Live Inference Test:
```python
>>> classify_message("URGENT! You have won a £2,000 cash prize. Call 0800-888-999!")
{'Prediction': 'SPAM', 'Confidence': '99.82%'}

>>> classify_message("Hey Mom, I will be home around 6:30 for dinner tonight.")
{'Prediction': 'HAM', 'Confidence': '99.94%'}
```

### 7. Conclusion
Both required algorithms from the project syllabus (Naive Bayes and Logistic Regression), along with Linear SVC, were implemented and benchmarked. The model achieves industry-level spam precision (>97%), preventing false alarms on legitimate emails while neutralizing over 95% of spam intrusions.
