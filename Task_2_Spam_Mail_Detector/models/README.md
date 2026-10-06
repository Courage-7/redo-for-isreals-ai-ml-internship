# Trained Spam Classification Models

This directory stores serialized NLP transformers and machine learning models saved using `joblib`.

- `spam_pipeline.joblib`: Production `Pipeline([('tfidf', vectorizer), ('classifier', model)])` capable of raw string ingestion and immediate prediction.
- `tfidf_vectorizer.joblib`: Fitted `TfidfVectorizer` containing vocabulary and IDF weights.
- `spam_best_model.joblib`: Fitted champion classifier.
