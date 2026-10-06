# Trained Models

This directory contains serialized scikit-learn models and preprocessing pipelines saved via `joblib`.

- `iris_pipeline.joblib`: Production `Pipeline` containing `StandardScaler` and the champion classifier.
- `iris_best_model.joblib`: The standalone fitted champion estimator.
- `iris_scaler.joblib`: Fitted `StandardScaler` for independent feature transformation.
