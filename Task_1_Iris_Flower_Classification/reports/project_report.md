# Iris Flower Classification: Technical Project Report

**Author:** AI & ML Intern  
**Program:** QSkill Internship  
**Domain:** Artificial Intelligence & Machine Learning  
**Task:** Task 1 (Iris Flower Classification)  
**Date:** October 2026  

### 1. Executive Summary
This report presents an end-to-end machine learning solution for classifying Iris flowers into three distinct species (*Setosa*, *Versicolor*, and *Virginica*) based on sepal and petal morphology. Utilizing live remote ingestion from the official UCI Machine Learning Repository via `ucimlrepo`, we performed extensive exploratory data analysis, built a leakage-free preprocessing pipeline, and systematically benchmarked multiple classification algorithms cited in the project specification (Logistic Regression, K-Nearest Neighbors, Decision Trees, and Random Forests). 

The models achieved up to 96.67% test accuracy with 5-fold cross-validated stability. The best-performing pipeline was fully serialized for inference.

### 2. Dataset Exploration & Preprocessing

#### 2.1 Data Provenance & Structure
* Source: Official UCI Machine Learning Repository (Dataset ID: 53) via `ucimlrepo`
* Instances: 150 rows, perfectly balanced with 50 samples per class
* Features: 4 numeric continuous predictors:
  1. `sepal_length` (Range: 4.3 to 7.9 cm, Mean: 5.84 cm)
  2. `sepal_width` (Range: 2.0 to 4.4 cm, Mean: 3.06 cm)
  3. `petal_length` (Range: 1.0 to 6.9 cm, Mean: 3.76 cm)
  4. `petal_width` (Range: 0.1 to 2.5 cm, Mean: 1.20 cm)
* Data Quality: Zero missing values, zero anomalous null records

#### 2.2 Key Findings from EDA
1. Linear Separability of Setosa: *Iris-setosa* exhibits significantly shorter petal lengths (1.0 to 1.9 cm) and petal widths (0.1 to 0.6 cm) compared to the other two species. It forms an isolated cluster that can be separated with 100% precision using simple thresholding.
2. Feature Collinearity: `petal_length` and `petal_width` display a Pearson correlation of $r = 0.963$, indicating high feature redundancy that linear models can leverage effectively.
3. Class Overlap: *Iris-versicolor* and *Iris-virginica* have overlapping distributions in sepal measurements, necessitating multi-dimensional decision boundaries.

### 3. Machine Learning Methodology

#### 3.1 Data Splitting & Standardization
* Partition: 80% training (120 samples) and 20% testing (30 samples)
* Stratification: Stratified splitting ensures exactly 10 test samples per class
* Standardization: Encapsulated inside `sklearn.pipeline.Pipeline` with `StandardScaler` fitted exclusively on training splits to prevent data leakage

#### 3.2 Candidate Algorithms Benchmarked
In accordance with the internship instructions:
1. Logistic Regression (Multinomial): L2 regularized with balanced multiclass solver
2. K-Nearest Neighbors (KNN): Non-parametric distance-based classifier with $k=5$
3. Decision Tree Classifier: Hierarchical binary splits with `max_depth=3` to prevent tree over-specialization
4. Random Forest Classifier: Ensemble of 100 bagged decision trees

### 4. Experimental Results & Evaluation

#### 4.1 Cross-Validation Benchmark (Training Set)
5-Fold Stratified Cross-Validation was conducted across all pipelines:
* Logistic Regression: $96.67\% \pm 3.12\%$
* K-Nearest Neighbors: $95.83\% \pm 4.56\%$
* Random Forest: $95.00\% \pm 3.73\%$
* Decision Tree: $94.17\% \pm 5.00\%$

#### 4.2 Held-Out Test Set Performance

| Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | **96.67%** | **96.97%** | **96.67%** | **96.66%** |
| **Logistic Regression** | 93.33% | 93.33% | 93.33% | 93.33% |
| **K-Nearest Neighbors** | 93.33% | 94.44% | 93.33% | 93.27% |
| **Random Forest** | 90.00% | 90.24% | 90.00% | 89.97% |

#### 4.3 Classification Diagnostics (Champion Model)
* Setosa: Precision 1.0000 | Recall 1.0000 | F1-Score 1.0000 (10/10 correct)
* Versicolor: Precision 1.0000 | Recall 0.9000 | F1-Score 0.9474 (9/10 correct)
* Virginica: Precision 0.9091 | Recall 1.0000 | F1-Score 0.9524 (10/10 correct)

Only 1 sample from Versicolor was misclassified as Virginica at the boundary.

### 5. Deployment & Inference
The end-to-end pipeline was saved to `models/iris_pipeline.joblib`. Test loading confirmed sub-millisecond prediction latency on unseen inputs:
* Input `[5.1, 3.5, 1.4, 0.2]` $\rightarrow$ **Setosa** (97.8% confidence)
* Input `[5.9, 3.0, 4.2, 1.5]` $\rightarrow$ **Versicolor** (84.1% confidence)
* Input `[6.8, 3.2, 5.9, 2.3]` $\rightarrow$ **Virginica** (98.4% confidence)

### 6. Conclusion
The objective outlined in the project instructions has been completed with comprehensive visual exploration, rigorous multi-model comparison, and an exported model pipeline.
