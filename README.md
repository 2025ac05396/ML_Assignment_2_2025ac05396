# Machine Learning Assignment 2 – Human Activity Recognition

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for Human Activity Recognition (HAR) using smartphone sensor data.

The models are trained to classify six different human activities:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

Five machine learning classification algorithms are implemented and evaluated using the UCI Human Activity Recognition Using Smartphones Dataset.

---

## 2. Dataset Description

The dataset used in this project is the **UCI Human Activity Recognition Using Smartphones Dataset**.

The dataset contains sensor measurements collected from smartphones worn by subjects while performing different physical activities.

The dataset contains:

- **561 features**
- **6 activity classes**
- **7352 training samples**
- **2947 testing samples**
- **10299 total samples**

The six activity classes are:

| Activity ID | Activity |
|---:|---|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

The dataset is divided into training and testing sets.

### Dataset files

The project uses the following dataset components:

- `X_train.txt` – Training feature data
- `y_train.txt` – Training activity labels
- `X_test.txt` – Testing feature data
- `y_test.txt` – Testing activity labels
- `features.txt` – Feature names
- `activity_labels.txt` – Activity names

---

## 3. Machine Learning Models Used

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest

---

## 4. Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.9549** | **0.9975** | **0.9567** | **0.9549** | **0.9548** | **0.9461** |
| Decision Tree | 0.8622 | 0.9173 | 0.8633 | 0.8622 | 0.8617 | 0.8348 |
| KNN | 0.8802 | 0.9764 | 0.8883 | 0.8802 | 0.8790 | 0.8578 |
| Naive Bayes | 0.7703 | 0.9583 | 0.7947 | 0.7703 | 0.7688 | 0.7286 |
| Random Forest | 0.9257 | 0.9953 | 0.9270 | 0.9257 | 0.9256 | 0.9109 |

---

## 6. Model Observations

### Logistic Regression

Logistic Regression achieved the best overall performance among the five evaluated models. It obtained an accuracy of **95.49%** and an AUC of **99.75%**. Its Precision, Recall and F1 Score were also above 95%, while the MCC was **0.9461**.

Therefore, Logistic Regression provided the strongest overall classification performance for this dataset.

### Decision Tree

The Decision Tree achieved an accuracy of **86.22%**, with an AUC of **91.73%**. Its F1 Score was **86.17%** and MCC was **0.8348**.

The model provides reasonable classification performance but performs considerably below Logistic Regression and Random Forest.

### KNN

KNN achieved an accuracy of **88.02%** and an AUC of **97.64%**. Its Precision was **88.83%**, Recall was **88.02%**, F1 Score was **87.90%**, and MCC was **0.8578**.

KNN provides good performance but does not outperform Logistic Regression or Random Forest.

### Naive Bayes

Naive Bayes achieved an accuracy of **77.03%**, which was the lowest accuracy among the five models. Its AUC was **95.83%**, while its F1 Score was **76.88%** and MCC was **0.7286**.

Although the model achieved a relatively high AUC, its overall classification performance was lower than the other models.

### Random Forest

Random Forest achieved the second-highest overall performance, with an accuracy of **92.57%** and an AUC of **99.53%**. Its Precision was **92.70%**, Recall was **92.57%**, F1 Score was **92.56%**, and MCC was **0.9109**.

Random Forest performed strongly but remained below Logistic Regression across the reported evaluation metrics.

---

## 7. Overall Winner

### Logistic Regression

**Logistic Regression is the overall winner.**

It achieved the highest values across all six reported evaluation metrics:

| Metric | Logistic Regression |
|---|---:|
| Accuracy | **95.49%** |
| AUC | **99.75%** |
| Precision | **95.67%** |
| Recall | **95.49%** |
| F1 Score | **95.48%** |
| MCC | **0.9461** |

Therefore, Logistic Regression was selected as the best-performing model for the Human Activity Recognition classification task.

---

## 8. Project Files

```text
ML_Assignment_2/
│
├── ML_Assignment_2_UCI_HAR.ipynb
├── model_comparison.csv
├── test_data.csv
├── requirements.txt
├── app.py
├── README.md
│
└── models/
    ├── activity_mapping.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
