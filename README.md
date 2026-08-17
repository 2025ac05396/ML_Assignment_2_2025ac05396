# Machine Learning Assignment 2 – Human Activity Recognition

## 1. Problem Statement

The objective of this project is to perform Human Activity Recognition using machine learning classification algorithms.

The task is to classify human physical activities based on sensor measurements collected from a smartphone. Multiple classification models are trained and evaluated on the same dataset, and their performances are compared using different evaluation metrics.

The following activities are classified:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

---

## 2. Dataset Description

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The dataset contains smartphone sensor measurements collected from subjects performing different physical activities.

The dataset contains:

- **10,299 total observations**
- **7,352 training observations**
- **2,947 testing observations**
- **561 sensor features**
- **6 activity classes**

The 561 features consist of processed measurements obtained from smartphone accelerometer and gyroscope sensor signals.

### Dataset Split

| Dataset | Number of Samples | Number of Features |
|---|---:|---:|
| Training | 7,352 | 561 |
| Testing | 2,947 | 561 |

The test dataset used by the Streamlit application contains the 561 features along with the activity information.

---

## 3. GitHub Repository

**GitHub Repository:**

https://github.com/2025ac05396/ML_Assignment_2_2025ac05396

The repository contains the complete source code, trained machine learning models, test data, requirements file, notebook, and project documentation.

---

## 4. Models Used

The following machine learning classification models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest Classifier (Ensemble Model)

### Evaluation Metrics

Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 4.1 Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.9549** | **0.9975** | **0.9567** | **0.9549** | **0.9548** | **0.9461** |
| Decision Tree | 0.8622 | 0.9173 | 0.8633 | 0.8622 | 0.8617 | 0.8348 |
| KNN | 0.8802 | 0.9764 | 0.8883 | 0.8802 | 0.8790 | 0.8578 |
| Naive Bayes | 0.7703 | 0.9583 | 0.7947 | 0.7703 | 0.7688 | 0.7286 |
| Random Forest (Ensemble) | 0.9257 | 0.9953 | 0.9270 | 0.9257 | 0.9256 | 0.9109 |

---

## 4.2 Observations on Model Performance

### Logistic Regression

Logistic Regression achieved the highest overall performance among the implemented models. It achieved an accuracy of **95.49%** and an AUC of **99.75%**. Its precision, recall, F1 score, and MCC were also the highest overall, making it the best-performing model for this dataset.

### Decision Tree

The Decision Tree achieved an accuracy of **86.22%** and an AUC of **91.73%**. It provided reasonable classification performance but performed below Logistic Regression, Random Forest, and KNN.

### KNN

KNN achieved an accuracy of **88.02%** and an AUC of **97.64%**. It performed better than the Decision Tree and Naive Bayes models but did not outperform Logistic Regression or Random Forest.

### Naive Bayes

Naive Bayes achieved an accuracy of **77.03%**, which was the lowest accuracy among the implemented models. Its F1 score and MCC were also lower compared with the other models.

### Random Forest (Ensemble)

Random Forest achieved an accuracy of **92.57%** and an AUC of **99.53%**. It was the second-best-performing model overall and performed substantially better than the Decision Tree, KNN, and Naive Bayes models.

---

## 4.3 Overall Winner

### Logistic Regression

**Logistic Regression is the overall winner for the chosen dataset.**

Its performance was:

| Metric | Score |
|---|---:|
| Accuracy | **0.9549** |
| AUC | **0.9975** |
| Precision | **0.9567** |
| Recall | **0.9549** |
| F1 Score | **0.9548** |
| MCC | **0.9461** |

Logistic Regression achieved the highest accuracy, AUC, precision, recall, F1 score, and MCC among the five implemented models.

---

## 5. Streamlit Application

A Streamlit web application was developed to demonstrate the trained machine learning models.

### Live Application

The deployed Streamlit application can be accessed here:

Live Demo: https://mlassignment22025ac05396.streamlit.app/

The application provides the following features:

### Application Workflow

```text
Upload Test Dataset
        ↓
Select Machine Learning Model
        ↓
Validate 561 Features
        ↓
Apply Required Preprocessing
        ↓
Generate Predictions
        ↓
Display Predicted Activities
        ↓
Classification Report
        ↓
Confusion Matrix
