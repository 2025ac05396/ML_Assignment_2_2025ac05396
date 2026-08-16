import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Human Activity Recognition")
st.write(
    "Machine Learning based Human Activity Recognition "
    "using smartphone sensor data."
)

# ---------------------------------------------------------
# Load trained models
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "models/logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            "models/decision_tree.pkl"
        ),

        "KNN": joblib.load(
            "models/knn.pkl"
        ),

        "Naive Bayes": joblib.load(
            "models/naive_bayes.pkl"
        ),

        "Random Forest": joblib.load(
            "models/random_forest.pkl"
        )
    }

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    activity_mapping = joblib.load(
        "models/activity_mapping.pkl"
    )

    return models, scaler, activity_mapping


# ---------------------------------------------------------
# Load models safely
# ---------------------------------------------------------

try:

    models, scaler, activity_mapping = load_models()

except Exception as e:

    st.error(
        "Unable to load the trained models. "
        "Please make sure all model files are present "
        "inside the models folder."
    )

    st.exception(e)

    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ---------------------------------------------------------
# File upload
# ---------------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------

model_metrics = {
    "Logistic Regression": {
        "Accuracy": 0.954869,
        "AUC": 0.997485,
        "Precision": 0.956650,
        "Recall": 0.954869,
        "F1": 0.954809,
        "MCC": 0.946140
    },

    "Decision Tree": {
        "Accuracy": 0.8622,
        "AUC": 0.9173,
        "Precision": 0.8633,
        "Recall": 0.8622,
        "F1": 0.8617,
        "MCC": 0.8348
    },

    "KNN": {
        "Accuracy": 0.8802,
        "AUC": 0.9764,
        "Precision": 0.8883,
        "Recall": 0.8802,
        "F1": 0.8790,
        "MCC": 0.8578
    },

    "Naive Bayes": {
        "Accuracy": 0.7703,
        "AUC": 0.9583,
        "Precision": 0.7947,
        "Recall": 0.7703,
        "F1": 0.7688,
        "MCC": 0.7286
    },

    "Random Forest": {
        "Accuracy": 0.9257,
        "AUC": 0.9953,
        "Precision": 0.9270,
        "Recall": 0.9257,
        "F1": 0.9256,
        "MCC": 0.9109
    }
}


# ---------------------------------------------------------
# Display evaluation metrics
# ---------------------------------------------------------

st.header("2. Model Evaluation")

metrics = model_metrics[selected_model_name]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.4f}"
    )

with col2:
    st.metric(
        "AUC",
        f"{metrics['AUC']:.4f}"
    )

with col3:
    st.metric(
        "Precision",
        f"{metrics['Precision']:.4f}"
    )

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Recall",
        f"{metrics['Recall']:.4f}"
    )

with col5:
    st.metric(
        "F1 Score",
        f"{metrics['F1']:.4f}"
    )

with col6:
    st.metric(
        "MCC",
        f"{metrics['MCC']:.4f}"
    )


# ---------------------------------------------------------
# Process uploaded data
# ---------------------------------------------------------

if uploaded_file is not None:

    st.header("3. Test Data")

    try:

        data = pd.read_csv(uploaded_file)

        st.write(
            f"Dataset shape: **{data.shape[0]} rows × "
            f"{data.shape[1]} columns**"
        )

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # -------------------------------------------------
        # Separate features and target
        # -------------------------------------------------

        target_columns = [
            "Activity",
            "Activity_Name"
        ]

        feature_columns = [
            column
            for column in data.columns
            if column not in target_columns
        ]

        X = data[feature_columns]

        y = None

        if "Activity" in data.columns:

            y = data["Activity"]

        # -------------------------------------------------
        # Make sure features are numeric
        # -------------------------------------------------

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.fillna(0)

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        st.header("4. Predictions")

        predictions = selected_model.predict(X)

        predicted_names = [
            activity_mapping.get(
                int(prediction),
                str(prediction)
            )
            for prediction in predictions
        ]

        prediction_output = pd.DataFrame({
            "Predicted Activity": predicted_names
        })

        st.dataframe(
            prediction_output.head(20),
            use_container_width=True
        )

        # -------------------------------------------------
        # Classification report and confusion matrix
        # -------------------------------------------------

        if y is not None:

            st.header("5. Classification Report")

            report = classification_report(
                y,
                predictions,
                target_names=[
                    activity_mapping[i]
                    for i in sorted(activity_mapping.keys())
                ],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df,
                use_container_width=True
            )

            st.header("6. Confusion Matrix")

            cm = confusion_matrix(
                y,
                predictions
            )

            labels = [
                activity_mapping[i]
                for i in sorted(activity_mapping.keys())
            ]

            fig, ax = plt.subplots(
                figsize=(9, 7)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )

            ax.set_xlabel(
                "Predicted Activity"
            )

            ax.set_ylabel(
                "Actual Activity"
            )

            ax.set_title(
                f"Confusion Matrix - {selected_model_name}"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.yticks(
                rotation=0
            )

            plt.tight_layout()

            st.pyplot(fig)

        else:

            st.info(
                "The uploaded CSV does not contain an "
                "'Activity' column. Predictions are shown "
                "without evaluation metrics."
            )

    except Exception as e:

        st.error(
            "An error occurred while processing the "
            "uploaded dataset."
        )

        st.exception(e)

else:

    st.info(
        "Upload test_data.csv above to generate "
        "predictions and the classification report."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Human Activity Recognition | "
    "Machine Learning Assignment 2"
)
