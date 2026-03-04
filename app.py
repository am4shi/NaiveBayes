import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Naive Bayes Classifier", layout="wide")


st.title("📊 Naive Bayes Classification App")
st.markdown("---")

# Sidebar for data selection
st.sidebar.header("Data Settings")
data_option = st.sidebar.radio("Choose Data Source", ["Upload CSV", "Use Credit.csv"])

df = None

if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    try:
        # Note: Ensure Credit.csv is in your root folder when deploying to Render
        df = pd.read_csv("Credit.csv")
    except FileNotFoundError:
        st.error("⚠️ 'Credit.csv' not found. Please upload a file manually.")

if df is not None:
    # Basic Cleaning: Remove common ID columns and Unnamed columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Column Selection
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox("🎯 Select Target Column", df.columns)

    if target_column:
        y_check = df[target_column]

        # Validation for Naive Bayes (Classification only)
        if y_check.dtype == "object" or y_check.nunique() <= 15:
            st.success("Detected: Classification Problem")
        else:
            st.warning("⚠️ Target appears continuous. Naive Bayes requires categorical data.")
            if not st.checkbox("Proceed anyway?"):
                st.stop()

        with col2:
            feature_columns = st.multiselect(
                "🛠️ Select Feature Columns",
                [col for col in df.columns if col != target_column],
                default=[col for col in df.columns if col != target_column]
            )

        # Model Parameters
        st.markdown("---")
        st.subheader("⚙️ Model Parameters")
        p1, p2 = st.columns(2)
        test_size = p1.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = p2.number_input("Random State", value=42, step=1)

        if st.button("🚀 Train & Evaluate Model"):
            try:
                # Preprocessing
                X = df[feature_columns].copy()
                y = df[target_column].copy()

                # Handle Missing Values (Mode Imputation)
                X = X.fillna(X.mode().iloc[0])

                # Encoding Categorical Features
                for col in X.columns:
                    if X[col].dtype == "object":
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))

                # Encoding Target
                if y.dtype == "object":
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y.astype(str))

                # Split Data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=int(random_state)
                )

                # Initialize and Train
                model = GaussianNB()
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Output Results
                st.markdown("---")
                st.subheader("📈 Performance Metrics")
                st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.write("**Confusion Matrix Visualization**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                with res_col2:
                    st.write("**Classification Report**")
                    st.text(classification_report(y_test, y_pred))

            except Exception as e:
                st.error(f"❌ An error occurred during training: {e}")