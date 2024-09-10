import streamlit as st
import pandas as pd
import plotly.express as px
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, pull as cls_pull, save_model as cls_save_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model
import os

# Function to analyze data
def analyze_data(df):
    st.header("Data Analysis")
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Display data types
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    # Display missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    # Display correlation matrix
    st.subheader("Correlation Matrix")
    st.write(df.corr())
    
    # Plot histograms for each column
    st.subheader("Histograms")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20:
            fig = px.line(df, y=col, title=f'Line Plot of {col}')
        else:
            fig = px.histogram(df, x=col, title=f'Bar Plot of {col}')
        st.plotly_chart(fig)
    
    # Plot scatter matrix
    st.subheader("Scatter Matrix")
    fig = px.scatter_matrix(df)
    st.plotly_chart(fig)

# Function to train and evaluate models
def train_and_evaluate(df, target_col, task_type):
    # Sample a subset of the data if it's too large
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)
        st.warning("Data too large, using a random sample of 2000 rows for training.")
    
    if task_type == "classification":
        cls_setup(df, target=target_col, silent=True)
        best_model = cls_compare_models(n_select=10)
        results = cls_pull()
        cls_save_model(best_model, 'best_classification_model')
    else:
        reg_setup(df, target=target_col, silent=True)
        best_model = reg_compare_models(n_select=10)
        results = reg_pull()
        reg_save_model(best_model, 'best_regression_model')
    
    st.header("Model Evaluation")
    st.write(results)
    
    # Save the best model
    if task_type == "Classification":
        st.success("Best classification model saved as 'best_classification_model.pkl'")
    else:
        st.success("Best regression model saved as 'best_regression_model.pkl'")

    return best_model

# Function to provide insights
def provide_insights(df, task_type):
    st.header("Data Insights")
    st.write("Here are some insights about your data and the models used:")
    
    # Data insights
    st.subheader("Data Insights")
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write("The basic statistics and data types have been displayed above.")
    
    # Model insights
    st.subheader("Model Insights")
    if task_type == "classification":
        st.write("For classification, we have used 10 different models to evaluate the performance.")
        st.write("The models have been evaluated using metrics such as accuracy, precision, recall, and F1-score.")
        st.write("The best model has been saved as 'best_classification_model.pkl'.")
    else:
        st.write("For regression, we have used 10 different models to evaluate the performance.")
        st.write("The models have been evaluated using metrics such as MSE (Mean Squared Error) and MAE (Mean Absolute Error).")
        st.write("The best model has been saved as 'best_regression_model.pkl'.")

# Streamlit UI
st.title("AutoML Web Application")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully!")
    st.write(df.head())
    
    # Analyze data
    analyze_data(df)
    
    # Select target column
    target_col = st.selectbox("Select the target column", df.columns)
    
    # Select task type
    task_type = st.selectbox("Select the task type", ["Classification", "Regression"])
    
    # Train and evaluate models
    if st.button("Train and Evaluate Models"):
        best_model = train_and_evaluate(df, target_col, task_type)
        provide_insights(df, task_type)
        
        # Add download button for the best model
        if task_type == "classification":
            with open('best_classification_model.pkl', 'rb') as f:
                st.download_button('Download Best Model', f, file_name="best_classification_model.pkl")
        else:
            with open('best_regression_model.pkl', 'rb') as f:
                st.download_button('Download Best Model', f, file_name="best_regression_model.pkl")
