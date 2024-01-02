import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import base64
# Load the saved scikit-learn KNeighborsClassifier model
with open('knnclassifier_model.pkl', 'rb') as file:
    loaded_knnclassifier = pickle.load(file)

# Function to predict fraud using the loaded model
def predict_fraud(features):
    # Make predictions
    prediction = loaded_knnclassifier.predict(features)
    return prediction

# Streamlit app
def main():
    st.title('Credit Fraud Detection')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Sidebar with navigation options
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model Prediction", "Data Visualization"])

    if page == "Model Prediction":
        model_prediction_page()
    elif page == "Data Visualization":
        data_visualization_page()

import pandas as pd
import streamlit as st

def model_prediction_page():
    # File upload for CSV containing feature values
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    predict_button = st.button("Predict")
    if uploaded_file is not None :
        num_rows = st.number_input("Enter the number of rows to predict", min_value=1, step=1)
    if uploaded_file is not None and predict_button:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        
        # Check column names in the uploaded file
        
        # Ensure column names match the expected features
        expected_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        data.columns = expected_features + ['Class']  # Assuming 'Class' is the prediction column
        
        with st.spinner('Predicting...'):
            # Make predictions using the loaded scikit-learn model for selected rows
            X = data.drop('Class', axis=1).head(num_rows)  # Features of selected rows
            predictions = predict_fraud(X)
            
            # Create a DataFrame to store row numbers and predictions
            fraud_predictions = pd.DataFrame({
                'row': range(1, num_rows + 1),
                'prediction': ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions]
            })

        

        # Download predictions as CSV using st.download_button
            st.download_button(
              label="Download Predictions as CSV",
             data=fraud_predictions.to_csv(index=False),
             file_name='fraud_predictions.csv',
             mime='text/csv',
             help='Click here to download the fraud predictions'
                              )
def data_visualization_page():
    # File upload for CSV containing feature values
    uploaded_file = st.file_uploader("Upload CSV file for Data Visualization", type="csv", key="file_upload_visualization")
    
    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Allow user to select columns for visualization
        st.sidebar.header("Column Selection")
        selected_columns = st.sidebar.multiselect("Select numeric columns for visualization", data.select_dtypes(include='number').columns)

        if not selected_columns:
            st.warning("Please select at least one numeric column for visualization.")
            return

        # Plot pair plot
        st.subheader("Pair Plot")
        plt.figure(figsize=(12, 10))
        sns.pairplot(data[selected_columns])
        st.pyplot()

        # Plot box plot
        st.subheader("Box Plot")
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=data[selected_columns])
        st.pyplot()

        # Plot scatterplot matrix
        st.subheader("Scatterplot Matrix")
        plt.figure(figsize=(12, 10))
        sns.pairplot(data[selected_columns])
        st.pyplot()

        # Plot correlation heatmap
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 10))
        sns.heatmap(data[selected_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
        st.pyplot()

if __name__ == '__main__':
    main()
