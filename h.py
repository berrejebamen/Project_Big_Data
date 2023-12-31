import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

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

def model_prediction_page():
    # File upload for CSV containing feature values
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        
        # Check column names in the uploaded file
        expected_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Ensure column names match the expected features
        data.columns = expected_features[:len(data.columns)]  # Trim or pad columns as needed

        with st.spinner('Predicting...'):
            # Select the first row for prediction
            first_row = data.iloc[[0]]
            
            # Make predictions using the loaded scikit-learn model for the first row
            X_first_row = first_row.drop('Amount', axis=1)  # Features of the first row, adjust as needed
            prediction_first_row = predict_fraud(X_first_row)
            
            # Display the prediction for the first row
            st.success("Prediction for the First Row:")
            if prediction_first_row[0] == 0:
                st.markdown("<div style='padding:10px;border-radius:5px;background-color:#b3ffb3;'>Not Fraud</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='padding:10px;border-radius:5px;background-color:#ffb3b3;'>Fraud</div>", unsafe_allow_html=True)

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
