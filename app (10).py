import streamlit as st
import pandas as pd
import pickle

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

st.title("Customer Churn Prediction App")

st.write("This app predicts whether a customer will churn based on their characteristics.")

# Add input fields for some features (add more for all your features)
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1]) # Assuming 0 and 1 are the values
partner = st.sidebar.selectbox("Partner", ['Yes', 'No'])
dependents = st.sidebar.selectbox("Dependents", ['Yes', 'No'])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 1)
phone_service = st.sidebar.selectbox("Phone Service", ['Yes', 'No'])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.sidebar.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
online_backup = st.sidebar.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
device_protection = st.sidebar.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
tech_support = st.sidebar.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
streaming_tv = st.sidebar.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.sidebar.number_input("Monthly Charges", value=20.0)
total_charges = st.sidebar.number_input("Total Charges", value=20.0)


# Create a pandas DataFrame from the inputs
input_data = pd.DataFrame([[gender, senior_citizen, partner, dependents, tenure, phone_service,
                            multiple_lines, internet_service, online_security, online_backup,
                            device_protection, tech_support, streaming_tv, streaming_movies,
                            contract, paperless_billing, payment_method, monthly_charges, total_charges]],
                          columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                   'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])


# Preprocess the input data (one-hot encode and scale)
# Note: You'll need to apply the same one-hot encoding logic as in your notebook
# and then use the loaded_scaler to scale the numerical features.

# Apply one-hot encoding to the input data - make sure columns match training data
input_data_encoded = pd.get_dummies(input_data, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                                       'PaperlessBilling', 'PaymentMethod'], drop_first=True)

# Ensure all columns from training data are present, fill missing with 0
# This is crucial because get_dummies might not create columns for categories not present in the single input row
training_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'] # Replace with actual training columns
for col in training_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Ensure the order of columns is the same as in the training data
input_data_encoded = input_data_encoded[training_columns]


# Scale the numerical features
numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'] # Identify your numerical columns
input_data_encoded[numerical_cols] = loaded_scaler.transform(input_data_encoded[numerical_cols])


# Make prediction
if st.sidebar.button("Predict Churn"):
    prediction = loaded_model.predict(input_data_encoded)
    if prediction[0] == 'Yes':
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is unlikely to churn.")

st.write("Fill in the customer information in the sidebar and click 'Predict Churn'.")
