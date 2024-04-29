import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
filename = 'BestModel_XGB_rev.pkl'
model = pickle.load(open(filename, 'rb'))

# Load the encoder objects
filename_gender_encode = 'gender_encode.pkl'
gender_encode = pickle.load(open(filename_gender_encode, 'rb'))

filename_geo_encode = 'oneHot_encode_geo.pkl'
geo_encoder = pickle.load(open(filename_geo_encode, 'rb'))

# Robust and MinMax scalers
robust_scaler = pickle.load(open('robust_scaler.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmax_scaler.pkl', 'rb'))

def main():
    st.title('Churn Prediction App')

    # Get user input
    credit_score = st.number_input("Credit Score", 0, 1000)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 0, 100)
    tenure = st.number_input("Tenure", 0, 100)
    balance = st.number_input("Balance", 0, 100000)
    num_of_products = st.number_input("Number of Products", 0, 10)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", 0, 1000000)

    if st.button('Predict'):
        # Preprocess the input features
        features = preprocess_features(credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary)

        # Make the prediction
        prediction = model.predict(features.values.reshape(1, -1))
        if prediction[0] == 0:
            result = 'Not Churn'
        else:
            result = 'Churn'

        st.success(f'The prediction is: {result}')

def preprocess_features(credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary):
    # Encode categorical features
    gender_encoded = gender_encode[gender]
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_member_encoded = 1 if is_active_member == "Yes" else 0
    geo_encoded = geo_encoder.transform([[geography]]).toarray()[0]

    # Robust scaling for Age and Credit Score
    age_scaled = robust_scaler.transform([[age]])[0][0]
    credit_score_scaled = robust_scaler.transform([[credit_score]])[0][0]

    # MinMax scaling for Balance and Estimated Salary
    balance_scaled = minmax_scaler.transform([[balance]])[0][0]
    estimated_salary_scaled = minmax_scaler.transform([[estimated_salary]])[0][0]

    # Create a DataFrame with the processed features
    features = pd.DataFrame({
        'CreditScore': [credit_score_scaled],
        'Gender': [gender_encoded],
        'Age': [age_scaled],
        'Tenure': [tenure],
        'Balance': [balance_scaled],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_encoded],
        'IsActiveMember': [is_active_member_encoded],
        'EstimatedSalary': [estimated_salary_scaled]
    })

    # Concatenate the one-hot encoded Geography features
    features = pd.concat([features, pd.DataFrame({f'Geography_{col}': geo_encoded[i:i+1]
                                                  for i, col in enumerate(geo_encoder.get_feature_names_out())},
                                                 index=[0])], axis=1)

    return features

if __name__ == '__main__':
    main()
