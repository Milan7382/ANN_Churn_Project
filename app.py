import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("App started")

st.write("Starting Customer Churn Prediction App")  # Debug: Check if this displays

try:
    st.title('Customer Churn Prediction')
    logging.info("Title set")

    # Load the trained model and encoders
    try:
        model = tf.keras.models.load_model('model.h5')
        st.write("Model loaded successfully")  # Debug: Confirm model loads
        logging.info("Model loaded")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logging.error("Model loading error", exc_info=True)
        raise

    try:
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            label_encoder_geo = pickle.load(file)
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        st.write("Encoders and scaler loaded successfully")  # Debug: Confirm encoders load
        logging.info("Encoders and scaler loaded")
    except Exception as e:
        st.error(f"Error loading encoders/scaler: {str(e)}")
        logging.error("Encoders/scaler loading error", exc_info=True)
        raise

    st.write("Ready to collect user input")  # Debug: Check if input widgets are reached

    # User input
    geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])
    logging.info("User input collected")

    st.write("Preparing input data")  # Debug: Check if data preparation starts

    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    try:
        geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        st.write("Input data prepared:", input_data)  # Debug: Show input data
        logging.info("Input data prepared")
    except Exception as e:
        st.error(f"Error in one-hot encoding: {str(e)}")
        logging.error("One-hot encoding error", exc_info=True)
        raise

    # Scale the input data
    try:
        input_data_scaled = scaler.transform(input_data)
        st.write("Scaled input data shape:", input_data_scaled.shape)  # Debug: Show scaled data shape
        logging.info("Input data scaled")
    except Exception as e:
        st.error(f"Error in scaling: {str(e)}")
        logging.error("Scaling error", exc_info=True)
        raise

    # Predict churn
    try:
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        st.write(f'Churn Probability: {prediction_proba:.2f}')  # Debug: Show prediction
        if prediction_proba > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')
        logging.info("Prediction completed")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        logging.error("Prediction error", exc_info=True)
        raise

except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    logging.error("Unexpected error", exc_info=True)