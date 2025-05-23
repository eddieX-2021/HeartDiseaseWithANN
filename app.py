import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import pickle

# Load trained model
import os
if not os.path.exists('heart_model.h5'):
    st.error("Model file not found!")
else:
    st.success(f"Model file found ({os.path.getsize('heart_model.h5')} bytes)")

model = tf.keras.models.load_model('heart_model.h5', safe_mode=False)

# Load scaler and preprocessor
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

st.title("Heart Disease Prediction App")

# User input
bmi = st.number_input("Enter your BMI", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
smoking = st.selectbox("Do you smoke?", ("Yes", "No"))
alcohol_drinking = st.selectbox("Do you drink alcohol?", ("Yes", "No"))
stroke = st.selectbox("Have you had a stroke?", ("Yes", "No"))
physical_health = st.number_input("How many days have you had poor physical health in the last 30 days?", 
                                 min_value=0, max_value=30, value=0)
mental_health = st.number_input("How many days have you had poor mental health in the last 30 days?", 
                               min_value=0, max_value=30, value=0)
diff_walking = st.selectbox("Do you have difficulty walking or climbing stairs?", ("Yes", "No"))
sex = st.selectbox("What is your Gender?", ("Male", "Female"))
age_category = st.selectbox("What is your age category?", 
                           ("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                            "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"))
race = st.selectbox("What is your Race?", ("White", "Black", "Asian", "American Indian/Alaskan Native", 
                                          "Hispanic", "Other"))
diabetic = st.selectbox("Do you have diabetes?", ("No", "Yes", "No, borderline diabetes", 
                                                 "Yes (during pregnancy)"))
physical_activity = st.selectbox("Do you do physical activity?", ("Yes", "No"))
gen_health = st.selectbox("How would you rate your general health?", 
                         ("Excellent", "Very good", "Good", "Fair", "Poor"))
sleep_time = st.number_input("How many hours do you sleep?", min_value=0, max_value=24, value=8)
asthma = st.selectbox("Do you have asthma?", ("Yes", "No"))
kidney_disease = st.selectbox("Do you have kidney disease?", ("Yes", "No"))
skin_cancer = st.selectbox("Do you have skin cancer?", ("Yes", "No"))

# Create input data dictionary
input_data = {
    'BMI': [bmi],
    'Smoking': [smoking],
    'AlcoholDrinking': [alcohol_drinking],
    'Stroke': [stroke],
    'PhysicalHealth': [physical_health],
    'MentalHealth': [mental_health],
    'DiffWalking': [diff_walking],
    'Sex': [sex],
    'AgeCategory': [age_category],
    'Race': [race],
    'Diabetic': [diabetic],
    'PhysicalActivity': [physical_activity],
    'GenHealth': [gen_health],
    'SleepTime': [sleep_time],
    'Asthma': [asthma],
    'KidneyDisease': [kidney_disease],
    'SkinCancer': [skin_cancer]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Preprocess binary columns
binary_cols = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", 
               "Asthma", "KidneyDisease", "SkinCancer", "PhysicalActivity"]
input_df[binary_cols] = input_df[binary_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))

# Make prediction when button is clicked
if st.button("Predict Heart Disease Risk"):
    try:
        # Transform using preprocessor
        encoded_input = preprocessor.transform(input_df)
        
        # Scale the features
        scaled_input = scaler.transform(encoded_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        probability = prediction[0][0]
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"Probability of having heart disease: {probability:.2%}")
        
        if probability > 0.5:
            st.error("High risk of heart disease detected")
        else:
            st.success("Low risk of heart disease detected")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")