import streamlit as st
import pandas as pd
import pickle

# Load the model, scaler, and encoders
with open("decision_tree_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoders.pkl", "rb") as encoder_file:
    label_encoders = pickle.load(encoder_file)

# Define the feature names
features = ['age', 'sex', 'address', 'famsize', 'Medu', 'Fedu', 
            'traveltime', 'studytime', 'failures', 'schoolsup', 
            'famsup', 'paid', 'activities', 'higher', 'internet', 
            'absences', 'G1', 'G2']

st.title("AI-Based Grade Prediction")
st.write("Enter the student's details to predict their final grade (G3).")

# Collect user input for all features
input_data = {}
for feature in features:
    if feature in label_encoders:  # For categorical features
        options = label_encoders[feature].classes_
        input_data[feature] = st.selectbox(f"{feature.capitalize()}:", options)
    else:  # For numerical features
        input_data[feature] = st.number_input(f"{feature.capitalize()}:", min_value=0, step=1)

# Convert input data into a DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical features
for feature in label_encoders:
    input_df[feature] = label_encoders[feature].transform(input_df[feature])

# Ensure all required features are present in input_df
for col in scaler.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing features with default value of 0

# Align the columns to match scaler input
input_df = input_df[scaler.feature_names_in_]

# Scale numeric inputs
input_df[scaler.feature_names_in_] = scaler.transform(input_df[scaler.feature_names_in_])

# Predict the grade
if st.button("Predict Grade"):
    prediction = model.predict(input_df)
    st.success(f"The predicted final grade (G3) is: {prediction[0]:.2f}")

