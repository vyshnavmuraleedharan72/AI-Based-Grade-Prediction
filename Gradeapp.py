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
all_features = ['age', 'sex', 'address', 'famsize', 'Medu', 'Fedu', 
                'traveltime', 'studytime', 'failures', 'schoolsup', 
                'famsup', 'paid', 'activities', 'higher', 'internet', 
                'absences', 'G1', 'G2']

# Features to collect from the user
user_input_features = ['G2', 'age', 'activities', 'absences', 'failures']

default_values = {
    'sex': 'F', 'address': 'U', 'famsize': 'GT3', 'Medu': 2, 'Fedu': 2, 
    'traveltime': 1, 'studytime': 2, 'schoolsup': 'no', 'famsup': 'yes',
    'paid': 'no', 'higher': 'yes', 'internet': 'yes', 'G1': 10
}

st.title("AI-Based Grade Prediction")
st.write("Enter the required details to predict the student's final grade (G3).")

# Collect user inputs for specific features
input_data = {}
input_data['G2'] = st.number_input("Grade 2 (G2):", min_value=0, step=1)
input_data['age'] = st.number_input("Age:", min_value=0, step=1)
input_data['activities'] = st.selectbox("Activities (yes/no):", ['yes', 'no'])
input_data['absences'] = st.number_input("Number of Absences:", min_value=0, step=1)
input_data['failures'] = st.number_input("Number of Failures:", min_value=0, step=1)

# Add default values for other features
for feature, value in default_values.items():
    input_data[feature] = value

# Convert input data into a DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical features using the label encoders
for feature in label_encoders:
    if feature in input_df.columns:
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

