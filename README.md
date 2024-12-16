# AI-Based-Grade-Prediction


# Project Overview

The AI-Based Grade Prediction project aims to predict a student’s final grade (G3) based on features such as:

* Age

* Study Time

* First Period Grade (G1)

* Second Period Grade (G2)

* Attendance

The project involves training a machine learning model (Decision Tree or Artificial Neural Network) to make accurate predictions and deploying it via a user-friendly web application built with Streamlit.

# Features of the Project

## Model Training:

A Decision Tree model is trained to predict final grades.

The model is evaluated using appropriate metrics (RMSE for regression).

## Feature Importance Analysis:

Insights are generated to identify key contributors to grade prediction.

# Web Application:

Users can input student details to get the predicted grade.

## Data Preprocessing:

Data cleaning and preparation steps include handling missing values, scaling, and feature selection.

# Files in the Project

## 1. Data

student_data.csv: The dataset used for training and testing the model. It includes features such as G1, G2, age, study time, and attendance.

## 2. Model and Scaler

decision_tree_model.pkl: Saved Decision Tree model for predicting grades.

scaler.pkl: Scaler used for preprocessing the input data.

## 3. Code Files

data_cleaning.ipynb: Jupyter Notebook for data cleaning and preprocessing.

model_training.ipynb: Jupyter Notebook for model training and evaluation.

evaluation.ipynb: Jupyter Notebook for evaluating the model and analyzing feature importance.

app.py: Streamlit application code for the web interface.

## 4. Requirements

requirements.txt: List of Python libraries required to run the project.

## 5. Documentation

README.md: Documentation for the project (this file).


