from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
app.logger.debug('Starting Flask app')

# Load the trained model and scaler
model = joblib.load("depression_model.pkl") 
scaler = joblib.load("scaler.pkl")
app.logger.debug('Model and scaler loaded successfully')

# Define feature columns (must match the model's training features)
feature_columns = [
    'Age', 'CGPA', 'Suicidal thoughts', 'Work/Study Hours', 'Family History of Mental Illness',
    'Sleep Duration_Less than 5 hours', 'Dietary Habits_Unhealthy', 'Dietary Habits_Healthy',
    'degree_group_Bachelor', 'degree_group_Master', 'degree_group_School',
    'Financial_Stress_Category_Low', 'Financial_Stress_Category_High', 'Financial_Stress_Category_Very High',
    'region_South', 'region_East', 'region_North', 'region_West',
    'Study_Satisfaction_Category_Neutral', 'Study_Satisfaction_Category_Very Dissatisfied',
    'Study_Satisfaction_Category_Very Satisfied', 'Academic_Pressure_Category_High Pressure',
    'Academic_Pressure_Category_Very High Pressure', 'Academic_Pressure_Category_Low Pressure'
]

# Define numerical columns to scale
columns_to_scale = ['Age', 'CGPA', 'Work/Study Hours']

# Preprocessing function
def preprocess_input(data, scaler, feature_columns):
    app.logger.debug(f'Preprocessing input: {data}')
    df_input = pd.DataFrame([data])
    # Convert numerical inputs to float
    for col in columns_to_scale:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
    # Scale numerical features
    df_input[columns_to_scale] = scaler.transform(df_input[columns_to_scale])
    # Encode categorical variables
    df_encoded = pd.get_dummies(df_input, columns=['Sleep Duration', 'Dietary Habits', 'region', 'degree_group',
                                                  'Financial_Stress_Category', 'Study_Satisfaction_Category',
                                                  'Academic_Pressure_Category'], dtype=int)
    # Ensure all model features are present
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[feature_columns]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        # Collect form data
        data = {
            'Age': request.form['Age'],
            'CGPA': request.form['CGPA'],
            'Work/Study Hours': request.form['Work_Study_Hours'],
            'Suicidal thoughts': 1 if request.form['Suicidal_thoughts'] == 'Yes' else 0,
            'Family History of Mental Illness': 1 if request.form['Family_History'] == 'Yes' else 0,
            'Gender_Male': 1 if request.form['Gender'] == 'Male' else 0,
            'Sleep Duration': request.form['Sleep_Duration'],
            'Dietary Habits': request.form['Dietary_Habits'],
            'region': request.form['region'],
            'degree_group': request.form['degree_group'],
            'Financial_Stress_Category': request.form['Financial_Stress'],
            'Study_Satisfaction_Category': request.form['Study_Satisfaction'],
            'Academic_Pressure_Category': request.form['Academic_Pressure']
        }
        app.logger.debug(f'Form data received: {data}')
        # Preprocess input
        X = preprocess_input(data, scaler, feature_columns)
        # Predict probability
        prob = model.predict_proba(X)[0][1]
        probability = round(prob * 100, 2)
        prediction = 'Yes' if prob > 0.45 else 'No'
        app.logger.debug(f'Prediction: {prediction}, Probability: {probability}%')
    return render_template('index.html', prediction=prediction, probability=probability)

@app.route('/therapy')
def therapy():
    app.logger.debug('Rendering therapy.html')
    return render_template('therapy.html')

# Log static file requests for debugging
@app.route('/static/<path:filename>')
def static_files(filename):
    app.logger.debug(f'Serving static file: {filename}')
    return app.send_static_file(filename)

if __name__ == '__main__':
    app.run(debug=True)