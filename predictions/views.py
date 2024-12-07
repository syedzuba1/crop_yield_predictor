import pandas as pd
import pickle
from django.shortcuts import render
from .forms import EconomicImpactForm, AdaptationPredictionForm, CrimePredictionForm, BudgetPredictionForm
import numpy as np
import joblib



#with open('predictions/models/preprocessing.pkl', 'rb') as file:
#    preprocessor1 = pickle.load(file)
with open('predictions/models/logistic_regression_model.pkl', 'rb') as file:
    model1 = pickle.load(file)

# Load the models and scaler
def load_pickle_files():
    with open('predictions/models/preprocessing_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    with open('predictions/models/ridge_model_escapees.pkl', 'rb') as escapee_model_file:
        ridge_model_escapees = pickle.load(escapee_model_file)
    
    with open('predictions/models/ridge_model_mental.pkl', 'rb') as mental_model_file:
        ridge_model_mental = pickle.load(mental_model_file)
    
    return scaler, ridge_model_escapees, ridge_model_mental

def load_model_and_scaler():
    with open('trained_model.pkl', 'rb') as model_file:
        model, scaler = pickle.load(model_file)
    return model, scaler

def load_preprocessed_data():
    with open('preprocessed_data.pkl', 'rb') as data_file:
        preprocessed_data = pickle.load(data_file)
    return preprocessed_data



# Load the model, scaler, and dataset once to avoid reloading on every request
with open("predictions/models/crime_model.pkl", "rb") as f:
    model, scaler, combined_df = pickle.load(f)


def Home(request):
    return render(request, 'Home.html')


# Define mappings based on training
age_group_mapping = {
    '16-18 years': 0,
    '18-30 years': 1,
    '30-50 years': 2,
    'Above 50 years': 3
}

gender_mapping = {
    'Male': 0,
    'Female': 1
}

def predict_view_1(request):
    result = None
    if request.method == 'POST':
        form = EconomicImpactForm(request.POST)
        if form.is_valid():
            # Extract data from form
            cleaned_data = form.cleaned_data
            age = cleaned_data['age']
            gender = cleaned_data['gender']

            # Determine the age group
            if age < 16:
                result = "Error: Age must be 16 or older."
                return render(request, 'predict_1.html', {'form': form, 'result': result})
            elif 16 <= age <= 18:
                age_group = '16-18 years'
            elif 18 < age <= 30:
                age_group = '18-30 years'
            elif 30 < age <= 50:
                age_group = '30-50 years'
            else:
                age_group = 'Above 50 years'

            # Map inputs to encoded format
            gender_encoded = gender_mapping.get(gender)
            age_group_encoded = age_group_mapping.get(age_group)

            # Create input DataFrame for the model
            transformed_data = pd.DataFrame([[gender_encoded, age_group_encoded]], columns=['Gender', 'Age Group'])

            # Make prediction
            prediction = model1.predict(transformed_data)[0]

            # Format the result
            result = f"Predicted Status: {'Undertrial' if prediction == 1 else 'Convicted'}"
    else:
        form = EconomicImpactForm()

    return render(request, 'predict_1.html', {'form': form, 'result': result})


def predict_view_2(request):
    result = None
    if request.method == 'POST':
        form = AdaptationPredictionForm(request.POST)
        if form.is_valid():
            # Extract data from the form
            cleaned_data = form.cleaned_data
            total_education_facilities = cleaned_data['total_education_facilities']
            escapee_rate = cleaned_data['escapee_rate']
            mental_illness_rate = cleaned_data['mental_illness_rate']

            # Prepare data in the same format as the training data
            input_data = pd.DataFrame([{
                'total_education_facilities': total_education_facilities,
                'escapee_rate': escapee_rate,
                'mental_illness_rate': mental_illness_rate
            }])

            # Load the models and scaler
            scaler, ridge_model_escapees, ridge_model_mental = load_pickle_files()

            # Preprocess the data using the scaler (standardize the features)
            input_data_scaled = scaler.transform(input_data)

            # Predict escapees and mental illness using the models
            predicted_escapees = ridge_model_escapees.predict(input_data_scaled)
            predicted_mental_illness = ridge_model_mental.predict(input_data_scaled)

            # Format the result
            result = {
                'predicted_escapees': f"Predicted number of escapees: {predicted_escapees[0]:.2f}",
                'predicted_mental_illness': f"Predicted number of mental illness cases: {predicted_mental_illness[0]:.2f}"
            }

    else:
        form = AdaptationPredictionForm()

    return render(request, 'predict_adaptation.html', {'form': form, 'result': result})

def predict_view_3(request):
    result = None
    if request.method == 'POST':
        form = CrimePredictionForm(request.POST)
        if form.is_valid():
            # Extract the form data
            cleaned_data = form.cleaned_data
            state = cleaned_data['state']
            crime_type = cleaned_data['crime_type']

            # Filter data for the given state and crime type
            filtered_df = combined_df[(combined_df['STATE/UT'] == state) & (combined_df['CRIME HEAD'] == crime_type)]
            
            if filtered_df.empty:
                result = f"No data available for state: {state} and crime type: {crime_type}"
            else:
                # Prepare features for prediction
                feature_data = filtered_df[['Grand Total_convicted', 'Grand Total_undertrial']].values
                
                # Preprocess the data using the scaler
                input_data_scaled = scaler.transform(feature_data)

                # Make prediction using the trained model
                prediction = model.predict(input_data_scaled)
                prediction_proba = model.predict_proba(input_data_scaled)

                # Format the result
                status = "Convicted" if prediction[0] == 1 else "Undertrial"
                confidence = max(prediction_proba[0]) * 100
                result = f": {status}"
    else:
        form = CrimePredictionForm()
        
    return render(request, 'predict_form.html', {'form': form, 'result': result})

MODEL_PATH = 'predictions/models/budget_prediction_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model4 = pickle.load(f)


def predict_view_4(request):
    result = None

    if request.method == 'POST':
        form = BudgetPredictionForm(request.POST)
        if form.is_valid():
            # Extract form data
            state_ut = form.cleaned_data['state_ut'].upper()
            year = form.cleaned_data['year']
            num_years = form.cleaned_data['num_years']

            

            input_data = pd.DataFrame({
                'STATE/UT': [state_ut],  # Wrap in a list
                'Year': [year],          # Wrap in a list
                'num_years': [num_years] # Wrap in a list
            })
            
            try:
                # Directly predict using the model
                predicted_budgets = model4.predict(input_data)
                
                result = f"Predicted budgets: {predicted_budgets[0]}"
            except Exception as e:
                result = f"Error during prediction: {str(e)}"
    else:
        form = BudgetPredictionForm()
    return render(request, 'predict_lstm_form.html', {'form': form, 'result': result,})