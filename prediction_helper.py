# codebasics ML course: codebasics.io, all rights reserverd

import pandas as pd
import joblib

model_young = joblib.load("artifacts\model_young.joblib")
model_rest = joblib.load("artifacts\model_rest.joblib")
scaler_young = joblib.load("artifacts\scaler_young.joblib")
scaler_rest = joblib.load("artifacts\scaler_rest.joblib")

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score

def preprocess_input(input_dict):
    # Define the expected columns and initialize the DataFrame with zeros
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])
    # df.fillna(0, inplace=True)

    # Manually assign values for each categorical input based on input_dict
    # Define encoding dictionaries
    region_encoding = {
        "Northwest": "region_Northwest",
        "Southeast": "region_Southeast",
        "Southwest": "region_Southwest"
    }

    bmi_encoding = {
        "Obesity": "bmi_category_Obesity",
        "Overweight": "bmi_category_Overweight",
        "Underweight": "bmi_category_Underweight"
    }

    smoking_encoding = {
        "Occasional": "smoking_status_Occasional",
        "Regular": "smoking_status_Regular"
    }

    employment_encoding = {
        "Salaried": "employment_status_Salaried",
        "Self-Employed": "employment_status_Self-Employed"
    }

    # insurance_plan_encoding is assumed already defined
    # Example: insurance_plan_encoding = {"Basic": 1, "Premium": 2, "Gold": 3}

    # Main loop
    for key, value in input_dict.items():

        # Gender
        if key == "Gender" and value == "Male":
            df["gender_Male"] = 1

        # Region
        elif key == "Region" and value in region_encoding:
            df[region_encoding[value]] = 1

        # Marital Status
        elif key == "Marital Status" and value == "Unmarried":
            df["marital_status_Unmarried"] = 1

        # BMI Category
        elif key == "BMI Category" and value in bmi_encoding:
            df[bmi_encoding[value]] = 1

        # Smoking Status
        elif key == "Smoking Status" and value in smoking_encoding:
            df[smoking_encoding[value]] = 1

        # Employment Status
        elif key == "Employment Status" and value in employment_encoding:
            df[employment_encoding[value]] = 1

        # Insurance Plan
        elif key == "Insurance Plan":
            df["insurance_plan"] = insurance_plan_encoding.get(value, 1)

        # Direct numerical features
        elif key == "Age":
            df["age"] = value
        elif key == "Number of Dependants":
            df["number_of_dependants"] = value
        elif key == "Income in Lakhs":
            df["income_lakhs"] = value
        elif key == "Genetical Risk":
            df["genetical_risk"] = value


    

    # Assuming the 'normalized_risk_score' needs to be calculated based on the 'age'
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)

    return df

def handle_scaling(age, df):
    # scale age and income_lakhs column
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])
