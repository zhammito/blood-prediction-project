import pandas as pd
import numpy as np
import pandasql as ps
import streamlit as st
from xgboost import XGBRegressor
import pickle

# Load historical data and apply SQL transformation
df = pd.read_csv('blood_demand_dataset 2.1.csv')
query = """
WITH df1 AS (
  SELECT 
    PBloodGroupTested,
    Req_Comp,
    DATE(Request_Date) AS REQDT,
    SUM(Quantity) AS n_qty_n1d,
    MAX(CASE WHEN CAST(strftime('%w', Request_Date) AS INTEGER) = 0 THEN 1 ELSE 0 END) AS ind_dow0,
    MAX(CASE WHEN CAST(strftime('%w', Request_Date) AS INTEGER) = 1 THEN 1 ELSE 0 END) AS ind_dow1,
    MAX(CASE WHEN CAST(strftime('%w', Request_Date) AS INTEGER) = 2 THEN 1 ELSE 0 END) AS ind_dow2,
    MAX(CASE WHEN CAST(strftime('%w', Request_Date) AS INTEGER) = 3 THEN 1 ELSE 0 END) AS ind_dow3,
    MAX(CASE WHEN CAST(strftime('%w', Request_Date) AS INTEGER) = 4 THEN 1 ELSE 0 END) AS ind_dow4,
    MAX(CASE WHEN CAST(strftime('%w', Request_Date) AS INTEGER) = 5 THEN 1 ELSE 0 END) AS ind_dow5,
    MAX(CASE WHEN CAST(strftime('%w', Request_Date) AS INTEGER) = 6 THEN 1 ELSE 0 END) AS ind_dow6,
    SUM(CASE WHEN Gender = 'Female' THEN Quantity ELSE 0 END) AS n_qty_fem_n1d
  FROM df
  GROUP BY PBloodGroupTested, Req_Comp, REQDT
),
rolling_aggregates AS (
  SELECT
    PBloodGroupTested,
    Req_Comp,
    REQDT,
    SUM(n_qty_n1d) OVER (PARTITION BY PBloodGroupTested, Req_Comp ORDER BY REQDT ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS n_qty_n7d,
    SUM(n_qty_n1d) OVER (PARTITION BY PBloodGroupTested, Req_Comp ORDER BY REQDT ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS n_qty_n21d
  FROM df1
)
SELECT 
  df1.PBloodGroupTested,
  df1.Req_Comp,
  df1.REQDT,
  df1.ind_dow0,
  df1.ind_dow1,
  df1.ind_dow2,
  df1.ind_dow3,
  df1.ind_dow4,
  df1.ind_dow5,
  df1.ind_dow6,
  df1.n_qty_n1d,
  rolling_aggregates.n_qty_n7d,
  rolling_aggregates.n_qty_n21d,
  df1.n_qty_fem_n1d
FROM df1
LEFT JOIN rolling_aggregates ON df1.PBloodGroupTested = rolling_aggregates.PBloodGroupTested 
  AND df1.Req_Comp = rolling_aggregates.Req_Comp 
  AND df1.REQDT = rolling_aggregates.REQDT
"""
historical_data = ps.sqldf(query, locals())

# Function to load a model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load your models
model_1d = load_model('xgboost_model_1d.pkl')
model_7d = load_model('model_7d.pkl')
model_21d = load_model('model_21d.pkl')

# Function to preprocess user input
def preprocess_user_input(input_details, historical_data):
    df_user = pd.DataFrame([input_details])
    df_user['Request_Date'] = pd.to_datetime(df_user['Request_Date'])
    
    # Deriving day of the week features
    for i in range(7):
        df_user[f'ind_dow{i}'] = (df_user['Request_Date'].dt.dayofweek == i).astype(int)
    
    # Assuming Gender and Quantity are input by the user
    df_user['n_qty_fem_n1d'] = np.where(df_user['Gender'] == 'Female', df_user['Quantity'], 0)

    # Handling holidays
    holidays = ['2023-01-26', '2023-08-15', '2023-10-02']
    df_user['ind_holiday'] = df_user['Request_Date'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)
    
    df_user['n_qty_n1d'] = df_user['Quantity']

    # Placeholder logic for 'n_qty_n7d' and 'n_qty_n21d' - to be replaced with your actual logic
    df_user['n_qty_n7d'] = 0  # Placeholder, replace with actual calculation
    df_user['n_qty_n21d'] = 0  # Placeholder, replace with actual calculation

    # Assuming rolling averages need calculation based on historical data
    # Placeholder for 'rolling_avg_7d' and 'rolling_avg_14d', replace with actual logic
    df_user['rolling_avg_7d'] = 0  # Placeholder
    df_user['rolling_avg_14d'] = 0  # Placeholder

    # Adding month and day_of_week based on 'Request_Date'
    df_user['month'] = df_user['Request_Date'].dt.month
    df_user['day_of_week'] = df_user['Request_Date'].dt.dayofweek
    
    # Append new input to historical data for rolling feature calculation
    combined_data = pd.concat([historical_data, df_user], ignore_index=True)
    combined_data['Request_Date'] = pd.to_datetime(combined_data['Request_Date'])
    combined_data.sort_values(by='Request_Date', inplace=True)

    # Filter for specific blood group and component
    filtered_data = combined_data[(combined_data['PBloodGroupTested'] == input_details['PBloodGroupTested']) & 
                                  (combined_data['Req_Comp'] == input_details['Req_Comp'])]

    filtered_data = filtered_data.copy()

    # Then, when you set new columns, use .loc to ensure the operations are done explicitly on the DataFrame copy
    filtered_data.loc[:, 'n_qty_n7d'] = filtered_data['Quantity'].rolling(window=7, min_periods=1).sum().shift()
    filtered_data.loc[:, 'n_qty_n21d'] = filtered_data['Quantity'].rolling(window=21, min_periods=1).sum().shift()
    filtered_data.loc[:, 'rolling_avg_7d'] = filtered_data['Quantity'].rolling(window=7, min_periods=1).mean().shift()
    filtered_data.loc[:, 'rolling_avg_14d'] = filtered_data['Quantity'].rolling(window=14, min_periods=1).mean().shift()
    # Extract the calculated features for the new input
    new_input_features = filtered_data.iloc[-1][['n_qty_n7d', 'n_qty_n21d', 'rolling_avg_7d', 'rolling_avg_14d']]
    
    # Update df_user with the calculated features
    for feature in ['n_qty_n7d', 'n_qty_n21d', 'rolling_avg_7d', 'rolling_avg_14d']:
        df_user[feature] = new_input_features[feature]

    # Return the updated DataFrame with new input and calculated features
    return df_user


# Define the logic for approving or rejecting the appeal
def approve_appeal(blood_supply, prediction_value, urgency_level):
    approval_threshold = {
        "Routine": 1.5,
        "Emergency": 0.8,
        "Planned Surgery": 1.2,
        "Other": 1.0
    }
    
    supply_demand_ratio = blood_supply / prediction_value if prediction_value else float('inf')
    
    if supply_demand_ratio >= approval_threshold.get(urgency_level, 1.0):
        return "Approved"
    else:
        return "Rejected"

   
# Prediction function
def make_prediction(model, processed_data, model_choice):
    # Adjust features for each model choice based on training details
    if model_choice == '1-Day':
        features = ['n_qty_n1d', 'ind_dow0', 'ind_dow1', 'ind_dow2', 'ind_dow3', 'ind_dow4',
                    'ind_dow5', 'ind_dow6', 'n_qty_fem_n1d', 'n_qty_n7d', 'n_qty_n21d', 'ind_holiday']
    elif model_choice == '7-Day':
        # Corrected to include the day-of-week indicators and n_qty_fem_n1d as per training
         features = ['n_qty_n1d', 'ind_dow0', 'ind_dow1', 'ind_dow2', 'ind_dow3', 'ind_dow4', 'ind_dow5', 'ind_dow6',
                                     'n_qty_n7d', 'n_qty_n21d', 'n_qty_fem_n1d', 'ind_holiday']
    elif model_choice == '21-Day':
        # Focused on rolling averages, month, day_of_week, and holiday indicator
        features = ['rolling_avg_7d', 'rolling_avg_14d', 'month', 'day_of_week', 'ind_holiday']

    # Filter the input data for the prediction to only include the relevant features
    prediction_data = processed_data[features]

    # Make prediction
    prediction = model.predict(prediction_data)
    return prediction

# Initialize session state for prediction and processed_input if not already present
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

# Streamlit UI
st.title('Blood Demand Prediction')

# Input form for user data
with st.form("user_input_form"):
    pblood_group_tested = st.selectbox('Blood Group Tested', ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'])
    req_comp = st.selectbox('Request Component', ['PRBC', 'WB', 'Platelets', 'RDP'])
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    quantity = st.number_input('Quantity', value=1, min_value=1)
    request_date = st.date_input('Request Date')
        # Additional fields based on dataset insights
    transfusion_indication = st.selectbox('Transfusion Indication', ['Surgery', 'Trauma', 'Other'])
    clinical_diagnosis = st.selectbox('Clinical Diagnosis', ['Sickle Cell Disease', 'Anemia', 'Thalassemia', 'Heart Surgery', 'Other'])
    healthcare_facility = st.selectbox('Healthcare Facility', ['Hospital', 'Blood Bank', 'Other'])
    urgency_level = st.selectbox('Urgency Level', ['Routine', 'Emergency', 'Planned Surgery', 'Other'])
    seasonal_health_trend = st.selectbox('Seasonal Health Trend', ['Normal', 'Flu Season', 'Dengue Outbreak', 'Other'])

    submit_user_input = st.form_submit_button("Submit User Input")

model_choice = st.radio('Choose the prediction model:', ('1-Day', '7-Day', '21-Day'))

# New form for model selection and prediction
with st.form("model_prediction_form"):
    submit_for_prediction = st.form_submit_button("Predict with Selected Model")

if submit_user_input:
    # Process the user input here but don't perform prediction yet
    user_input = {
        'PBloodGroupTested': pblood_group_tested,
        'Req_Comp': req_comp,
        'Gender': gender,
        'Quantity': quantity,
        'Request_Date': request_date.strftime('%Y-%m-%d')
    }
    
    processed_input = preprocess_user_input(user_input, historical_data)
    # Store the processed input in the session state to use it for prediction later
    st.session_state['processed_input'] = processed_input
    st.write("User input processed. Select a model and click 'Predict with Selected Model' to proceed.")

if submit_for_prediction and 'processed_input' in st.session_state:
    # Retrieve the processed input from session state
    processed_input = st.session_state['processed_input']

    # Choose the model based on the selection
    if model_choice == '1-Day':
        model_to_use = model_1d
    elif model_choice == '7-Day':
        model_to_use = model_7d
    else:  # '21-Day'
        model_to_use = model_21d

    # Perform the prediction and store the result in session state
    st.session_state['prediction'] = make_prediction(model_to_use, st.session_state['processed_input'], model_choice)
    st.write(f'Predictions using the {model_choice} model:')
    st.write(st.session_state['prediction'])
    
# Blood Appeal Submission section
st.write("## Blood Appeal Submission")
blood_supply = st.number_input("Current Blood Supply", value=0, min_value=0)
urgency_level = st.selectbox("Urgency Level", ["Routine", "Emergency", "Planned Surgery", "Other"])

# Process the appeal submission
submit_appeal = st.button("Submit Appeal")

if submit_appeal:
    # Before proceeding, check if 'prediction' is in the session state and has a value
    if 'prediction' in st.session_state and st.session_state['prediction'] is not None:
        # Use the stored prediction from session state for appeal logic
        prediction_value = st.session_state['prediction'][0] if isinstance(st.session_state['prediction'], np.ndarray) else st.session_state['prediction']
        appeal_result = approve_appeal(blood_supply, prediction_value, urgency_level)
        if appeal_result == "Approved":
            st.success("Your appeal has been approved.")
        else:
            st.error("Your appeal has been rejected.")
    else:
        st.error("No prediction available. Please complete the prediction step before submitting an appeal.")