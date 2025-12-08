import warnings, requests, zipfile, io
import pandas as pd
import numpy as np

import streamlit as st

import os
import boto3
import sagemaker
from sagemaker.image_uris import retrieve
from sklearn.model_selection import train_test_split

from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

warnings.simplefilter('ignore')

ACCESS_KEY = st.secrets["ACCESS_KEY"]
SECRET_KEY = st.secrets["SECRET_KEY"]

REGION = 'us-east-1'


session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION
)

s3_resource = session.resource('s3')
s3_client = session.client('s3')

bucket_name = r"my-test-bucket-for-boto3-check-12345"

# Model Mapping in Python
MODEL_ENDPOINTS = {
    "MSFT Stock Predictor": {
        "endpoint": "linear-learner-2025-12-05-04-32-57-036",
        "inputs": ["temp", "price_index"] # For dynamic form generation
    },
    "Bitcoin Signal Predictor": {
        "endpoint": "linear-learner-2025-12-05-04-32-57-036",
        "inputs": ["transaction_amount", "time_of_day"]
    }
}

sagemaker_session = sagemaker.Session(boto_session=session)


# --- Configuration: Model Registry ---
# In a real app, this would be loaded from a config file or database.
# The 'inputs' list defines the features required by each model.
MODEL_ENDPOINTS = {
    "MSFT Stock Predictor": {
        "endpoint": "linear-learner-2025-12-05-04-32-57-036",
        "inputs": [
            {"name": "MSFT_WT", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "GOOGL", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "SP500", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "MSFT_3WT", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "DJIA", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "MSFT_6WT", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "DEXJPUS", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "MSFT_12WT", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "VIXCLS", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "DEXUSUK", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "IBM", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0}
        ] 										
    },
    "Bitcoin Signal Predictor": {
        "endpoint": "linear-learner-2025-12-05-04-32-57-036",
        "inputs": [
            {"name": "MSFT_WT", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "GOOGL", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "SP500", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "MSFT_3WT", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "DJIA", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "MSFT_6WT", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "DEXJPUS", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "MSFT_12WT", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "VIXCLS", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "DEXUSUK", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0},
            {"name": "IBM", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0}
        ]
    }
}

FEATURE_KEYS = [ "GOOGL", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS", "MSFT_WT", "MSFT_3WT","MSFT_6WT", "MSFT_12WT", "IBM"]

# --- Backend Logic (Placeholder for AWS API Call) ---
@st.cache_resource
def call_model_api(endpoint_url, feature_values):

    linear_predictor = Predictor(
        endpoint_name=endpoint_url,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer(),
        # Linear Learner typically outputs JSON lines with scores
        deserializer=JSONDeserializer() 
    )
    
    print("\n[Tool Executor]: Formatting data for SageMaker...")
    
    # Extract values in the predefined order
    try:
        data_row = [feature_values[key] for key in FEATURE_KEYS]
        input_array = np.array(data_row).reshape(1, -1)
    except KeyError as e:
        print(f"Error: Missing required feature {e}.")
        return 0.0 # Return a zero prediction on error

    input_df = pd.DataFrame(input_array)
    #csv_input_string = input_df.to_csv(header=False, index=False)
     
    try:
         prediction_result = linear_predictor.predict(input_df)
         final_prediction = prediction_result#['predictions'][0]['score'] 
         return final_prediction, 200
    except Exception as e:
         print(f"[Tool Executor]: ERROR: SageMaker prediction failed: {e}"), 500
         return f"Prediction failed due to API error: {e}", 500

# --- Streamlit Frontend Definition ---

st.set_page_config(
    page_title="Deployed Model Compiler",
    layout="wide"
)

st.title("👨‍💻 ML Deployment Compiler")
st.markdown("Select a deployed model from the semester and input features to get a prediction.")

# Model Selection
st.sidebar.header("Model Selection")
model_names = list(MODEL_ENDPOINTS.keys())
selected_model_name = st.sidebar.selectbox(
    "Choose a Model:",
    model_names
)

# Get the configuration for the selected model
model_config = MODEL_ENDPOINTS[selected_model_name]
endpoint = model_config["endpoint"]
required_inputs = model_config["inputs"]

st.subheader(f"Model: {selected_model_name}")
st.markdown(f"**Endpoint:** `{endpoint}`")

st.markdown("---")

# Dynamic Input Form Generation
st.subheader("Input Features")
input_data = {} # Dictionary to store user inputs

with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)
    
    # Iterate through the required inputs and create appropriate widgets
    for i, feature in enumerate(required_inputs):
        
        # Determine which column to place the widget in
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            name = feature['name'].replace('_', ' ').upper()
            
            if feature['type'] == 'number':
                # Numerical input (e.g., temperature, age)
                input_data[feature['name']] = st.number_input(
                    name,
                    min_value=feature['min'],
                    max_value=feature['max'],
                    value=feature['default'],
                    step=0.01
                )
            
            elif feature['type'] == 'checkbox':
                # Boolean input (e.g., promotion_running)
                input_data[feature['name']] = st.checkbox(
                    name,
                    value=feature['default']
                )
            
            elif feature['type'] == 'selectbox':
                # Categorical input (e.g., region)
                input_data[feature['name']] = st.selectbox(
                    name,
                    options=feature['options'],
                    index=feature['options'].index(feature['default'])
                )
    
    # Submission button for the form
    st.markdown("---")
    submitted = st.form_submit_button("Run Prediction")

# Prediction Execution and Display
st.markdown("## Prediction Output")

if submitted:
    st.info("Preparing data and calling model API...")
    
    # Call the API
    prediction_result, status_code = call_model_api(endpoint, input_data)

    # Display Results
    if status_code == 200:
        st.success("Prediction successful!")
        
        if "predictions" in prediction_result:
            # Display the main result using st.metric for emphasis
            prediction_result_tmp = prediction_result['predictions'][0]['score']
            st.metric(
                label=f"Predicted Value ({selected_model_name})", 
                value=f"{prediction_result_tmp}" if isinstance(prediction_result_tmp, (int, float)) else prediction_result_tmp
            )

        # Display all returned data
        st.subheader("API Response")
        st.json(prediction_result)
        
    else:
        st.error(f"Prediction failed with status code {status_code}. See details below.")
        st.json(prediction_result)

st.markdown("---")
