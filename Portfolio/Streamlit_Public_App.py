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

sagemaker_session = sagemaker.Session(boto_session=session)

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Build the path to the 'src' directory (relative to the notebook's location)
#module_path = os.path.abspath('..') 

# Add the 'src' directory to the system path list
#if module_path not in sys.path:
#    sys.path.append(module_path)

from src.feature_utils import get_bitcoin_historical_prices 
from src.feature_utils import extract_features 

df_prices = get_bitcoin_historical_prices()
df_features = extract_features()

MIN_VAL = 0.5*df_prices.min()[0]
MAX_VAL = 2*df_prices.max()[0]
STEP_VAL = 100
DEFAULT_VAL = df_prices.mean()[0]

# --- Configuration: Model Registry ---
# In a real app, this would be loaded from a config file or database.
# The 'inputs' list defines the features required by each model.
MODEL_ENDPOINTS = {
    "MSFT Stock Predictor": {
        "endpoint": "lasso-pipeline-endpoint-auto",
        "inputs": [
            {"name": "GOOGL", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step":0.01},
            {"name": "IBM", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step":0.01},
            {"name": "DEXJPUS", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step":0.01},
            {"name": "DEXUSUK", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step":0.01},
            {"name": "SP500", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step":0.01},
            {"name": "DJIA", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step":0.01},
            {"name": "VIXCLS", "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step":0.01}
        ] 										
    },
    "Bitcoin Signal Predictor": {
        "endpoint": "logistic-pipeline-endpoint-auto-1",
        "inputs": [
            {"name": "Close Price", "type": "number", "min": MIN_VAL, "max": MAX_VAL, "default": DEFAULT_VAL, "step":STEP_VAL}
        ]
    }
}

FEATURE_KEYS = [ "GOOGL", "IBM", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS"]

# --- Backend Logic (Placeholder for AWS API Call) ---
@st.cache_resource
def call_model_api(endpoint_url, feature_values):

    predictor = Predictor(
        endpoint_name=endpoint_url,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer(),
        # Linear Learner typically outputs JSON lines with scores
        deserializer=JSONDeserializer() 
    )
    
    print("\n[Tool Executor]: Formatting data for SageMaker...")
    
    # Extract values in the predefined order
    try:
        if len(feature_values.values()) > 1:
            data_row = [feature_values[key] for key in FEATURE_KEYS]
            #input_array = np.array(data_row).reshape(1, -1)
            #input_df = pd.DataFrame(input_array)
            input_df = pd.concat([df_features,pd.DataFrame([data_row],columns=df_features.columns)])
            prediction_result = predictor.predict(input_df)
            final_prediction = pd.DataFrame(prediction_result).values[-1][0]
            #final_prediction = prediction_result#['predictions'][0]['score']
            return final_prediction, 200
        else:
            data_row = feature_values.values()
            input_df = pd.concat([df_prices,pd.DataFrame(data_row,columns=df_prices.columns)])
            prediction_result = predictor.predict(input_df)
            final_prediction = pd.DataFrame(prediction_result).replace({-1:"SELL",0:"HOLD",1:"BUY"}).values[-1][0]
            return final_prediction, 200
        
    except KeyError as e:
        print(f"Error: Missing required feature {e}.")
        return 0.0 # Return a zero prediction on error
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
                    step=feature['step']
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

        st.metric(
            label=f"Predicted Value ({selected_model_name})", 
            value=f"{prediction_result}" if isinstance(prediction_result, (int, float)) else prediction_result
        )
        
    else:
        st.error(f"Prediction failed with status code {status_code}. See details below.")

st.markdown("---")
