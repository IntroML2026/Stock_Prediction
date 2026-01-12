import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap


# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices, extract_features

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]


# AWS Session Management
@st.cache_resource # Use this to avoid downloading the file every time the page refreshes
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

# Cache the Explainer (Downloading and loading is slow)
@st.cache_resource
def load_shap_explainer(_session, bucket, key):
    s3_client = _session.client('s3')
    local_path = '/tmp/explainer.shap'
    
    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
        
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

session = get_session(aws_id, aws_secret, aws_token)
explainer = load_shap_explainer(session, aws_bucket, "explainer/explainer.shap")

sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_prices = get_bitcoin_historical_prices()
df_features = extract_features()

# Dynamic bounds for Bitcoin model
MIN_VAL = 0.5 * df_prices.iloc[:, 0].min()
MAX_VAL = 2.0 * df_prices.iloc[:, 0].max()
DEFAULT_VAL = df_prices.iloc[:, 0].mean()

MODEL_ENDPOINTS = {
    "MSFT Stock Predictor": {
        "endpoint": "lasso-pipeline-endpoint-auto-15",
        "keys": ["GOOGL", "IBM", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS"],
        "inputs": [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} for k in ["GOOGL", "IBM", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS"]]
    },
    "Bitcoin Signal Predictor": {
        "endpoint": "logistic-pipeline-endpoint-auto-1",
        "keys": ["Close Price"],
        "inputs": [{"name": "Close Price", "type": "number", "min": MIN_VAL, "max": MAX_VAL, "default": DEFAULT_VAL, "step": 100.0}]
    }
}

# Prediction Logic
def call_model_api(input_df, model_name):
    config = MODEL_ENDPOINTS[model_name]
    
    predictor = Predictor(
        endpoint_name=config["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer() 
    )

    try:
        # Create input row based on the specific keys required for this model
        #data_row = [feature_dict[k] for k in config["keys"]]
        
        # Prepare data (Stock predictor uses df_features, Bitcoin uses df_prices)
        #base_df = df_features if "MSFT" in model_name else df_prices
        #input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)])
        
        raw_pred = predictor.predict(input_df)

        pred_val = pd.DataFrame(raw_pred).values[-1][0]

        # Formatting for Bitcoin classification
        if "Bitcoin" in model_name:
            mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
            return mapping.get(pred_val, pred_val), 200
        
        return round(float(pred_val), 4), 200

    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability
def display_explanation(shap_values):
    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(fig)
    # top feature   
    top_feature = shap_values[0].feature_names[0]
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# Streamlit UI
st.set_page_config(page_title="ML Deployment Compiler", layout="wide")
st.title("👨‍💻 ML Deployment Compiler")

selected_model = st.sidebar.selectbox("Choose a Model:", list(MODEL_ENDPOINTS.keys()))
config = MODEL_ENDPOINTS[selected_model]

with st.form("pred_form"):
    st.subheader(f"Inputs for {selected_model}")
    cols = st.columns(2)
    user_inputs = {}
    
    for i, inp in enumerate(config["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
            )
    
    submitted = st.form_submit_button("Run Prediction")

if submitted:

    data_row = [user_inputs[k] for k in config["keys"]]
    # Prepare data (Stock predictor uses df_features, Bitcoin uses df_prices)
    base_df = df_features if "MSFT" in selected_model else df_prices
    input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)])
    
    res, status = call_model_api(input_df,selected_model)
    if status == 200:
        st.metric("Prediction Result", res)
        shap_values = explainer(input_df)
        display_explanation(shap_values)
    else:
        st.error(res)



