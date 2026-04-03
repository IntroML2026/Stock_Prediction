import joblib
import os
import pandas as pd
import numpy as np
import sys
import json

model_dir = os.environ.get('SM_MODEL_DIR') 

if model_dir not in sys.path:
    sys.path.append(model_dir)

from src.Custom_Classes import FeatureEngineer

# --- REQUIRED FUNCTION 1: model_fn ---
def model_fn(model_dir):
    """
    Loads the serialized Scikit-learn pipeline from the model directory.
    This function is executed once when the endpoint container starts.
    """
    print(f"Loading model from {model_dir}")

    # Load the entire fitted pipeline object from the saved file
    file_path = os.path.join(model_dir, 'finalized_pca_model.joblib')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at {file_path}")
        
    best_pipeline = joblib.load(file_path)
    print("Pipeline loaded successfully.")
    return best_pipeline

# --- REQUIRED FUNCTION 2: predict_fn ---
def predict_fn(input_data, model):
    """
    Applies the loaded pipeline (model) to the incoming request data.
    This function runs for every prediction request to the endpoint.
    
    :param input_data: Data converted from the request body (default: numpy array)
    :param model: The pipeline object returned by model_fn
    :return: The prediction result (e.g., predicted values)
    """
    print("Generating predictions...")
    
    # SageMaker's default deserializer often passes numpy arrays. 
    # Since our Scikit-learn pipeline expects features in the order they were trained,
    # we convert the input NumPy array back to a DataFrame for safety/consistency,
    # though in simple cases, the pipeline can handle NumPy directly.
    if isinstance(input_data, np.ndarray):
        # We assume the input data is a 2D array of features, matching the training order
        input_df = pd.DataFrame(input_data)
    else:
        # Handle other formats if necessary (e.g., if you use a JSON serializer)
        input_df = input_data 
        
    # The predict call executes all steps (imputer, scaler, lasso) sequentially
    predictions = model.predict(input_df)
    
    print("Prediction complete.")
    return predictions


def input_fn(request_body, request_content_type):
    print(f"Receiving data of type: {request_content_type}")

    dataset = pd.read_csv(r'Portfolio/SP500Data.csv',index_col=0)
    target = 'MSFT'
    random = 'IBM'
    random_price = json.loads(request_body)[random]
    closest_date = (dataset[random] - float(random_price)).abs().idxmin()

    return_period = 5

    X = np.log(dataset.drop([target],axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [name + "_CR_Cum" for name in X.columns]

    return X.loc[[closest_date]]

# Note: SageMaker uses its own internal serializers/deserializers (like CSVSerializer) 
# to handle the input/output formatting, so we only need model_fn and predict_fn.
