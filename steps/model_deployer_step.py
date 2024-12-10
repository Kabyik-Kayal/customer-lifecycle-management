import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from zenml import step
from mlflow import MlflowClient
import logging


@step
def model_fetcher() -> str:
    """Fetch the latest model version URI from MLflow."""
    model_name = "CLTV_Prediction"
    alias_name = "champion"
    logging.info(f"Fetching latest version for model: {model_name}")
    
    # Set the MLflow tracking URI
    os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'
    
    client = MlflowClient()
    
    # Get all versions of the model
    versions = client.get_model_version_by_alias(model_name, alias_name)
    
    if not versions:
        logging.error(f"No versions found for model {model_name}")
        raise RuntimeError(f"Registered Model with name={model_name} not found")
    
    model_uri = f"models:/{model_name}/{versions}"
    logging.info(f"Returning model URI: {model_uri}")
    return model_uri