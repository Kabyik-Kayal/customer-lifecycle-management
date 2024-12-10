import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from zenml import pipeline
from zenml.client import Client
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from steps.model_deployer_step import model_fetcher

@pipeline
def deploy_pipeline():
    """Deployment pipeline that fetches the latest model from MLflow.
    """
    model_uri = model_fetcher()
    
    deploy_model = mlflow_model_deployer_step(
        model_name="CLTV_Prediction",
        model = model_uri
    )

if __name__ == "__main__":
    # Run the pipeline
    deploy_pipeline()