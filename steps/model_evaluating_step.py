import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing_extensions import Annotated
import pandas as pd
import numpy as np
import logging
import mlflow
from sklearn.pipeline import Pipeline
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from src.model_evaluator import ModelEvaluator, CLTVEvaluator

# Set up logging
logger = get_logger(__name__)


experiment_tracker=Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_evaluating_step(
    trained_model: Pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> dict:
    """
    Evaluates the trained model on the test dataset using regression metrics.

    Parameters:
    model (object) : The trained model (XGBoost or other regression model).
    X_test (pd.DataFrame) : The testing dataset.
    y_test (pd.Series) : The true target values for testing.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    logger.info("Starting model evaluation process")
    
    # Validate inputs
    if not isinstance(X_test, pd.DataFrame):
        logger.error("X_test is not a pandas DataFrame.")
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, (pd.Series, np.ndarray)):
        logger.error("y_test is not a pandas Series or NumPy array.")
        raise TypeError("y_test must be a pandas Series or NumPy array.")
    if hasattr(y_test, "isna") and y_test.isna().any():
        logger.error("y_test contains NaN values.")
        raise ValueError("y_test contains NaN values.")
    if X_test.isna().any().any():
        logger.error("X_test contains NaN values.")
        raise ValueError("X_test contains NaN values.")
    if not hasattr(trained_model, "predict"):
        logger.error("The provided model does not have a predict method.")
        raise AttributeError("Model must have a predict method.")
       
    try:
        # Predict on the test dataset
        logger.info("Generating predictions on the test dataset")
        preprocessor = trained_model.named_steps['preprocessor']
        logging.info(f"Evaluating model with parameters: {trained_model['model'].get_params()}")
        
        if preprocessor:
            # Log the columns being scaled if a scaler exists in the 'num' transformer
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'num':  # Check if the transformer is for numerical columns
                    logging.info(f"Numerical columns {cols} are being scaled using the scaler in the pipeline.")
   
            # Apply the preprocessing and scaling to the test data
            X_test_preprocessed  = trained_model.named_steps['preprocessor'].transform(X_test)
            # print("First 5 rows of X_test_processed:")
            # print(X_test_preprocessed[:5]) 
            
        else:
            logging.warning("No 'preprocessor' step found in the pipeline. Skipping preprocessing.")
            X_test_preprocessed  = X_test   

        evaluator = ModelEvaluator(CLTVEvaluator())
        metrics = evaluator.evaluate_model(trained_model.named_steps["model"], X_test_preprocessed, y_test)

        # log metrics to mlflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            
        return metrics        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise
