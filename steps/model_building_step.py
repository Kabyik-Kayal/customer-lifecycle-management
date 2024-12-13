import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd  
import warnings
warnings.filterwarnings("ignore")
from zenml import step, Model
from zenml.client import Client
from zenml.logger import get_logger
from typing import Dict, Any
from src.model_building import GeneralizedModelTrainingStrategy, ModelTrainer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from materializer.custom_materializer import SklearnPipelineMaterializer, NumpyInt64Materializer
import mlflow

# Initialize logger
logger = get_logger(__name__)

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
if experiment_tracker is None:
    raise ValueError("No experiment tracker is configured in the active ZenML stack.")
else:
    logger.info(f"Using experiment tracker: {experiment_tracker.name}")

# Define the model
model = Model(
    name="CLTV_Prediction",
    version=None,
    license="MIT",
    description="A model to predict customer lifetime value (CLTV) using RF, XGBoost, or LightGBM.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model, output_materializers=[SklearnPipelineMaterializer, NumpyInt64Materializer])
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_choice: str = "xgboost",
    fine_tuning: bool = True
) -> Pipeline:
    """
    A ZenML step for building and training a model using GeneralizedModelTrainingStrategy.

    Args:
        X_train: Training feature set.
        y_train: Training target values.
        model_choice: Choice of model to train ("random_forest", "xgboost", "lightgbm").
        fine_tuning: Whether to perform hyperparameter tuning.

    Returns:
        A trained sklearn pipeline.
    """
    logger.info(f"Starting model training with {model_choice}.")

    # Define preprocessors
    numerical_features = ['frequency', 'total_amount', 'avg_order_value', 'recency', 'customer_age', 'lifetime', 'purchase_frequency']
    preprocessors = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="passthrough"
    )

    # Define parameter grids for fine-tuning
    param_grids = {
        "random_forest": {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 10, 20]},
        "xgboost": {'n_estimators': [100, 200, 300, 500],'learning_rate': [0.01, 0.05, 0.1, 0.2],'max_depth': [3, 5, 7, 9],'min_child_weight': [1, 3, 5, 7],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'gamma': [0, 0.1, 0.3, 0.5],'reg_alpha': [0, 0.1, 1, 10],'reg_lambda': [1, 5, 10, 20]},
        "lightgbm": {'model__num_leaves': [31, 50, 70], 'model__learning_rate': [0.01, 0.1, 0.2]},
    }

    # Model mapping
    model_mapping = {
        "random_forest": RandomForestRegressor(random_state=42),
        "xgboost": XGBRegressor(random_state=42),
        "lightgbm": LGBMRegressor(random_state=42),
    }

    if model_choice not in model_mapping:
        raise ValueError(f"Invalid model choice: {model_choice}. Choose from {list(model_mapping.keys())}")

    # Initialize strategy
    strategy = GeneralizedModelTrainingStrategy(
        model=model_mapping[model_choice],
        param_grid=param_grids.get(model_choice),
    )

    trainer = ModelTrainer(strategy)
    trained_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessors),
        ("model", trainer.train(X_train, y_train, fine_tuning))
    ])

    # End any active MLflow run
    if mlflow.active_run():
        mlflow.end_run()

    # Start a new MLflow run and log the model
    with mlflow.start_run():
        try:
            # Enable MLflow autologging
            mlflow.sklearn.autolog()
            logger.info("Fitting the pipeline.")
            trained_pipeline.fit(X_train, y_train)
            logger.info("Pipeline fitting completed.")

            # Log the model
            input_example = X_train.iloc[0].to_dict()
            mlflow.set_experiment(experiment_tracker.name)
            model_info = mlflow.sklearn.log_model(
                sk_model=trained_pipeline,
                artifact_path="model",
                input_example=input_example
            )
            model_uri = model_info.model_uri
            logger.info("Model logged to MLflow.")

            # Register the model in MLflow Model Registry
            registered_model_name = "CLTV_Prediction"
            registered_model = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            logger.info(f"Model registered in MLflow Model Registry with name: {registered_model_name}")

            return trained_pipeline
        except Exception as e:
            logger.error(f"Model training and logging failed: {e}")
            raise
        finally:
            mlflow.end_run()
