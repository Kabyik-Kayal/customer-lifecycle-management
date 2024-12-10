import os
import logging
import joblib
import pandas as pd
from typing import Any
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Training Strategy
class ModelTrainingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool) -> Pipeline:
        pass


# Generalized Strategy for Model Training
class GeneralizedModelTrainingStrategy(ModelTrainingStrategy):
    def __init__(self, model, param_grid=None):
        self.model = model
        self.param_grid = param_grid

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool) -> Pipeline:
        model_name = self.model.__class__.__name__
        logging.info(f"Training {model_name} model.")
        try:
            # Hyperparameter tuning if enabled
            if fine_tuning and self.param_grid:
                logging.info(f"Starting hyperparameter tuning for {model_name}.")
                best_model = self._perform_hyperparameter_tuning(X_train, y_train)
            else:
                logging.info(f"Training {model_name} without hyperparameter tuning.")
                self.model.fit(X_train, y_train)
                best_model = self.model

            # Log metrics
            train_score = best_model.score(X_train, y_train)
            logging.info(f"Training score for {model_name}: {train_score}")

            # Save the model
            self._save_model(best_model, model_name)

            logging.info(f"Model training completed for {model_name}.")
            return best_model

        except Exception as e:
            logging.error(f"Error in training {model_name}: {e}")
            raise e

    def _perform_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if not self.param_grid:
            raise ValueError("Parameter grid is required for hyperparameter tuning.")
        
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            n_iter=10,
            cv=KFold(n_splits=3),
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        search.fit(X_train, y_train)
        logging.info(f"Best parameters for {self.model.__class__.__name__}: {search.best_params_}")
        return search.best_estimator_

    def _save_model(self, model: Pipeline, model_name: str) -> None:
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name.lower()}_cltv_model.pkl"
        joblib.dump(model, model_path)
        logging.info(f"{model_name} model saved successfully at {model_path}.")


# Model Trainer Context
class ModelTrainer:
    def __init__(self, strategy: ModelTrainingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelTrainingStrategy) -> None:
        self._strategy = strategy

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, fine_tuning: bool) -> Any:
        return self._strategy.build_and_train_model(X_train, y_train, fine_tuning)


if __name__ == "__main__":
    # # Example parameter grids for fine-tuning
    # random_forest_params = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
    # xgboost_params = {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]}
    # lightgbm_params = {"num_leaves": [31, 50, 70], "learning_rate": [0.01, 0.1, 0.2]}

    # # Initialize strategies for CLTV prediction models
    # strategies = {
    #     "random_forest": GeneralizedModelTrainingStrategy(RandomForestRegressor(random_state=42), random_forest_params),
    #     "xgboost": GeneralizedModelTrainingStrategy(XGBRegressor(random_state=42), xgboost_params),
    #     "lightgbm": GeneralizedModelTrainingStrategy(LGBMRegressor(random_state=42), lightgbm_params)
    # }

    # # Example usage with dummy data
    # # Replace this with your actual CLTV dataset
    # X_train = pd.DataFrame(...)  # Add your features
    # y_train = pd.Series(...)  # Add your target values (CLTV)

    # for model_name, strategy in strategies.items():
    #     logging.info(f"Training {model_name} model for CLTV prediction.")
    #     trainer = ModelTrainer(strategy)
    #     trained_model = trainer.train(X_train, y_train, fine_tuning=True)
    pass
