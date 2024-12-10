from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


"""
Model Evaluator module for CLTV predictions.
Implements concrete evaluation strategies following the Strategy pattern.
"""

class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies."""
    
    @abstractmethod
    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model predictions against true values.
        
        Args:
            y_test: Actual CLTV values
            y_pred: Predicted CLTV values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass

class CLTVEvaluator(EvaluationStrategy):
    """Concrete strategy for evaluating CLTV model performance."""
    
    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate CLTV predictions using standard regression metrics.
        
        Args:
            y_test: Actual CLTV values
            y_pred: Predicted CLTV values
            
        Returns:
            Dictionary with metrics including RMSE, MAE, R2 Score.
        """
        if y_test.shape != y_pred.shape:
            raise ValueError("y_test and y_pred must have the same shape.")
        if np.any(np.isnan(y_test)) or np.any(np.isnan(y_pred)):
            raise ValueError("Input arrays contain NaN values.")
            
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2_Score": r2
        }

class ModelEvaluator:
    """Main class for handling model evaluation."""
    
    def __init__(self, strategy: EvaluationStrategy = None):
        self.strategy = strategy or CLTVEvaluator()
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model using the provided evaluator.

        Args:
            model: The trained model
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        return self.strategy.evaluate(y_test, y_pred)

# Example usage (for testing or debugging):
if __name__ == "__main__":
    # # Example data
    # actual_cltv_values = np.array([100, 200, 300])
    # predicted_cltv_values = np.array([110, 190, 285])

    # # Initialize evaluator
    # evaluator = ModelEvaluator()

    # # Evaluate model predictions
    # metrics = evaluator.evaluate_model(actual_cltv_values, predicted_cltv_values)

    # # Print results
    # for metric_name, value in metrics.items():
    #     print(f"{metric_name}: {value:.4f}")
    pass