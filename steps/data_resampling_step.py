import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Tuple
import numpy as np
import pandas as pd
from zenml import step
from src.resampling import BootstrapResampler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step()
def data_resampling_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Resamples the training data to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Tuple containing resampled features and labels
    """
    logger.info(f"Starting data resampling. Input shape: X={X_train.shape}, y={y_train.shape}")
    
    bootstrapper = BootstrapResampler(n_samples=100, sample_size=1.0, random_state=42)
    X_bootstraps, y_bootstraps = bootstrapper.resample(X_train, y_train)
    
    logger.info(f"Completed resampling.")
    logger.info(f"Generated {len(X_bootstraps)} bootstrap samples")

    return X_bootstraps, y_bootstraps