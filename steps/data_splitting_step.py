import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Tuple
import pandas as pd
from zenml import step
from src.data_splitter import DataSplitter

@step
def data_splitting_step(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing datasets.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target column for the prediction task.

    Returns:
    tuple: Data split into (X_train, X_test, y_train, y_test).
    """
    splitter = DataSplitter(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split_data(df, target_col=target_col)
    return X_train, X_test, y_train, y_test
