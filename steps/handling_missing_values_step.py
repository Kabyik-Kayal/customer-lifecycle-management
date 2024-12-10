import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from zenml import step
from src.handle_missing_values import MissingValuesHandler, DropMissingValues

@step
def handling_missing_values_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the input dataframe by dropping rows with missing values.

    Parameters:
    df (pd.DataFrame): The dataframe to handle.
        
    Returns:
    pd.DataFrame: A dataframe with missing values handled.
    """
    handler = MissingValuesHandler(DropMissingValues())
    df_cleaned = handler.handle_missing_values(df)
    return df_cleaned