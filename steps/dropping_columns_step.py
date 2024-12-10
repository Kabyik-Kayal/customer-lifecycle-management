import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from zenml import step
from src.drop_columns import ColumnDropper

@step
def dropping_columns_step(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Step to drop specified columns from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_drop (list): List of columns to drop. 
    
    # We are dropping these columns["Country","Description","InvoiceNo","StockCode"]
    
    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    dropper = ColumnDropper(columns_to_drop=columns_to_drop)
    df_dropped = dropper.drop_columns(df)
    return df_dropped