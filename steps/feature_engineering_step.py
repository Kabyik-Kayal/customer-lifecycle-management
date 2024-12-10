import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from zenml import step
from src.feature_engineering import FeatureEngineer, CLTVFeatureEngineering

@step
def feature_engineering_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The dataframe containing features for engineering.

    Returns:
    pd.DataFrame: The dataframe containing engineered features.
    """

    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    cltv_engineer = FeatureEngineer(CLTVFeatureEngineering(
    customer_id_col="CustomerID",
    date_col="InvoiceDate",
    amount_col="Amount"))
    
    df_cltv = cltv_engineer.apply_feature_engineering(df)
    df_cltv = df_cltv.drop('CustomerID', axis=1)

    return df_cltv