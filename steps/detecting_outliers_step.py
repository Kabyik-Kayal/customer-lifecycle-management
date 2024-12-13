import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from zenml import step
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection

@step
def detecting_outliers_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in the given DataFrame using the Z-score method.

    Parameters:
    data (pd.DataFrame): The dataframe containing features for outlier detection.

    Returns:
    pd.DataFrame: A boolean dataframe indicating where outliers are located.
    """
    # Create the outlier detector with the Z-score strategy
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # Detect outliers using the Z-score method
    outliers = outlier_detector.detect_outliers(df)
    df_cleaned = outlier_detector.handle_outliers(df, method="remove")

    return df_cleaned