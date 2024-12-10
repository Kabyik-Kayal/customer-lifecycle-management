import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.ingest_data import DataIngestor, XLSXIngestion
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """
    Ingests data from an XLSX file into a DataFrame.

    Parameters:
    file_path (str): The path to the XLSX file.

    Returns:
    pd.DataFrame: A dataframe containing the ingested data.
    """
    # Initialize the DataIngestor with an XLSXIngestion strategy
    ingestor = DataIngestor(XLSXIngestion())
    # Ingest data from the specified file
    df = ingestor.ingest_data(file_path)
    return df