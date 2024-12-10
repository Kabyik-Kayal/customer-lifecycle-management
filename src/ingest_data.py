import logging
import pandas as pd
from abc import ABC, abstractmethod

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Data Ingestion Strategy
# ------------------------------------------------
# This class defines a common interface for different data ingestion strategies.
# Subclasses must implement the `ingest` method.
class DataIngestionStrategy(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a file into a DataFrame.

        Parameters:
        file_path (str): The path to the data file to ingest.

        Returns:
        pd.DataFrame: A dataframe containing the ingested data.
        """
        pass
    
# Concrete Strategy for XLSX File Ingestion
# -----------------------------------------
# This strategy handles the ingestion of data from an XLSX file.
class XLSXIngestion(DataIngestionStrategy):
    def __init__(self, sheet_name=0):
        """
        Initializes the XLSXIngestion with optional sheet name.

        Parameters:
        sheet_name (str or int): The sheet name or index to read, default is the first sheet.
        """
        self.sheet_name = sheet_name

    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingests data from an XLSX file into a DataFrame.

        Parameters:
        file_path (str): The path to the XLSX file.

        Returns:
        pd.DataFrame: A dataframe containing the ingested data.
        """
        try:
            logging.info(f"Attempting to read XLSX file: {file_path}")
            df = pd.read_excel(file_path,dtype={'InvoiceNo': str, 'StockCode': str, 'Description':str}, sheet_name=self.sheet_name)
            logging.info(f"Successfully read XLSX file: {file_path}")
            return df
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            logging.error(f"File is empty: {file_path}")
        except Exception as e:
            logging.error(f"An error occurred while reading the XLSX file: {e}")
        return pd.DataFrame()


# Context Class for Data Ingestion
# --------------------------------
# This class uses a DataIngestionStrategy to ingest data from a file.
class DataIngestor:
    def __init__(self, strategy: DataIngestionStrategy):
        """
        Initializes the DataIngestor with a specific data ingestion strategy.

        Parameters:
        strategy (DataIngestionStrategy): The strategy to be used for data ingestion.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataIngestionStrategy):
        """
        Sets a new strategy for the DataIngestor.

        Parameters:
        strategy (DataIngestionStrategy): The new strategy to be used for data ingestion.
        """
        logging.info("Switching data ingestion strategy.")
        self._strategy = strategy

    def ingest_data(self, file_path: str) -> pd.DataFrame:
        """
        Executes the data ingestion using the current strategy.

        Parameters:
        file_path (str): The path to the data file to ingest.

        Returns:
        pd.DataFrame: A dataframe containing the ingested data.
        """
        logging.info("Ingesting data using the current strategy.")
        return self._strategy.ingest(file_path)


# Example usage
if __name__ == "__main__":
    # Example file path for XLSX file
    # file_path = "../data/raw/your_data_file.xlsx"

    # XLSX Ingestion Example
    # xlsx_ingestor = DataIngestor(XLSXIngestion(sheet_name=0))
    # df = xlsx_ingestor.ingest_data(file_path)

    # Show the first few rows of the ingested DataFrame if successful
    # if not df.empty:
    #     logging.info("Displaying the first few rows of the ingested data:")
    #     print(df.head())
    pass