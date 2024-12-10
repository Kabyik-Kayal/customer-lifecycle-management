import logging
import pandas as pd
from abc import ABC, abstractmethod

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Missing Values Handling Strategy
# --------------------------------------------------------
# This class defines a common interface for different missing values handling strategies.
# Subclasses must implement the `handle` method.
class MissingValuesHandlerStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle rows with missing values in a specified column.

        Parameters:
        df (pd.DataFrame): The dataframe to handle.
        
        Returns:
        pd.DataFrame: A dataframe with the missing values handled.
        """
        pass


# Concrete Strategy to Drop Rows with Missing Values
# --------------------------------------------------
# This strategy removes rows where a specified column has missing values.
class DropMissingValues(MissingValuesHandlerStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows with missing values in the specified column.

        Parameters:
        df (pd.DataFrame): The dataframe to handle.
        
        Returns:
        pd.DataFrame: A dataframe with rows containing missing values in the column removed.
        """
        logging.info(f"Dropping rows which has missing values.")
        df_cleaned = df.dropna()
        logging.info("Rows with missing values dropped successfully.")
        return df_cleaned


# Concrete Strategy to Fill Missing Values with a Specified Value
# ---------------------------------------------------------------
# This strategy fills missing values in a specified column with a user-defined value.
class FillMissingValues(MissingValuesHandlerStrategy):
    def __init__(self, fill_value):
        """
        Initializes the FillMissingValues strategy with a specified value to fill.

        Parameters:
        fill_value (Any): The value to use for filling missing entries.
        """
        self.fill_value = fill_value
        def handle(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Fills missing values in the dataframe with the specified fill value.

            Parameters:
            df (pd.DataFrame): The dataframe to handle.
            
            Returns:
            pd.DataFrame: A dataframe with missing values filled with the specified value.
            """
            logging.info(f"Filling missing values with {self.fill_value}.")
            df_filled = df.fillna(self.fill_value)
            logging.info("Missing values filled successfully.")
            return df_filled



# Context Class for Handling Missing Values
# -----------------------------------------
# This class uses a MissingValuesHandlerStrategy to handle rows with missing values in a dataset.
class MissingValuesHandler:
    def __init__(self, strategy: MissingValuesHandlerStrategy):
        """
        Initializes the MissingValuesHandler with a specific strategy for handling missing values.

        Parameters:
        strategy (MissingValuesHandlerStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValuesHandlerStrategy):
        """
        Sets a new strategy for the MissingValuesHandler.

        Parameters:
        strategy (MissingValuesHandlerStrategy): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing values handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the handling of missing values using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to handle.

        Returns:
        pd.DataFrame: A dataframe with the missing values handled according to the strategy.
        """
        logging.info(f"Handling missing values")
        return self._strategy.handle(df)


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # data = {
    #     "Name": ["Alice", "Bob", "Charlie", None, "Eve"],
    #     "Age": [25, None, 30, 35, 40],
    #     "Salary": [50000, 60000, None, 70000, 80000],
    # }
    # df = pd.DataFrame(data)

    # # Drop Missing Values Example
    # drop_handler = MissingValuesHandler(DropMissingValues())
    # df_dropped = drop_handler.handle_missing_values(df)

    # logging.info("Dataframe after dropping rows with missing values:")
    # print(df_dropped)

    # # Fill Missing Values Example
    # fill_handler = MissingValuesHandler(FillMissingValues(fill_value=0))
    # df_filled = fill_handler.handle_missing_values(df)

    # logging.info("Dataframe after filling missing values with 0:")
    # print(df_filled)
    pass