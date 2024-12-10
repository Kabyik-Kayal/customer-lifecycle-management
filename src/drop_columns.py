import logging
import pandas as pd
from typing import List

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ColumnDropper:
    """
    A class for dropping specified columns from a Pandas DataFrame.
    """

    def __init__(self, columns_to_drop: List[str]):
        """
        Initializes the ColumnDropper with a list of columns to drop.

        Parameters:
        columns_to_drop (List[str]): A list of column names to be dropped.
        """
        self.columns_to_drop = columns_to_drop

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame from which columns will be dropped.

        Returns:
        pd.DataFrame: A DataFrame with the specified columns removed.
        """
        logging.info(f"Attempting to drop columns: {self.columns_to_drop}")

        # Check if all columns to drop are present in the DataFrame
        missing_columns = [col for col in self.columns_to_drop if col not in df.columns]
        if missing_columns:
            logging.warning(f"Columns not found in DataFrame and will be ignored: {missing_columns}")

        # Drop the specified columns, ignoring errors for columns that do not exist
        df_dropped = df.drop(columns=self.columns_to_drop, errors='ignore')
        logging.info(f"Columns successfully dropped: {self.columns_to_drop}")
        return df_dropped


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # data = {
    #     "Name": ["John", "Jane", "Mary", "Mike", "Chris"],
    #     "Age": [28, 34, 29, 42, 36],
    #     "Gender": ["Male", "Female", "Female", "Male", "Male"],
    #     "Salary": [70000, 80000, 65000, 120000, 90000],
    #     "Department": ["HR", "Finance", "IT", "Management", "Sales"]
    # }
    
    # df = pd.DataFrame(data)
    
    # # Display original DataFrame
    # logging.info("Original DataFrame:")
    # print(df)

    # # Drop specific columns
    # columns_to_drop = ["Gender", "Department"]
    # dropper = ColumnDropper(columns_to_drop=columns_to_drop)
    # df_dropped = dropper.drop_columns(df)

    # # Display DataFrame after dropping columns
    # logging.info("DataFrame after dropping specified columns:")
    # print(df_dropped)
    pass