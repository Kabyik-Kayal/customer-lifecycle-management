import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplitter:
    """
    A class for splitting data into training and testing sets.
    """

    def __init__(self, test_size=0.2, random_state=42, stratify_col=None):
        """
        Initializes the DataSplitter with parameters for the data split.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator for reproducibility.
        stratify_col (str): The column to use for stratified splitting (if any).
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_col = stratify_col

    def split_data(self, df: pd.DataFrame, target_col: str) -> tuple:
        """
        Splits the DataFrame into training and testing datasets.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column for the prediction task.

        Returns:
        tuple: Data split into (X_train, X_test, y_train, y_test).
        """
        logging.info("Splitting the data into training and testing sets.")

        # Extracting features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Determine if stratified splitting is needed
        stratify = df[self.stratify_col] if self.stratify_col else None

        # Perform the data split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )

        logging.info(f"Data split completed. Training size: {len(X_train)}, Testing size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    # # Sample data
    # data = {
    #     "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     "Age": [25, 45, 35, 32, 40, 50, 28, 33, 41, 38],
    #     "Total_Spent": [5000, 15000, 7000, 10000, 12000, 20000, 8000, 9500, 13000, 14000],
    #     "Frequency": [12, 30, 15, 25, 27, 35, 18, 21, 28, 32],
    #     "CLTV": [12000, 35000, 18000, 27000, 30000, 40000, 19000, 26000, 31000, 32000]
    # }

    # df = pd.DataFrame(data)

    # # Initialize the DataSplitter
    # splitter = DataSplitter(test_size=0.3, random_state=42)

    # # Perform the data split
    # X_train, X_test, y_train, y_test = splitter.split_data(df, target_col="CLTV")

    # print("Training features:")
    # print(X_train)
    # print("\nTraining target:")
    # print(y_train)
    # print("\nTesting features:")
    # print(X_test)
    # print("\nTesting target:")
    # print(y_test)
    pass