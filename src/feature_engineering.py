import logging
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import numpy as np

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed

# Concrete Strategy for CLTV Feature Engineering
# --------------------------------------------
# This strategy creates specific features needed for CLTV prediction
class CLTVFeatureEngineering(FeatureEngineeringStrategy):
    def __init__(self, customer_id_col, date_col, amount_col):
        """
        Initialize with required column names for CLTV calculations.

        Parameters:
        customer_id_col (str): Name of customer ID column
        date_col (str): Name of purchase date column
        amount_col (str): Name of transaction amount column
        """
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.amount_col = amount_col

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates CLTV-specific features including:
        - Total purchase amount per customer
        - Purchase frequency
        - Average order value
        - Purchase recency
        - Customer age (in days)

        Parameters:
        df (pd.DataFrame): Input transaction data

        Returns:
        pd.DataFrame: DataFrame with CLTV features
        """
        logging.info("Creating CLTV features")
        df_transformed = df.copy()
        df_transformed[self.date_col] = pd.to_datetime(df_transformed[self.date_col])
        
        # Calculate customer-level metrics
        customer_metrics = df_transformed.groupby(self.customer_id_col).agg({
            self.date_col: ['min', 'max', 'count'],
            self.amount_col: ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = [
            self.customer_id_col, 'first_purchase', 'last_purchase', 
            'frequency', 'total_amount', 'avg_order_value'
        ]
        
        # Calculate recency,customer age and lifetime
        current_date = df_transformed[self.date_col].max()
        customer_metrics['recency'] = (current_date - customer_metrics['last_purchase']).dt.days
        customer_metrics['customer_age'] = (current_date - customer_metrics['first_purchase']).dt.days
        customer_metrics['lifetime'] = customer_metrics['customer_age'] - customer_metrics['recency']
        
        # Calculate purchase frequency per week
        customer_metrics['purchase_frequency'] = customer_metrics['frequency'] / (customer_metrics['customer_age'] / 7)

        # Calculate CLTV
        customer_metrics['CLTV'] = customer_metrics['avg_order_value'] * customer_metrics['purchase_frequency'] * customer_metrics['customer_age']

        # Drop Datetime columns
        customer_metrics.drop(columns=['first_purchase', 'last_purchase'], inplace=True)        
        

        # Dropping Null value rows if created by feature engineering
        customer_metrics = customer_metrics.dropna(subset=['CLTV'])
        logging.info("CLTV feature engineering completed")

        return customer_metrics

# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # data = {
    #     "Age": [25, 32, 47, 51, 26],
    #     "Salary": [50000, 60000, 80000, 120000, 70000],
    #     "Neighborhood": ["A", "B", "A", "C", "B"],
    # }
    # df = pd.DataFrame(data)

    # # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=["Salary"]))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)
    # logging.info("Dataframe after log transformation:")
    # print(df_log_transformed)

    # # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=["Age", "Salary"]))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)
    # logging.info("Dataframe after standard scaling:")
    # print(df_standard_scaled)

    # # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=["Age", "Salary"], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)
    # logging.info("Dataframe after Min-Max scaling:")
    # print(df_minmax_scaled)

    # # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=["Neighborhood"]))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)
    # logging.info("Dataframe after one-hot encoding:")
    # print(df_onehot_encoded)

    # # Example dataframe for CLTV feature engineering
    # data = {
    #     "CustomerID": [1, 1, 2, 2, 3, 3, 3],
    #     "TransactionDate": [
    #         "2023-01-01", "2023-02-01", "2023-01-15", "2023-03-15",
    #         "2023-01-20", "2023-02-20", "2023-03-20"
    #     ],
    #     "Amount": [100, 150, 200, 250, 300, 350, 400]
    # }
    # df = pd.DataFrame(data)

    # # CLTV Feature Engineering Example
    # cltv_engineer = FeatureEngineer(CLTVFeatureEngineering(
    #     customer_id_col="CustomerID",
    #     date_col="TransactionDate",
    #     amount_col="Amount"
    # ))
    # df_cltv_features = cltv_engineer.apply_feature_engineering(df)
    # logging.info("Dataframe with CLTV features:")
    # print(df_cltv_features)
    pass