from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.utils import resample


class BaseResampler(ABC):
    """Abstract base class for resampling strategies."""
    
    @abstractmethod
    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Abstract method to perform resampling."""
        pass

class BootstrapResampler(BaseResampler):
    """
    Bootstrap resampling implementation for creating multiple samples with replacement.
    
    Attributes:
        n_samples (int): Number of bootstrap samples to generate
        sample_size (float): Size of each bootstrap sample as a fraction of original data
        random_state (Optional[int]): Random seed for reproducibility
    """
    
    def __init__(self, n_samples: int = 100, sample_size: float = 1.0, random_state: Optional[int] = None):
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.random_state = random_state
        np.random.seed(random_state)
        
    def resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform bootstrap resampling on the input data.
        
        Args:
            X (pd.DataFrame): Feature matrix as a pandas DataFrame
            y (pd.Series): Target variable as a pandas Series
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: A single resampled feature matrix and target variable
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X must be a pandas DataFrame and y must be a pandas Series")
            
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
            
        # Number of samples to draw in each bootstrap sample (based on sample_size)
        n_samples = int(len(X) * self.sample_size)
        
        X_bootstraps = []
        y_bootstraps = []
        
        for _ in range(self.n_samples):
            # Perform bootstrap resampling with replacement
            X_boot, y_boot = resample(X, y,
                                      n_samples=n_samples,
                                      replace=True,
                                      random_state=np.random.randint(0, 10000))
            X_bootstraps.append(X_boot)
            y_bootstraps.append(y_boot)
        
        # Concatenate all bootstrap samples to create the final resampled DataFrame and Series
        X_resampled = pd.concat(X_bootstraps, ignore_index=True)  # Ensure it's a DataFrame
        y_resampled = pd.concat(y_bootstraps, ignore_index=True)  # Ensure it's a Series
        
        return X_resampled, y_resampled
    
    def get_params(self) -> dict:
        """
        Get the parameters of the resampler.
        
        Returns:
            dict: Dictionary containing the resampler parameters
        """
        return {
            'n_samples': self.n_samples,
            'sample_size': self.sample_size,
            'random_state': self.random_state
        }

if __name__ == "__main__":
    # # Example usage:
    # X_train = pd.DataFrame({
    #     'feature1': [1, 2, 3, 4, 5],
    #     'feature2': [5, 4, 3, 2, 1]
    # })
    # y_train = pd.Series([0, 1, 0, 1, 0])
    
    # # Initialize bootstrap resampler
    # bootstrapper = BootstrapResampler(n_samples=100, sample_size=1.0, random_state=42)
    
    # # Perform resampling
    # X_resampled, y_resampled = bootstrapper.resample(X_train, y_train)
    
    # # Output the final resampled X and y
    # print("Resampled X (DataFrame):")
    # print(X_resampled)
    # print("Resampled y (Series):")
    # print(y_resampled)

    pass