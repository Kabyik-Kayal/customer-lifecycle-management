from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
from typing import Union, List

import matplotlib.pyplot as plt

class UnivariateAnalyzer(ABC):
    """Abstract base class for univariate analysis"""
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame, column: str) -> None:
        """Abstract method to perform univariate analysis"""
        pass

class NumericalAnalyzer(UnivariateAnalyzer):
    """Concrete class for numerical column analysis"""
    
    def analyze(self, data: pd.DataFrame, column: str) -> None:
        if data[column].dtype not in ['int64', 'float64']:
            raise ValueError(f"Column {column} is not numerical")
        
        # Basic statistics
        stats = data[column].describe()
        print(f"\nNumerical Analysis for {column}:")
        print(stats)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        sns.histplot(data=data, x=column, ax=ax1)
        ax1.set_title(f'Histogram of {column}')
        
        # Box plot
        sns.boxplot(data=data, y=column, ax=ax2)
        ax2.set_title(f'Boxplot of {column}')
        
        plt.tight_layout()
        plt.show()

class CategoricalAnalyzer(UnivariateAnalyzer):
    """Concrete class for categorical column analysis"""
    
    def analyze(self, data: pd.DataFrame, column: str) -> None:
        if data[column].dtype not in ['object', 'category']:
            raise ValueError(f"Column {column} is not categorical")
        
        # Value counts and proportions
        value_counts = data[column].value_counts()
        proportions = data[column].value_counts(normalize=True)
        
        print(f"\nCategorical Analysis for {column}:")
        print("\nValue Counts:")
        print(value_counts)
        print("\nProportions:")
        print(proportions)
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=column)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
        plt.show()

def perform_univariate_analysis(data: pd.DataFrame, column: str) -> None:
    """Factory function to create and use appropriate analyzer"""
    try:
        if data[column].dtype in ['int64', 'float64']:
            analyzer = NumericalAnalyzer()
        else:
            analyzer = CategoricalAnalyzer()
            
        analyzer.analyze(data, column)
        
    except KeyError:
        print(f"Column {column} not found in the dataset")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    # Get user input for column
    column_name = input("Enter the column name for analysis: ")
    perform_univariate_analysis(df, column_name)