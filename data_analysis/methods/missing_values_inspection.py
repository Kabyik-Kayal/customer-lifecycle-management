from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

class MissingValuesAnalyzer(ABC):
    """Abstract base class for missing values analysis"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
    
    @abstractmethod
    def analyze(self):
        """Abstract method that must be implemented by concrete classes"""
        pass

class MissingValuesStatistics(MissingValuesAnalyzer):
    """Concrete class for statistical analysis of missing values"""
    
    def analyze(self) -> dict:
        """Analyze missing values and return statistics"""
        stats = {
            'total_missing': self.df.isnull().sum().sum(),
            'missing_by_column': self.df.isnull().sum(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'columns_with_missing': self.df.columns[self.df.isnull().any()].tolist()
        }
        return stats

class MissingValuesVisualizer(MissingValuesAnalyzer):
    """Concrete class for visualizing missing values"""
    
    def analyze(self):
        """Create visualizations for missing values"""
        plt.figure(figsize=(10, 6))
        
        # Heatmap of missing values
        sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Bar plot of missing value percentages
        plt.figure(figsize=(10, 6))
        missing_percentages = (self.df.isnull().sum() / len(self.df)) * 100
        missing_percentages.plot(kind='bar')
        plt.title('Percentage of Missing Values by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing Values (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def analyze_missing_values(dataframe: pd.DataFrame):
    """Helper function to perform both statistical and visual analysis"""
    
    # Statistical analysis
    stats_analyzer = MissingValuesStatistics(dataframe)
    statistics = stats_analyzer.analyze()
    
    print("Missing Values Statistics:")
    print(f"Total missing values: {statistics['total_missing']}")
    print("\nMissing values by column:")
    print(statistics['missing_by_column'])
    print("\nMissing values percentage:")
    print(statistics['missing_percentage'])
    
    # Visual analysis
    viz_analyzer = MissingValuesVisualizer(dataframe)
    viz_analyzer.analyze()

# Example usage:
if __name__ == "__main__":
    # Sample DataFrame
    df = pd.DataFrame({
        'A': [1, None, 3, None, 5],
        'B': [None, 2, None, 4, 5],
        'C': [1, 2, None, 4, 5]
    })
    
    analyze_missing_values(df)