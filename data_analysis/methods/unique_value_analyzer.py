import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
class DataFrameAnalyzer(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, columns: list):
        pass

class UniqueValueAnalyzer(DataFrameAnalyzer):
    def analyze(self, df: pd.DataFrame, columns: list):
        unique_values = {}
        for column in columns:
            if column in df.columns:
                unique_values[column] = df[column].nunique()
            else:
                unique_values[column] = None
        
        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.bar(unique_values.keys(), unique_values.values())
        plt.xlabel('Columns')
        plt.ylabel('Number of Unique Values')
        plt.title('Unique Values')
        plt.show()
        return unique_values

# Example usage:
if __name__ == "__main__":
    data = {
        'A': [1, 2, 2, 3, 4],
        'B': [1, 1, 1, 1, 1],
        'C': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    analyzer = UniqueValueAnalyzer()
    columns_to_analyze = ['A', 'B', 'C', 'D']
    result = analyzer.analyze(df, columns_to_analyze)
    print(result)