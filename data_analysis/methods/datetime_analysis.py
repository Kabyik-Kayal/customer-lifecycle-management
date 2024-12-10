from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

class DateTimeAnalyzer(ABC):
    """Abstract base class for datetime analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    @abstractmethod
    def process_datetime(self, datetime_column: str) -> pd.DataFrame:
        """Process datetime column and return transformed dataframe"""
        pass
    
    @abstractmethod
    def extract_time_features(self, datetime_column: str) -> Dict[str, Any]:
        """Extract time-based features from datetime column"""
        pass
    
    @abstractmethod
    def visualize_time_patterns(self, datetime_column: str) -> None:
        """Visualize time-based patterns"""
        pass

class BasicDateTimeAnalyzer(DateTimeAnalyzer):
    """Concrete implementation of DateTimeAnalyzer"""
    
    def process_datetime(self, datetime_column: str) -> pd.DataFrame:
        """Process datetime column by converting to datetime type"""
        self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])
        return self.df
    
    def extract_time_features(self, datetime_column: str) -> Dict[str, Any]:
        """Extract basic time features from datetime column"""
        dt_series = pd.to_datetime(self.df[datetime_column])
        
        features = {
            'year': dt_series.dt.year,
            'month': dt_series.dt.month,
            'day': dt_series.dt.day,
            'hour': dt_series.dt.hour,
            'dayofweek': dt_series.dt.dayofweek,
            'quarter': dt_series.dt.quarter
        }
        
        return features
    
    def visualize_time_patterns(self, datetime_column: str) -> None:
        """Create visualizations for time-based patterns"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Distribution by hour
        plt.subplot(2, 2, 1)
        sns.histplot(data=self.df, x=self.df[datetime_column].dt.hour)
        plt.title('Distribution by Hour')
        plt.xlabel('Hour (24-hour format)')
        plt.ylabel('Count')
        
        # Plot 2: Distribution by day of week
        plt.subplot(2, 2, 2)
        sns.histplot(data=self.df, x=self.df[datetime_column].dt.dayofweek)
        plt.title('Distribution by Day of Week')
        plt.xlabel('Day (0=Monday to 6=Sunday)')
        plt.ylabel('Count')
        
        # Plot 3: Distribution by month
        plt.subplot(2, 2, 3)
        sns.histplot(data=self.df, x=self.df[datetime_column].dt.month)
        plt.title('Distribution by Month')
        plt.xlabel('Month (1-12)')
        plt.ylabel('Count')
        
        # Plot 4: Distribution by quarter
        plt.subplot(2, 2, 4)
        sns.histplot(data=self.df, x=self.df[datetime_column].dt.quarter)
        plt.title('Distribution by Quarter')
        plt.xlabel('Quarter (1-4)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()

    def plot_time_series(self, datetime_column: str, value_column: Optional[str] = None) -> None:
        """Create a line graph showing temporal patterns"""
        plt.figure(figsize=(15, 6))
        
        if value_column:
            self.df.set_index(datetime_column)[value_column].plot(kind='line')
            plt.ylabel(value_column)
        else:
            self.df.groupby(datetime_column).size().plot(kind='line')
            plt.ylabel('Count')
            
        plt.title('Time Series Analysis')
        plt.xlabel('Date')
        plt.grid(True)
        plt.show()
# Example usage
if __name__ == "__main__":
    # Create sample data with more entries for better visualization
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    data = {
        'order_date': dates,
        'customer_id': range(len(dates)),
        'order_value': [100 + i % 50 for i in range(len(dates))]  # Adding order values for time series
    }
    df = pd.DataFrame(data)

    # Initialize analyzer
    analyzer = BasicDateTimeAnalyzer(df)

    # Process datetime
    df_processed = analyzer.process_datetime('order_date')

    # Extract time features
    time_features = analyzer.extract_time_features('order_date')

    # Add features to dataframe
    for feature_name, feature_values in time_features.items():
        df_processed[feature_name] = feature_values

    # Create visualizations
    analyzer.visualize_time_patterns('order_date')
    
    # Plot time series with order values
    analyzer.plot_time_series('order_date', 'order_value')