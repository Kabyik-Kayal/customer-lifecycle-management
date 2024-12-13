import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from steps.data_ingestion_step import data_ingestion_step
from steps.handling_missing_values_step import handling_missing_values_step
from steps.dropping_columns_step import dropping_columns_step
from steps.detecting_outliers_step import detecting_outliers_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitting_step import data_splitting_step
from steps.model_building_step import model_building_step
from steps.model_evaluating_step import model_evaluating_step
from steps.data_resampling_step import data_resampling_step
from zenml import Model, pipeline


@pipeline(model=Model(name='CLTV_Prediction'))
def training_pipeline():
    """
    Defines the complete training pipeline for CLTV Prediction.
    Steps:
    1. Data ingestion
    2. Handling missing values
    3. Dropping unnecessary columns
    4. Detecting and handling outliers
    5. Feature engineering
    6. Splitting data into train and test sets
    7. Resampling the training data
    8. Model training
    9. Model evaluation
    """
    # Step 1: Data ingestion
    raw_data = data_ingestion_step(file_path='data/Online_Retail.xlsx')

    # Step 2: Drop unnecessary columns
    columns_to_drop = ["Country", "Description", "InvoiceNo", "StockCode"]
    refined_data = dropping_columns_step(raw_data, columns_to_drop)

    # Step 3: Detect and handle outliers
    outlier_free_data = detecting_outliers_step(refined_data)

    # Step 4: Feature engineering
    features_data = feature_engineering_step(outlier_free_data)
    
    # Step 5: Handle missing values
    cleaned_data = handling_missing_values_step(features_data)
    
    # Step 6: Data splitting
    train_features, test_features, train_target, test_target = data_splitting_step(cleaned_data,"CLTV")

    # Step 7: Data resampling
    train_features_resampled, train_target_resampled = data_resampling_step(train_features, train_target)

    # Step 8: Model training
    trained_model = model_building_step(train_features_resampled, train_target_resampled)

    # Step 9: Model evaluation
    evaluation_metrics = model_evaluating_step(trained_model, test_features, test_target)

    # Return evaluation metrics
    return evaluation_metrics


if __name__ == "__main__":
    # Run the pipeline
    training_pipeline()
