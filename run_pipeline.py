import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import click
from pipelines.training_pipeline import training_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Set the MLflow tracking URI
    os.environ['MLFLOW_TRACKING_URI'] = get_tracking_uri()


    # Run the pipeline
    run = training_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "Click on the CLTV_Prediction Model and set the Alias to 'champion'\n"
        "Then run the deployment pipeline to deploy the model."
    )


if __name__ == "__main__":
    main()