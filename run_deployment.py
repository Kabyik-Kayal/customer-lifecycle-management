import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipelines.deployment_pipeline import deployment_pipeline

def main():
    """
    Run the deployment pipeline.
    """
    # Run the deployment pipeline
    run = deployment_pipeline()

if __name__ == "__main__":
    # Run the deployment pipeline
    main()