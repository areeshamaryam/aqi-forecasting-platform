"""
Entry point for the DAILY automated retraining pipeline.

Run with:
    python -m ml_pipeline.run_daily_pipeline

This is the single command the GitHub Actions daily workflow
calls. It does two things in order:
    1. Rebuild train/validation/test datasets from the latest
       data in the Hopsworks feature store
    2. Retrain and re-register the production Ridge model
       (creates a new version in the Hopsworks Model Registry)
"""

from .training.prepare_training_data import main as prepare_training_data
from .model_registry.regsiter_model import main as register_model


def main():

    print("#" * 60)
    print("# DAILY PIPELINE START")
    print("#" * 60)

    # --------------------------------------------------------
    # Step 1: Rebuild training data from the latest features
    # --------------------------------------------------------

    prepare_training_data()

    # --------------------------------------------------------
    # Step 2: Retrain and re-register the model
    # --------------------------------------------------------

    print("\n" + "#" * 60)
    print("# RETRAINING MODEL")
    print("#" * 60)

    register_model()

    print("\n" + "#" * 60)
    print("# DAILY PIPELINE COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()