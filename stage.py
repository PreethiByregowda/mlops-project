import os
import argparse
from datetime import datetime

import mlflow
from prefect import flow, task, get_run_logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect.task_runners import SequentialTaskRunner


@task(name="Register and stage best emotion model")
def stage_model(tracking_uri, experiment_name):
    """Register and stage the best emotion recognition model."""
    logger = get_run_logger()

    logger.info("Connecting to MLflow tracking server")
    client = MlflowClient(tracking_uri=tracking_uri)

    # Retrieve top-performing runs
    logger.info(f"Fetching best runs from experiment '{experiment_name}'")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.mae_valid ASC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No valid runs found in the experiment.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    experiment_id = best_run.info.experiment_id

    # Register model
    model_name = f"EmotionRecognitionModel-{run_id}"
    logger.info(f"Registering model as: {model_name}")
    try:
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_name,
        )
    except Exception:
        client.create_registered_model(model_name)
        registered_model = client.create_model_version(
            name=model_name,
            source=f"s3://mlflow-models-artifact-store-cmd/{experiment_id}/{run_id}/artifacts/model",
            run_id=run_id,
        )

    # Transition to staging
    logger.info("Transitioning model to 'Staging'")
    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage="Staging",
    )

    # Add description
    logger.info("Updating model version description")
    client.update_model_version(
        name=model_name,
        version=registered_model.version,
        description=f"[{datetime.now()}] Model transitioned to Staging from experiment '{experiment_name}'.",
    )


@flow(name="mlflow-staging", task_runner=SequentialTaskRunner())
def main(tracking_uri, experiment_name):
    stage_model(tracking_uri=tracking_uri, experiment_name=experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", help="MLflow tracking URI.")
    parser.add_argument("--experiment_name", help="MLflow experiment name.")
    args = parser.parse_args()

    main(tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)
