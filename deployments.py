from datetime import datetime, timedelta
from prefect.deployments import Deployment, FlowScript
from prefect.orion.schemas.schedules import CronSchedule, IntervalSchedule

date_str = datetime.today().strftime("%Y-%m-%d")

Deployment(
    name="deploy-mlflow-training",
    schedule=IntervalSchedule(interval=timedelta(days=7)),  # weekly training
    flow=FlowScript(path="./main.py", name="mlflow-training"),
    parameters={},  # no parameters needed
    tags=["ml-training", "emotion-recognition"],
)

Deployment(
    name="deploy-mlflow-staging",
    schedule=CronSchedule(cron="0 9 1 * *"),  # monthly staging run
    flow=FlowScript(path="./stage.py", name="mlflow-staging"),
    parameters={
        "tracking_uri": "http://127.0.0.1:5000",
        "experiment_name": f"emotion-recognition-experiment-{date_str}",
    },
    tags=["ml-staging", "emotion-recognition"],
)
