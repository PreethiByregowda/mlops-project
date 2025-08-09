from main import mlflow_training
from stage import mlflow_staging
from datetime import timedelta, datetime

date_str = datetime.today().strftime("%Y-%m-%d")

# Deploy training flow (no schedule)
mlflow_training.deploy(
    name="mlflow-training-deployment",
    work_pool_name="docker-pool",
    image="preethibyregowda/mlops-project:latest",
    push=False,  # 👈 Don't try to push or build a Docker image
    build=False,
    tags=["training", "ml"],
)

# Deploy staging flow (no schedule)
mlflow_staging.deploy(
    name="mlflow-staging-deployment",
    work_pool_name="docker-pool",
    image="preethibyregowda/mlops-project:latest",
    push=False,  # 👈 Don't try to push or build a Docker image
    build=False,
    parameters={
    "tracking_uri": "http://172.20.255.182:5000",
    "experiment_name": f"emotion-recognition-experiment-{date_str}",
    },
    tags=["staging", "ml"],
)
