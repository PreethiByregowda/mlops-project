# 🧠 Emotion Recognition - MLOps Project (End-to-End)

This is a full end-to-end MLOps project that builds, tracks, orchestrates, and deploys a machine learning model to recognize **emotions from text** using FastAPI and Docker.


## 💡 Problem

This project demonstrates a robust MLOps pipeline that:

- Trains a text-based emotion recognition model.
- Tracks experiments and models with [MLflow](https://mlflow.org).
- Orchestrates workflows using [Prefect](https://orion-docs.prefect.io/).
- Deploys a trained model as a web service using [FastAPI](https://fastapi.tiangolo.com/) and Docker.

## 📦 Dataset

We use a labeled emotion dataset for training and validation. You can preprocess the dataset using the provided `goemotion_dataset.ipynb` notebook.

## 🚀 Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-recognition-mlops.git
cd emotion-recognition-mlops
```

### 2. Install dependencies
```bash
make setup
```

### 3. Add current directory to Python path
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### 4. Set up environment variables
No environment variables are required for local development. 

### 5. Train & Register the Model
#### Train the model locally
```bash
python main.py
```
#### Register & Stage the Best Model
```bash
python stage.py --tracking_uri http://127.0.0.1:5000 --experiment_name your_experiment_name
```

### 6. Orchestrate with Prefect
#### Create deployments
```bash
prefect deployment create deployments.py
```
#### Create work queues
```bash
prefect work-queue create -t "ml-training" ml-training-queue
prefect work-queue create -t "ml-staging" ml-staging-queue
```
#### Run Prefect Orion server
```bash
prefect orion start
```
Access the Prefect UI at: http://localhost:4200
#### Trigger scheduled deployments
```bash
prefect deployment run mlflow-training/deploy-mlflow-training
prefect deployment run mlflow-staging/deploy-mlflow-staging
```

### 7. Deploy the Web Service
Navigate to the web_service/ directory and build the Docker image:
```bash
cd web_service
make build_webservice
```
This will:
- Build the Docker image
- Run code quality checks
- Expose the service at http://localhost:9696

### 8. Run the test client
You can run the test client to verify the model predictions:
```bash
python test.py
```
## 🖥️ User Interfaces

### FastAPI Web Service UI

Interact with the deployed model via the FastAPI web interface:

<img src="images/FastApi UI.png" alt="FastAPI UI" width="600"/>

Example prediction request screen:

<img src="images/FastApi UI 2.png" alt="FastAPI Prediction Example" width="600"/>

---

### MLflow Experiment Tracking UI

Track experiments, compare runs, and manage models with MLflow:

<img src="images/MLflow UI.png" alt="MLflow UI" width="600"/>

---

### Prefect Orion Workflow Orchestration UI

Monitor and manage your workflows using Prefect Orion:

<img src="images/Prefect UI 1.png" alt="Prefect Orion Dashboard" width="600"/>

Detailed view of deployments and task runs:

<img src="images/Prefect UI 2.png" alt="Prefect Deployment Details" width="600"/>

---

### Model Management UI

Visualize model registration and lifecycle stages:

<img src="images/Model UI.png" alt="Model Management UI" width="600"/>

## 📂 Project Structure
```
.
├── emotion_dataset_load.ipynb       # Data loading and preprocessing
├── main.py                          # Training pipeline
├── stage.py                         # Model registration/staging
├── deployments.py                   # Prefect deployment config
├── mlflow.db                        # MLflow local backend
├── model/
│   └── model-rgr.pkl                # Saved model
├── utils/
│   └── prepare.py                   # Feature engineering logic
├── web_service/
│   ├── Dockerfile
│   ├── Makefile
│   ├── Pipfile / Pipfile.lock
│   ├── app/
│   │   └── main.py                  # FastAPI app
│   ├── emotion_model/
│   │   ├── predict.py               # Model inference
│   │   └── utils.py                 # Text preprocessing
│   └── test.py                      # Request test script
```

## 🚀 Run FastAPI Web Service Locally

If you prefer to run the FastAPI app without Docker, follow these steps::

```bash
pipenv install
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
uvicorn main:app --host 127.0.0.1 --port 8000
```

| Component               | UI URL (Default)                    | Notes                                                  | Docker Image / Location                                 |
| ----------------------- | ----------------------------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| **MLflow Tracking UI**  | `http://localhost:5000`             | MLflow UI for experiment tracking                      | Runs locally or in a container (mlflow image or custom) |
| **Prefect Orion UI**    | `http://localhost:4200`             | Prefect’s orchestration UI                             | Runs locally or in Prefect agent container              |
| **FastAPI Web Service** | `http://localhost:8000` | Your deployed model API & Swagger UI (auto at `/docs`) | Custom web\_service Docker image (your FastAPI app)     |
