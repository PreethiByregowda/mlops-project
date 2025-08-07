# 🚀 Emotion Recognition Web Service

This web service wraps a machine learning model that predicts emotions from input text. It's containerized with Docker and easy to run locally or deploy anywhere.

---

## 📦 Build the Docker Image

```bash
docker build -t emotion-recognition-service:v1
```

## 🏃 Run the Docker Container

```bash
docker run -it --rm -v $(pwd):/app -p 9696:9696 emotion-recognition-service:v1
```

## 🧰 Run in Detached Mode
```bash
docker run -d -it --rm -v $(pwd):/app -p 9696:9696 emotion-recognition-service:v1
```

## 🌐 Access the Service
Open your browser and go to [http://localhost:9696](http://localhost:9696) to access the web service.

## 📝 API Documentation
The API documentation is available at [http://localhost:9696/docs](http://localhost:9696/docs).

## 🧪 Test the Service
You can test the service using `curl` or any HTTP client. Here's an example using `curl`:
```bash
curl -X POST "http://localhost:9696/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'
``` 

## 📅 Build + Run + Test
```bash
make build_webservice
```

## 📂 Project Structure
```
web_service/
├── Dockerfile
├── Makefile
├── Pipfile
├── Pipfile.lock
├── README.md
├── app/
│   └── main.py               # FastAPI/Flask entry point
├── emotion_model/
│   ├── predict.py            # Inference logic
│   └── utils.py              # Input preprocessing
├── test.py                   # Test client for prediction
```

## 🔒 Environment
```
No environment variables or secrets are required for running the web service locally. The model and its dependencies are included in the Docker image.