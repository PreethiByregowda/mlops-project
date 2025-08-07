LOCAL_TAG := $(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME := emotion-model:${LOCAL_TAG}

# Run unit tests
test:
	pytest tests/

# Format and organize code
quality_checks:
	isort .
	black .

# Build and run the Docker container for the web service
build_webservice: quality_checks
	docker build -t ${LOCAL_IMAGE_NAME} .
	docker run --env-file .env -d -it \
		--rm -v $(pwd):/app \
		-p 9696:9696 ${LOCAL_IMAGE_NAME}
	python test.py

# Optional integration testing if applicable
integration_test: build_webservice
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration-test/run.sh

# Publish image to container registry (if configured)
publish: build_webservice integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

# Set up local dev environment
setup:
	pipenv install --dev
