FROM python:3.9.13-slim

# Install pip and pipenv
RUN pip install --upgrade pip
RUN pip install pipenv

# Set working directory
WORKDIR /app

# Copy Pipenv files first for better layer caching
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --system --deploy

# Copy the rest of the source code into the container
COPY . .

# Expose the API port
EXPOSE 9696

# Run FastAPI app with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app.main:app"]
