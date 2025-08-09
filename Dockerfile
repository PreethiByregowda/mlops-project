FROM python:3.9.13-slim

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock for dependencies
COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies system-wide
RUN pip install pipenv && pipenv install --system --deploy

# Copy all source code
COPY . .

# (Optional) expose port if needed by your app
# EXPOSE 9696

# Default command: run your Prefect flow script
CMD ["python", "main.py"]
