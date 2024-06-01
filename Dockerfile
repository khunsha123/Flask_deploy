# Use the official Python 3.11 image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies for dlib
RUN apt-get update && \
    apt-get install -y cmake build-essential && \
    pip install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
