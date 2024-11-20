# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# For opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Create and set the working directory
WORKDIR /src

# Copy the requirements file
COPY requirements.txt /src/

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY . /src/

ENV PYTHONPATH="/src"