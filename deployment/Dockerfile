# Use the official Python image from the Docker Hub
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# For opencv-python
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 

# For opencv-python and necessary packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3 \
    python3-pip

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy

# Create and set the working directory
WORKDIR /src

# Copy the requirements file
COPY requirements.txt /src/

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY . /src/

ENV PYTHONPATH="/src"