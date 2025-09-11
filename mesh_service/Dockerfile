# Use the official NVIDIA CUDA image as a parent image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev \
    build-essential \
    libxcursor1 \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libxft2 \
    libxinerama1\
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglu1-mesa \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


RUN mkdir -p /app/temp /app/logs /app/weights


COPY . /app/


EXPOSE 5003


CMD ["uvicorn", "mesh_service.main_mesh_service:app", "--host", "0.0.0.0", "--port", "5003"]
