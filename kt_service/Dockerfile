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
    tesseract-ocr \
    tesseract-ocr-rus \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglu1-mesa \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch first
RUN pip3 install --no-cache-dir torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f https://download.pytorch.org/whl/cu124/torch_stable.html -f https://download.pytorch.org/whl/cu124/torchvision_stable.html -f https://download.pytorch.org/whl/cu124/torchaudio_stable.html scikit-learn

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/temp

# Copy the application
COPY . /app/

# Expose the port
EXPOSE 5000

# Command to run the application
CMD ["uvicorn", "ocr_miratorg:app", "--host", "0.0.0.0", "--port", "5000"] 