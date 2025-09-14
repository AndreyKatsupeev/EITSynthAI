# Use the official NVIDIA CUDA image as a parent image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies in a single RUN layer (no duplicates)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        tesseract-ocr \
        tesseract-ocr-rus \
        poppler-utils \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        libxi6 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        libglu1-mesa \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch and torchvision with CUDA 12.4 support
RUN pip3 install --no-cache-dir torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f \
    https://download.pytorch.org/whl/cu124/torch_stable.html \
    -f https://download.pytorch.org/whl/cu124/torchvision_stable.html \
    -f https://download.pytorch.org/whl/cu124/torchaudio_stable.html

# Install scikit-learn separately (already compatible)
RUN pip3 install --no-cache-dir scikit-learn

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# This will resolve numpy conflict by using numpy==1.23.2 (required by nii2dcm)
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/temp

# Copy the application code
COPY . /app/

# Expose port
EXPOSE 5001

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "kt_service.main_kt_service:app", "--host", "0.0.0.0", "--port", "5001"]
