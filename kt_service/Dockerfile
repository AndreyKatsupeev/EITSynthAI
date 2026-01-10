# Use the official NVIDIA CUDA image as a parent image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
WORKDIR /app/kt_service
# Install system dependencies and Python 3.10 explicitly
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


# Ensure pip is up to date and linked to Python 3.10
RUN pip3 install --no-cache-dir --upgrade pip setuptools


# Install PyTorch 2.5.0 with CUDA 12.4 (explicitly for Python 3.10)
RUN pip3 install --no-cache-dir \
torch==2.5.0+cu124 \
torchvision==0.20.0+cu124 \
torchaudio==2.5.0+cu124 \
--extra-index-url https://download.pytorch.org/whl/cu124


# Install scikit-learn (compatible with Python 3.10)
RUN pip3 install --no-cache-dir scikit-learn


# Copy and install regular requirements (excluding NVIDIA packages)
COPY requirements.txt .
RUN grep -vE "^(nvidia-|tensorrt)" requirements.txt > /tmp/requirements-common.txt && \
pip3 install --no-cache-dir -r /tmp/requirements-common.txt


# Install NVIDIA-specific packages from NGC
RUN grep -E "^(nvidia-|tensorrt)" requirements.txt > /tmp/requirements-nvidia.txt && \
pip3 install --no-cache-dir \
--extra-index-url https://pypi.ngc.nvidia.com \
-r /tmp/requirements-nvidia.txt


# Create runtime directories
RUN mkdir -p /app/temp


# Copy application code into the kt_service package directory
COPY . /app/kt_service


# Expose the service port
EXPOSE 5001


# Run the FastAPI app using Python 3.10
CMD ["python3", "-m", "uvicorn", "kt_service.main_kt_service:app", "--host", "0.0.0.0", "--port", "5001"]