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

# Ensure pip is up to date
RUN pip3 install --no-cache-dir --upgrade pip setuptools

# --- КЛЮЧЕВОЙ ШАГ: Устанавливаем PyTorch с CUDA ---
RUN pip3 install --no-cache-dir --force-reinstall \
    torch==2.5.0+cu124 \
    torchvision==0.20.0+cu124 \
    torchaudio==2.5.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# --- Запрещаем pip ставить torch с PyPI ---
# Создаём constraint-файл, чтобы блокировать любые попытки поставить torch
RUN echo "torch==2.5.0+cu124" > /tmp/torch-constraint.txt && \
    echo "torchvision==0.20.0+cu124" >> /tmp/torch-constraint.txt && \
    echo "torchaudio==2.5.0+cu124" >> /tmp/torch-constraint.txt

# Copy requirements.txt
COPY requirements.txt .

# --- Удаляем torch из списка зависимостей (на всякий случай) ---
RUN grep -vE "^(nvidia-|tensorrt|torch)" requirements.txt > /tmp/requirements-no-torch.txt

# --- Устанавливаем ОСТАЛЬНЫЕ пакеты с ограничением на torch ---
RUN pip3 install --no-cache-dir \
    -r /tmp/requirements-no-torch.txt \
    --constraint /tmp/torch-constraint.txt

# --- Устанавливаем NVIDIA-пакеты ---
RUN grep -E "^(nvidia-|tensorrt)" requirements.txt > /tmp/requirements-nvidia.txt && \
    pip3 install --no-cache-dir \
    --extra-index-url https://pypi.ngc.nvidia.com \
    -r /tmp/requirements-nvidia.txt

# Create runtime directories
RUN mkdir -p /app/temp

# Copy application code
COPY . /app/kt_service

# Expose port
EXPOSE 5001

# Run app
CMD ["python3", "-m", "uvicorn", "kt_service.main_kt_service:app", "--host", "0.0.0.0", "--port", "5001"]