FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app/kt_service

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential git \
    tesseract-ocr tesseract-ocr-rus poppler-utils \
    libxcursor1 libxft2 libxinerama1 libxi6 libsm6 \
    libxext6 libxrender-dev libgl1 libglu1-mesa libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools

# Установка CPU PyTorch
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.5.0+cpu torchvision==0.20.0+cpu torchaudio==2.5.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Установка без torch и nvidia (grep исключает их)
RUN grep -vE "(nvidia-tensorrt|torch|onnxruntime-gpu|nvidia-)" requirements.txt > /tmp/requirements-cpu.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cpu.txt
# CPU ONNX Runtime
RUN pip install --no-cache-dir onnxruntime

RUN mkdir -p /app/temp

COPY . /app/kt_service

EXPOSE 5001

CMD ["python3", "-m", "uvicorn", "kt_service.main_kt_service:app", "--host", "0.0.0.0", "--port", "5001"]
