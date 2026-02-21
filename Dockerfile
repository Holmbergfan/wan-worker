# Base image provides Python 3.10 + CUDA 12.1 + cuDNN
FROM runpod/base:0.6.1-cuda12.1.0

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY handler.py .
COPY scripts/ scripts/

# Path where model weights are expected on the network volume
ENV MODEL_PATH=/runpod-volume/models

# Hugging Face cache inside the network volume (avoids re-downloads)
ENV HF_HOME=/runpod-volume/.cache/huggingface

CMD ["python", "-u", "handler.py"]
