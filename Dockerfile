FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Redirect HuggingFace cache to the network volume so model downloads
# don't fill up the small container disk (/app).
ENV HF_HOME=/runpod-volume/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface/transformers \
    HF_HUB_DISABLE_XET=1 \
    TMPDIR=/runpod-volume/tmp

WORKDIR /app

# Install system dependencies for video encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /runpod-volume/.cache/huggingface /runpod-volume/tmp || true

COPY handler.py .

CMD ["python", "-u", "handler.py"]
