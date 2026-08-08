# Procr v2 - MinerU 2.5 Pro OCR Inference
# Install order matches colab_v2 (tested, no resolver conflicts)
# Compatible with: OVH AI Deploy, RunPod, any Docker+GPU host

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VLLM_USE_V1=0

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# === INSTALL DEPS IN EXACT COLAB ORDER (avoids resolver conflicts) ===

# Step 1: Magic-PDF + numpy pin
RUN pip install --no-cache-dir --break-system-packages \
    magic-pdf==1.1.0 "numpy<2.0.0"

# Step 2: MinerU VLM + vLLM (the GPU inference backbone)
RUN pip install --no-cache-dir --break-system-packages \
    "mineru-vl-utils[vllm]==0.2.6" \
    qwen-vl-utils==0.0.8 \
    "vllm>=0.6.0"

# Step 3: Transformers + accelerate + bitsandbytes
RUN pip install --no-cache-dir --break-system-packages \
    "transformers>=4.45.0" accelerate bitsandbytes scipy

# Step 4: Server stack
RUN pip install --no-cache-dir --break-system-packages \
    fastapi==0.115.0 uvicorn==0.30.6 python-multipart==0.0.9 \
    Pillow==10.4.0 pydantic==2.9.2 httpx==0.27.2

# === PRE-DOWNLOAD MODEL WEIGHTS ===
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('opendatalab/MinerU2.5-Pro-2604-1.2B')"

# === COPY APP ===
COPY app/ ./app/

EXPOSE 8080

# Health check (OVH AI Deploy uses this)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/diagnostic')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300"]
