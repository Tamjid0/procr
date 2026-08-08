# Procr v2 - MinerU 2.5 Pro OCR Inference
# PyTorch base (matches Colab environment) + vLLM + MinerU
# Compatible with: OVH AI Deploy, RunPod, any Docker+GPU host

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VLLM_USE_V1=0

# System deps (libgl1 for Pillow/OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# === INSTALL DEPS IN EXACT COLAB ORDER ===

# Step 1: Magic-PDF + numpy pin
RUN pip install --no-cache-dir --no-build-isolation \
    magic-pdf==1.1.0 "numpy<2.0.0"

# Step 2: MinerU VLM + vLLM
RUN pip install --no-cache-dir --no-build-isolation \
    "mineru-vl-utils[vllm]==0.2.6" \
    qwen-vl-utils==0.0.8 \
    "vllm>=0.6.0"

# Step 3: Transformers + accelerate + bitsandbytes
RUN pip install --no-cache-dir --no-build-isolation \
    "transformers>=4.45.0" accelerate bitsandbytes scipy

# Step 4: Server stack
RUN pip install --no-cache-dir \
    fastapi==0.115.0 uvicorn==0.30.6 python-multipart==0.0.9 \
    Pillow==10.4.0 pydantic==2.9.2 httpx==0.27.2

# === PRE-DOWNLOAD MODEL WEIGHTS ===
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('opendatalab/MinerU2.5-Pro-2604-1.2B')"

# === COPY APP ===
COPY app/ ./app/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/diagnostic')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300"]
