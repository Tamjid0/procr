# Procr v2 - MinerU 2.5 Pro OCR Inference
# OVH AI Deploy compatible (non-root user 42420)
# Also works on RunPod, any Docker+GPU host

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# OVH AI Deploy: container runs as user 42420 (non-root)
# HOME must be /workspace for HF cache, triton, etc.
ENV HOME=/workspace
ENV HF_HOME=/workspace/hf_cache
ENV HF_HUB_CACHE=/workspace/hf_cache/hub
ENV VLLM_CACHE_ROOT=/workspace/vllm_cache
ENV VLLM_CONFIG_ROOT=/workspace/vllm_config
ENV OUTLINES_CACHE_DIR=/tmp/.outlines
ENV VLLM_USE_V1=0

# System deps (libgl1 for Pillow/OpenCV, gcc for Triton JIT)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# === WORKSPACE SETUP ===
WORKDIR /workspace
RUN mkdir -p /workspace/hf_cache /workspace/vllm_cache /workspace/vllm_config && \
    chown -R 42420:42420 /workspace

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
# Download to /workspace so non-root user can access
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('opendatalab/MinerU2.5-Pro-2604-1.2B', cache_dir='/workspace/hf_cache')" && \
    chown -R 42420:42420 /workspace/hf_cache

# === COPY APP ===
COPY --chown=42420:42420 app/ /workspace/app/

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/diagnostic')" || exit 1

USER 42420

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300"]
