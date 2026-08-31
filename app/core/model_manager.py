import os
# V0 engine + CUDA graphs: 43% faster on L4 than V1+eager (3.8s vs 6.7s per page)
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_CPU_OFFLOAD_GB"] = "0"

import logging

logger = logging.getLogger("procr")

# OCR_MODE controls which engine is loaded. 'mineru' = GPU MinerU (scale), 'vlm' = serverless CF+Gemini (credit burn).
OCR_MODE = os.getenv("OCR_MODE", "mineru").lower()

def _get_gpu_config():
    """Auto-detect GPU and return optimal vLLM config."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"🖥️  GPU detected: {gpu_name} ({vram_gb:.0f} GB VRAM)")

    if vram_gb >= 40:
        return {"gpu_memory_utilization": 0.95, "max_num_seqs": 64, "max_model_len": 4096}
    elif vram_gb >= 20:
        # L4 (24GB) — max_model_len=4096 to prevent truncation on complex pages
        return {"gpu_memory_utilization": 0.92, "max_num_seqs": 48, "max_model_len": 4096}
    elif vram_gb >= 14:
        return {"gpu_memory_utilization": 0.90, "max_num_seqs": 24, "max_model_len": 4096}
    else:
        raise RuntimeError(f"GPU VRAM too low: {vram_gb:.0f} GB (need ≥14 GB)")

class ModelManager:
    _instance = None
    _client = None
    _is_ready = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def initialize_models(self):
        """Eagerly load OCR engine based on OCR_MODE."""
        if self._is_ready:
            return

        # ── VLM mode: no GPU, no MinerU load ─────────────────────────
        if OCR_MODE == "vlm":
            logger.info("🚀 OCR_MODE=vlm — initializing serverless VLM (CF Qwen + Gemini failover)")
            try:
                from app.services.vlm_provider import VlmClient
                self._client = VlmClient()
                self._is_ready = True
                logger.info("🌟 Procr Model Manager READY (VLM mode, no GPU)")
                return
            except Exception as e:
                logger.error(f"❌ Failed to initialize VLM client: {str(e)}")
                raise e

        # ── MinerU mode: GPU path (original) ──────────────────────────
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        model_path = "opendatalab/MinerU2.5-Pro-2604-1.2B"
        # Check if model is pre-downloaded locally (OVH cache)
        import os
        local_model = "/workspace/hf_cache/hub/models--opendatalab--MinerU2.5-Pro-2604-1.2B"
        if os.path.exists(local_model):
            # Find the snapshot directory
            snapshots_dir = os.path.join(local_model, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    model_path = os.path.join(snapshots_dir, sorted(snapshots)[0])
                    logger.info(f"📦 Using pre-downloaded model: {model_path}")
        gpu_config = _get_gpu_config()
        logger.info(f"🚀 Initializing Procr v2... Model: {model_path}")
        logger.info(f"⚙️  vLLM config: {gpu_config}")

        try:
            import vllm
            from mineru_vl_utils import MinerUClient

            tuned_engine = vllm.LLM(
                model=model_path,
                gpu_memory_utilization=gpu_config["gpu_memory_utilization"],
                max_num_seqs=gpu_config["max_num_seqs"],
                max_model_len=gpu_config["max_model_len"],
                enable_chunked_prefill=True,
                trust_remote_code=True,
                dtype="bfloat16"
            )

            self._client = MinerUClient(
                backend="vllm-engine",
                vllm_llm=tuned_engine,
                image_analysis=True
            )

            # Warm up VLM kernels
            logger.info("🔥 Warming up VLM kernels...")
            try:
                from PIL import Image
                dummy_img = Image.new('RGB', (64, 64), color='white')
                self._client.two_step_extract(dummy_img)
            except Exception as e:
                logger.warning(f"Warmup skipped: {e}")

            self._is_ready = True
            logger.info("🌟 Procr Model Manager is READY (MinerU mode)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize MinerU model: {str(e)}")
            raise e

    def get_client(self):  # type: ignore[no-untyped-def]
        if not self._is_ready:
            self.initialize_models()
        return self._client

    def get_status(self):
        # Import torch lazily — not available in VLM CPU mode
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            vram = f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB" if has_cuda else "N/A"
            device = "cuda" if has_cuda else "cpu"
        except ImportError:
            has_cuda = False
            vram = "N/A (VLM mode)"
            device = "cpu"
        return {
            "ready": self._is_ready,
            "mode": OCR_MODE,
            "device": device,
            "vram_allocated": vram
        }

model_manager = ModelManager()
