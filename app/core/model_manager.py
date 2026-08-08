import os
# Force stable V0 engine (V1 has issues on consumer GPUs)
os.environ["VLLM_USE_V1"] = "0"
# Optimize CPU-bound startup
os.environ["VLLM_CPU_OFFLOAD_GB"] = "0"

import torch
import logging
from mineru_vl_utils import MinerUClient

logger = logging.getLogger("procr")

def _get_gpu_config():
    """Auto-detect GPU and return optimal vLLM config."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"🖥️  GPU detected: {gpu_name} ({vram_gb:.0f} GB VRAM)")

    if vram_gb >= 40:
        # L40S, A100, H100 — generous VRAM
        return {"gpu_memory_utilization": 0.95, "max_num_seqs": 32, "max_model_len": 8192}
    elif vram_gb >= 20:
        # L4 (24GB) — good headroom
        return {"gpu_memory_utilization": 0.90, "max_num_seqs": 24, "max_model_len": 8192}
    elif vram_gb >= 14:
        # T4 (16GB) — tight, conservative settings
        return {"gpu_memory_utilization": 0.90, "max_num_seqs": 16, "max_model_len": 8192}
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
        """Eagerly load the MinerU 2.5 Pro model into VRAM."""
        if self._is_ready:
            return

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        model_path = "opendatalab/MinerU2.5-Pro-2604-1.2B"
        gpu_config = _get_gpu_config()
        logger.info(f"🚀 Initializing Procr v2... Model: {model_path}")
        logger.info(f"⚙️  vLLM config: {gpu_config}")
        
        try:
            import vllm
            
            tuned_engine = vllm.LLM(
                model=model_path,
                gpu_memory_utilization=gpu_config["gpu_memory_utilization"],
                max_num_seqs=gpu_config["max_num_seqs"],
                enforce_eager=False,
                max_model_len=gpu_config["max_model_len"],
                enable_chunked_prefill=False,
                trust_remote_code=True
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
            logger.info("🌟 Procr Model Manager is READY")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MinerU model: {str(e)}")
            raise e

    def get_client(self) -> MinerUClient:
        if not self._is_ready:
            self.initialize_models()
        return self._client

    def get_status(self):
        return {
            "ready": self._is_ready,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "vram_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "N/A"
        }

model_manager = ModelManager()
