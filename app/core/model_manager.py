import os
# Force stable V0 engine to prevent T4 initialization crashes (V1 is unstable here)
os.environ["VLLM_USE_V1"] = "0"

import torch
import logging
from mineru_vl_utils import MinerUClient

logger = logging.getLogger("procr")

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
        logger.info(f"🚀 Initializing Procr (Proven Path)... Model: {model_path}")
        
        try:
            import vllm
            
            # 1. Manually initialize the vLLM engine with our high-performance settings
            # HYPER-TUNING FOR SUB-10S LATENCY (Speed Mode)
            # We enable CUDA Graphs (enforce_eager=False) to eliminate Python overhead.
            # NOTE: The first request after startup will have a ~45s 'warmup' penalty.
            logger.info("🔥 Hyper-tuning vLLM engine for T4 (Speed Mode)...")
            tuned_engine = vllm.LLM(
                model=model_path,
                gpu_memory_utilization=0.90, # Restored to 0.90 for maximum KV Cache speed
                max_num_seqs=8,
                enforce_eager=False,          # Re-enabled CUDA Graphs for blazing-fast inference speed
                max_model_len=8192,           # Reduced to 2048 to save memory
                enable_chunked_prefill=False,
                trust_remote_code=True,
                mm_processor_kwargs={"max_pixels": 282240} # Clamp resolution for 2s speed
            )
            
            # 2. Pass the pre-initialized engine to MinerU
            self._client = MinerUClient(
                backend="vllm-engine",
                vllm_llm=tuned_engine, 
                image_analysis=True
            )
            
            # Keep the warmup to avoid first-request timeout
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
