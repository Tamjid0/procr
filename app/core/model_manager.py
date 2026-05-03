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
            # Switching to high-performance vLLM backend
            self._client = MinerUClient(
                model_path=model_path,
                backend="vllm-engine", 
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
