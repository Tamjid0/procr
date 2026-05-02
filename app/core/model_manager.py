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

        model_path = "opendatalab/MinerU2.5-Pro-2604-1.2B"
        logger.info(f"🚀 Initializing Procr (MinerU 2.5 Pro)... Model: {model_path}")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Target Device: {device}")

            # Explicitly load model and processor for the transformers backend
            # Using 'auto' for torch_dtype ensures BF16 on T4 GPUs
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype="auto", 
                device_map="auto",
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            self._client = MinerUClient(
                backend="transformers", 
                model=model,
                processor=processor,
                image_analysis=True
            )
            
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
