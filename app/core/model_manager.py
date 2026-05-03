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

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        import torch

        model_path = "opendatalab/MinerU2.5-Pro-2604-1.2B"
        logger.info(f"🚀 Initializing Optimized Procr... Model: {model_path}")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Target Device: {device}")

            # 4-bit Quantization Configuration for T4 Speedup
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Explicitly load model with optimizations
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, 
                quantization_config=quant_config,
                attn_implementation="sdpa", # Torch native Flash-Attention path
                device_map="auto",
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(
                model_path, 
                trust_remote_code=True,
                min_pixels=256*28*28,
                max_pixels=1024*28*28 # Slightly tighter for more speed
            )
            
            self._client = MinerUClient(
                backend="transformers", 
                model=model,
                processor=processor,
                image_analysis=True
            )
            
            # Warm up the model with a tiny dummy inference
            logger.info("🔥 Warming up optimized VLM kernels...")
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
