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
            # We use enforce_eager=True because CUDA Graphs are too slow to capture on T4.
            logger.info("🔥 Hyper-tuning vLLM engine for T4...")
            tuned_engine = vllm.LLM(
                model=model_path,
                gpu_memory_utilization=0.95,
                max_num_seqs=16,
                enforce_eager=True,
                max_model_len=4096,
                trust_remote_code=True
            )
            
            # 2. Pass the pre-initialized engine to MinerU
            self._client = MinerUClient(
                backend="vllm-engine",
                vllm_llm=tuned_engine, 
                image_analysis=True,
                layout_image_size=(896, 896)
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
