import os
# Force stable V0 engine at the absolute entry point (V1 crashes on T4/Colab)
os.environ["VLLM_USE_V1"] = "0"

import base64
import io
import time
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from app.core.model_manager import model_manager
from app.services.adapter import MinerUAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("procr")

app = FastAPI(title="Procr v2 - MinerU 2.5 Pro")

from typing import Optional

class OCRRequest(BaseModel):
    document_id: str
    page_index: int
    image_data: str  # Base64 encoded image
    image_width: int
    image_height: int
    processing_flags: Optional[dict] = None

@app.on_event("startup")
async def startup_event():
    # Eagerly load the model
    model_manager.initialize_models()

@app.get("/diagnostic")
async def diagnostic():
    return {
        "service": "procr",
        "version": "2.5.0-Pro-2604",
        "status": model_manager.get_status(),
        "timestamp": time.time()
    }

@app.post("/api/v1/ocr/process-page")
async def process_page(request: OCRRequest):
    start_time = time.perf_counter()
    try:
        # 1. Decode Image
        logger.info("📸 Decoding image...")
        img_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        page_width, page_height = image.size
        
        # --- SAFE PERFORMANCE TEST: Drop visual tokens by 60% ---
        # We downscale to 768px width (standard A4 ratio gives ~768x1024).
        # We preserve the original page_width and page_height for the adapter geometry mapping!
        image.thumbnail((768, 1024), Image.Resampling.LANCZOS)
        
        logger.info(f"📄 Image Decoded: {page_width}x{page_height} (Downscaled to {image.size[0]}x{image.size[1]} for 2s speed)")
        decode_time = time.perf_counter()
        
        # 2. VLM Inference
        logger.info("🧠 Running VLM Inference...")
        client = model_manager.get_client()
        mineru_output = client.two_step_extract(image)
        inference_time = time.perf_counter()
        
        # 3. Adapt Output
        logger.info("🎯 Processing results...")
        structured_data = MinerUAdapter.transform(mineru_output, page_width, page_height)
        mapping_time = time.perf_counter()
        
        # 4. Consolidate Text
        def get_content(r):
            if isinstance(r, dict):
                return str(r.get("content", "") or "")
            return str(getattr(r, "content", "") or "")
            
        consolidated_text = "\n".join([get_content(r) for r in mineru_output if r is not None])

        # Performance Logging
        total_time = mapping_time - start_time
        logger.info(
            f"⏱️ PERFORMANCE: Total {total_time:.2f}s | "
            f"Inference {inference_time - decode_time:.2f}s | "
            f"Mapping {mapping_time - inference_time:.2f}s"
        )
        
        # Persistent stats logging (Absolute path for cross-environment visibility)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(project_root, "ocr_stats.log")
        with open(log_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Res: {page_width}x{page_height} | Total: {total_time:.2f}s | Inf: {inference_time - decode_time:.2f}s\n")
        
        # Final Flat Response (Surya v1 Contract)
        return {
            "page_index": request.page_index,
            "page_width": page_width,
            "page_height": page_height,
            "reading_order_hints": [r["region_index"] for r in structured_data["extracted_regions"]],
            "extracted_regions": structured_data["extracted_regions"],
            "text": consolidated_text,
            "confidence": 0.95
        }
        
    except Exception as e:
        logger.error(f"Error processing page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable if set, otherwise default to 9001
    port = int(os.environ.get("PORT", 9001))
    uvicorn.run(app, host="127.0.0.1", port=port)
