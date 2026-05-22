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
from typing import Optional
import asyncio

from app.core.model_manager import model_manager
from app.services.adapter import MinerUAdapter
from app.services.paddle_client import paddle_client
from app.services.merger import OCRMerger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("procr")

app = FastAPI(title="Procr v2 - MinerU 2.5 Pro + PaddleOCR v4")

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
        "version": "2.5.0-Pro-2604-Paddle",
        "status": model_manager.get_status(),
        "timestamp": time.time()
    }

@app.post("/api/v1/ocr/process-page")
async def process_page(request: OCRRequest):
    start_time = time.perf_counter()
    try:
        # 1. Decode Image
        logger.info(f"📸 Decoding image for doc {request.document_id} page {request.page_index}...")
        img_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        page_width, page_height = image.size
        
        # --- SAFE PERFORMANCE TEST: Drop visual tokens by 60% ---
        # For MinerU, we downscale.
        mineru_image = image.copy()
        mineru_image.thumbnail((512, 680), Image.Resampling.LANCZOS)
        mineru_width, mineru_height = mineru_image.size
        
        logger.info(f"📄 Image Decoded: {page_width}x{page_height} | Downscaled MinerU Image: {mineru_width}x{mineru_height}")
        decode_time = time.perf_counter()
        
        # 2. Run OCR Tasks Concurrently
        logger.info("🧠 Running VLM and PaddleOCR Concurrently...")
        
        async def run_mineru():
            try:
                client = model_manager.get_client()
                return client.two_step_extract(mineru_image)
            except Exception as e:
                logger.error(f"❌ MinerU extraction failed: {str(e)}")
                return []

        async def run_paddle():
            try:
                return await paddle_client.get_line_bboxes(
                    request.document_id, 
                    request.page_index, 
                    request.image_data
                )
            except Exception as e:
                logger.error(f"❌ PaddleOCR client call failed: {str(e)}")
                return None

        # Run both in parallel
        mineru_task = asyncio.create_task(run_mineru())
        paddle_task = asyncio.create_task(run_paddle())
        
        mineru_output, paddle_output = await asyncio.gather(mineru_task, paddle_task)
        inference_time = time.perf_counter()
        
        # 3. Adapt & Merge Output
        logger.info("🎯 Merging results...")
        
        # Transform MinerU output first
        structured_data = MinerUAdapter.transform(
            mineru_output, 
            page_width, 
            page_height, 
            mineru_width=mineru_width, 
            mineru_height=mineru_height
        )
        
        # Merge with PaddleOCR lines
        final_data = OCRMerger.merge(structured_data, paddle_output)
        mapping_time = time.perf_counter()
        
        # 4. Consolidate Text for the response
        consolidated_text_lines = []
        for reg in final_data["extracted_regions"]:
            for line in reg.get("extracted_lines", []):
                consolidated_text_lines.append(line.get("text", ""))
        
        consolidated_text = "\n".join(consolidated_text_lines)

        # Performance Logging
        total_time = mapping_time - start_time
        logger.info(
            f"⏱️ PERFORMANCE: Total {total_time:.2f}s | "
            f"Inference (Parallel) {inference_time - decode_time:.2f}s | "
            f"Mapping/Merge {mapping_time - inference_time:.2f}s"
        )
        
        # Persistent stats logging
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(project_root, "ocr_stats.log")
        with open(log_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Doc: {request.document_id} | Res: {page_width}x{page_height} | Total: {total_time:.2f}s | Paddle: {'Success' if paddle_output else 'Failed'}\n")
        
        # Final Flat Response (Surya v1 Contract)
        return {
            "page_index": request.page_index,
            "page_width": page_width,
            "page_height": page_height,
            "reading_order_hints": [r["region_index"] for r in final_data["extracted_regions"]],
            "extracted_regions": final_data["extracted_regions"],
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
