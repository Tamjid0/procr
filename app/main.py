import base64
import io
import time
import logging
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
        img_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        page_width, page_height = image.size
        decode_time = time.perf_counter()
        
        # 2. VLM Inference
        client = model_manager.get_client()
        mineru_output = client.two_step_extract(image)
        inference_time = time.perf_counter()
        
        # 3. Adapt Output
        structured_data = MinerUAdapter.transform(mineru_output, page_width, page_height)
        mapping_time = time.perf_counter()
        
        # 4. Consolidate Text
        consolidated_text = "\n".join([getattr(r, "content", "") or r.get("content", "") for r in mineru_output])

        # Performance Logging
        total_time = mapping_time - start_time
        logger.info(
            f"⏱️ PERFORMANCE: Total {total_time:.2f}s | "
            f"Inference {inference_time - decode_time:.2f}s | "
            f"Mapping {mapping_time - inference_time:.2f}s"
        )
        
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
    uvicorn.run(app, host="0.0.0.0", port=8080)
