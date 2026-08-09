import base64
import io
import json
import time
import logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
from typing import Optional

from app.core.model_manager import model_manager
from app.services.adapter import MinerUAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("procr")

app = FastAPI(title="Procr v2 - MinerU 2.5 Pro")

MAX_SIDE = 1280
MAX_BATCH_SIZE = 64


class OCRRequest(BaseModel):
    document_id: str
    page_index: int
    image_data: str
    image_width: int
    image_height: int
    processing_flags: Optional[dict] = None


def _decode_image_bytes(data: bytes) -> tuple[Image.Image, int, int]:
    """Decode raw image bytes, resize if needed, return (image, orig_w, orig_h)."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    orig_w, orig_h = image.size
    if max(orig_w, orig_h) > MAX_SIDE:
        ratio = MAX_SIDE / max(orig_w, orig_h)
        image = image.resize((int(orig_w * ratio), int(orig_h * ratio)), Image.LANCZOS)
    return image, orig_w, orig_h


def _get_content(region) -> str:
    """Extract text content from a MinerU ContentBlock or dict."""
    if isinstance(region, dict):
        return str(region.get("content", "") or "")
    return str(getattr(region, "content", "") or "")


def _format_single_result(
    mineru_output, page_index: int, page_width: int, page_height: int
) -> dict:
    """Transform MinerU output into the standard OCR response format."""
    structured_data = MinerUAdapter.transform(mineru_output, page_width, page_height)
    consolidated_text = "\n".join(
        _get_content(r) for r in mineru_output if r is not None
    )
    return {
        "page_index": page_index,
        "page_width": page_width,
        "page_height": page_height,
        "reading_order_hints": [
            r["region_index"] for r in structured_data["extracted_regions"]
        ],
        "extracted_regions": structured_data["extracted_regions"],
        "text": consolidated_text,
        "confidence": 0.95,
    }


@app.on_event("startup")
async def startup_event():
    model_manager.initialize_models()


@app.get("/diagnostic")
async def diagnostic():
    return {
        "service": "procr",
        "version": "2.5.0-Pro-2604",
        "status": model_manager.get_status(),
        "timestamp": time.time(),
    }


@app.post("/api/v1/ocr/process-page")
async def process_page(request: OCRRequest):
    start_time = time.perf_counter()
    try:
        logger.info("Decoding image...")
        img_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        page_width, page_height = image.size
        logger.info(f"Image Decoded: {page_width}x{page_height}")

        if max(page_width, page_height) > MAX_SIDE:
            ratio = MAX_SIDE / max(page_width, page_height)
            new_w, new_h = int(page_width * ratio), int(page_height * ratio)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"Resized to {new_w}x{new_h} (ratio={ratio:.2f})")

        decode_time = time.perf_counter()

        logger.info("Running VLM Inference...")
        client = model_manager.get_client()
        mineru_output = client.two_step_extract(image)
        inference_time = time.perf_counter()

        logger.info("Processing results...")
        result = _format_single_result(
            mineru_output, request.page_index, page_width, page_height
        )
        mapping_time = time.perf_counter()

        total_time = mapping_time - start_time
        logger.info(
            f"PERFORMANCE: Total {total_time:.2f}s | "
            f"Inference {inference_time - decode_time:.2f}s | "
            f"Mapping {mapping_time - inference_time:.2f}s"
        )

        return result

    except Exception as e:
        logger.error(f"Error processing page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ocr/process-batch")
async def process_batch(
    files: list[UploadFile] = File(...),
    metadata: str = Form(...),
):
    """Multipart binary upload for batch OCR.

    - ``files``: ordered list of raw image binaries (PNG/JPEG).
    - ``metadata``: JSON string with keys:
        - ``document_id``: str
        - ``pages``: list of ``{"page_index": int, "page_width": int, "page_height": int}``

    Files must be ordered to match the ``pages`` array by index.
    """
    batch_start = time.perf_counter()
    page_count = len(files)

    if page_count == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if page_count > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, detail=f"Batch size limited to {MAX_BATCH_SIZE} pages"
        )

    try:
        meta = json.loads(metadata)
        page_metas: list[dict] = meta.get("pages", [])
        document_id: str = meta.get("document_id", "")

        if len(page_metas) != page_count:
            raise HTTPException(
                status_code=400,
                detail=f"Mismatch: {page_count} files but {len(page_metas)} metadata entries",
            )

        logger.info(f"Batch decoding {page_count} images (multipart)...")
        decode_start = time.perf_counter()

        images = []
        for i, (f, pm) in enumerate(zip(files, page_metas)):
            raw = await f.read()
            image, orig_w, orig_h = _decode_image_bytes(raw)
            images.append(image)
            pm["width"] = orig_w
            pm["height"] = orig_h

        decode_time = time.perf_counter() - decode_start
        logger.info(f"Decoded {page_count} images in {decode_time:.2f}s")

        logger.info(f"Running batched VLM Inference on {page_count} pages...")
        inference_start = time.perf_counter()

        client = model_manager.get_client()
        batch_results = client.batch_two_step_extract(images)

        inference_time = time.perf_counter() - inference_start
        logger.info(
            f"Batch inference complete in {inference_time:.2f}s "
            f"({inference_time / page_count:.2f}s/page effective)"
        )

        logger.info("Adapting batch results...")
        adapt_start = time.perf_counter()

        pages = []
        for result, pm in zip(batch_results, page_metas):
            pages.append(
                _format_single_result(result, pm["page_index"], pm["width"], pm["height"])
            )

        adapt_time = time.perf_counter() - adapt_start
        total_time = time.perf_counter() - batch_start

        logger.info(
            f"BATCH PERFORMANCE: Total {total_time:.2f}s | "
            f"Decode {decode_time:.2f}s | "
            f"Inference {inference_time:.2f}s | "
            f"Adapt {adapt_time:.2f}s | "
            f"Pages {page_count} | "
            f"Effective {total_time / page_count:.2f}s/page"
        )

        return {"pages": pages}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
