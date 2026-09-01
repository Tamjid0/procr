# ═══════════════════════════════════════════════════════════
# CRITICAL: tqdm_patch MUST be imported BEFORE vllm/mineru_vl_utils
# The patch monkey-patches tqdm.tqdm.update() so vLLM's progress
# bars write to progress_store. If vllm is imported first, the
# patch has no effect because vLLM has already captured the
# original tqdm reference.
# ═══════════════════════════════════════════════════════════
from app.core.tqdm_patch import set_job_id, clear_job_id  # noqa: E402

import asyncio
import base64
import io
import json
import time
import logging
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
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
    mineru_output, page_index: int, page_width: int, page_height: int, page_image: Optional[Image.Image] = None
) -> dict:
    """Transform MinerU output into the standard OCR response format."""
    structured_data = MinerUAdapter.transform(mineru_output, page_width, page_height, page_image)
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
            mineru_output, request.page_index, page_width, page_height, image
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
        for result, pm, page_image in zip(batch_results, page_metas, images):
            pages.append(
                _format_single_result(result, pm["page_index"], pm["width"], pm["height"], page_image)
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


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _run_batch_inference(
    job_id: str,
    batch_start: float,
    decode_time: float,
    page_count: int,
    page_metas: list[dict],
    images: list,
):
    """Run batch inference in background and store result."""
    from app.core.progress_store import progress_store

    try:
        logger.info(f"[async-batch] Running batched VLM Inference on {page_count} pages (job: {job_id})...")
        inference_start = time.perf_counter()

        client = model_manager.get_client()
        batch_results = client.batch_two_step_extract(images)

        inference_time = time.perf_counter() - inference_start
        logger.info(f"[async-batch] Batch inference complete in {inference_time:.2f}s (job: {job_id})")

        logger.info(f"[async-batch] Adapting batch results (job: {job_id})...")
        adapt_start = time.perf_counter()

        pages = []
        for result, pm, page_image in zip(batch_results, page_metas, images):
            pages.append(
                _format_single_result(result, pm["page_index"], pm["width"], pm["height"], page_image)
            )

        adapt_time = time.perf_counter() - adapt_start
        total_time = time.perf_counter() - batch_start
        logger.info(
            f"[async-batch] BATCH PERFORMANCE: Total {total_time:.2f}s | "
            f"Decode {decode_time:.2f}s | Inference {inference_time:.2f}s | "
            f"Adapt {adapt_time:.2f}s | Pages {page_count} | "
            f"Effective {total_time / page_count:.2f}s/page (job: {job_id})"
        )

        result = {"pages": pages}
        progress_store.complete(job_id, result)

    except Exception as e:
        logger.error(f"[async-batch] Batch failed (job: {job_id}): {str(e)}")
        progress_store.fail(job_id, str(e))
    finally:
        clear_job_id()


@app.post("/api/v1/ocr/process-batch-async")
async def process_batch_async(
    files: list[UploadFile] = File(...),
    metadata: str = Form(...),
):
    """Async batch OCR - returns job_id immediately, streams progress via SSE.

    The multipart upload + image decoding happens inline (fast, <2s).
    The heavy inference runs in a background task.

    Client then opens GET /api/v1/ocr/batch-stream/{job_id} for progress.
    """
    from app.core.progress_store import progress_store

    batch_start = time.perf_counter()
    page_count = len(files)

    if page_count == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    if page_count > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size limited to {MAX_BATCH_SIZE} pages")

    try:
        meta = json.loads(metadata)
        page_metas: list[dict] = meta.get("pages", [])
        document_id: str = meta.get("document_id", "")

        if len(page_metas) != page_count:
            raise HTTPException(
                status_code=400,
                detail=f"Mismatch: {page_count} files but {len(page_metas)} metadata entries",
            )

        logger.info(f"[async-batch] Decoding {page_count} images (multipart)...")
        decode_start = time.perf_counter()

        images = []
        for i, (f, pm) in enumerate(zip(files, page_metas)):
            raw = await f.read()
            image, orig_w, orig_h = _decode_image_bytes(raw)
            images.append(image)
            pm["width"] = orig_w
            pm["height"] = orig_h

        decode_time = time.perf_counter() - decode_start
        logger.info(f"[async-batch] Decoded {page_count} images in {decode_time:.2f}s")

        job_id = uuid.uuid4().hex
        progress_store.init_job(job_id, document_id, page_count)
        progress_store.set_phase(job_id, "infer_layout", phase_total=page_count)

        set_job_id(job_id)

        asyncio.create_task(
            _run_batch_inference(job_id, batch_start, decode_time, page_count, page_metas, images)
        )

        return {"job_id": job_id, "document_id": document_id, "total_pages": page_count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[async-batch] Setup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ocr/batch-stream/{job_id}")
async def batch_stream(job_id: str):
    """SSE endpoint - streams progress events for a batch OCR job.

    Events:
        progress: {phase, phase_completed, phase_total, total_pages, percent}
        done:     {result: {pages: [...]}}
        error:    {error: str}
    """
    from app.core.progress_store import progress_store

    async def event_stream():
        job = progress_store.get(job_id)
        if not job:
            yield _sse_event("error", {"error": f"Job {job_id} not found"})
            return

        last_completed = -1
        last_phase = ""

        while True:
            job = progress_store.get(job_id)
            if not job:
                yield _sse_event("error", {"error": f"Job {job_id} disappeared"})
                return

            current_phase = job.phase
            current_completed = job.phase_completed

            if current_phase != last_phase or current_completed != last_completed:
                last_phase = current_phase
                last_completed = current_completed

                if current_phase == "decoding":
                    percent = 0
                elif current_phase == "infer_layout":
                    percent = int((job.phase_completed / max(job.phase_total, 1)) * 65) if job.phase_total else 0
                elif current_phase == "infer_text":
                    text_progress = job.phase_completed / max(job.phase_total, 1) if job.phase_total else 0
                    percent = int(65 + text_progress * 35)
                elif current_phase == "done":
                    percent = 100
                elif current_phase == "failed":
                    percent = 0
                else:
                    percent = 0

                yield _sse_event("progress", {
                    "phase": current_phase,
                    "phase_completed": job.phase_completed,
                    "phase_total": job.phase_total,
                    "total_pages": job.total_pages,
                    "percent": percent,
                })

            if job.phase == "done" and job.result:
                yield _sse_event("done", {"result": job.result})
                return
            if job.phase == "failed":
                yield _sse_event("error", {"error": job.error or "Unknown error"})
                return

            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/helper/extract-lines")
async def helper_extract_lines(
    files: list[UploadFile] = File(...),
    metadata: str = Form(...),
):
    """Line-helper for Node VLM path: image+blocks -> line-level extracted_regions.

    Node does AI (CF/Gemini) and sends area blocks [0-1000] with scaled pixel bbox;
    procr does PIL _line_bboxes_from_pixels + _wrap_text_to_lines.
    This keeps citations pixel-level without procr calling AI.
    """
    from app.services.adapter import _line_bboxes_from_pixels, _wrap_text_to_lines, _runs, MAX_INK_GAP_PIXELS, MIN_INK_RATIO, LINE_INK_RATIO

    try:
        meta = json.loads(metadata)
        pages_meta: list[dict] = meta.get("pages", [])
        if len(pages_meta) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Mismatch: {len(files)} files but {len(pages_meta)} metadata entries",
            )

        pages = []
        for f, pm in zip(files, pages_meta):
            raw = await f.read()
            image, orig_w, orig_h = _decode_image_bytes(raw)
            blocks = pm.get("blocks", [])
            page_index = pm.get("page_index", 0)

            # Process VLM blocks: each has {type, bbox, content, confidence, line_count?, line_height_hint?}
            # bbox may be [0-1000] normalized or already pixel-scaled
            extracted_regions = []
            for idx, block in enumerate(blocks):
                btype = block.get("type", "text")
                bbox = block.get("bbox", [0, 0, 0, 0])
                content = block.get("content", "").strip()
                confidence = block.get("confidence", 0.95)
                vlm_line_count = block.get("line_count")
                vlm_line_height_hint = block.get("line_height_hint")

                if not content:
                    continue

                # Scale [0-1000] normalized coords to pixel coordinates
                bx0, by0, bx1, by1 = bbox
                is_normalized = all(0 <= v <= 1000 for v in bbox) and any(v > 1 for v in bbox)
                if is_normalized:
                    x0 = round((bx0 / 1000) * orig_w)
                    y0 = round((by0 / 1000) * orig_h)
                    x1 = round((bx1 / 1000) * orig_w)
                    y1 = round((by1 / 1000) * orig_h)
                else:
                    x0, y0, x1, y1 = int(bx0), int(by0), int(bx1), int(by1)

                # Calculate adaptive gap from VLM hint or block geometry
                block_height_px = y1 - y0
                if vlm_line_height_hint and is_normalized:
                    line_height_px = round((vlm_line_height_hint / 1000) * orig_h)
                elif vlm_line_count and vlm_line_count > 1:
                    line_height_px = max(1, round(block_height_px / vlm_line_count))
                else:
                    line_height_px = None

                adaptive_gap = max(1, round(line_height_px * 0.4)) if line_height_px else MAX_INK_GAP_PIXELS

                # Run PIL line detection with adaptive gap
                line_bboxes = _line_bboxes_from_pixels(image, (x0, y0, x1, y1), orig_w, orig_h, max_gap=adaptive_gap)

                # Validate against VLM line_count hint — retry with tighter gap if wrong
                if vlm_line_count and line_bboxes and len(line_bboxes) != vlm_line_count:
                    tighter_gap = max(1, adaptive_gap // 2)
                    retry_bboxes = _line_bboxes_from_pixels(image, (x0, y0, x1, y1), orig_w, orig_h, max_gap=tighter_gap)
                    if retry_bboxes and abs(len(retry_bboxes) - vlm_line_count) < abs(len(line_bboxes) - vlm_line_count):
                        line_bboxes = retry_bboxes

                if line_bboxes:
                    lines = _wrap_text_to_lines(content, line_bboxes)
                    geometry_source = "pixel_projection"
                    geometries = line_bboxes
                else:
                    lines = [content]
                    geometry_source = "block"
                    geometries = [{"x0": x0, "y0": y0, "x1": x1, "y1": y1}]

                extracted_lines = []
                for line_text, line_bbox in zip(lines, geometries):
                    extracted_lines.append({
                        "text": line_text,
                        "bbox": line_bbox,
                        "confidence_score": confidence,
                        "geometry_source": geometry_source,
                        "style": {
                            "font_size": round(max(1, line_bbox["y1"] - line_bbox["y0"]) * 0.8, 2),
                            "is_bold": btype in ["header", "title"]
                        }
                    })

                extracted_regions.append({
                    "region_index": idx,
                    "region_type": btype,
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "confidence_score": confidence,
                    "extracted_lines": extracted_lines,
                })

            consolidated_text = "\n\n".join(
                "\n".join(l["text"] for l in r["extracted_lines"])
                for r in extracted_regions
            )

            pages.append({
                "page_index": page_index,
                "page_width": orig_w,
                "page_height": orig_h,
                "reading_order_hints": [r["region_index"] for r in extracted_regions],
                "extracted_regions": extracted_regions,
                "text": consolidated_text,
                "confidence": 0.95,
            })

        return {"pages": pages}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[helper] extract-lines failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
