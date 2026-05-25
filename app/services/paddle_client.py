import httpx
import logging
import os

logger = logging.getLogger("procr")

class PaddleOCRClient:
    def __init__(self, base_url: str = None):
        # Default to localhost:9002 if not provided, or use environment variable
        # For local testing with Colab, set PADDLE_OCR_URL to your ngrok tunnel
        url = base_url or os.environ.get("PADDLE_OCR_URL", "http://127.0.0.1:9002")
        
        # Robustly clean the URL
        if url:
            url = url.strip().strip('"').strip("'").rstrip('/')
            
            # ── PILLAR C V15.0: HTTPS Force (Fixes 405 Redirects) ──
            # If it's ngrok, it MUST be https or the POST will become a GET during redirect
            if "ngrok" in url:
                if url.startswith("http://"):
                    url = "https://" + url[7:]
                elif not url.startswith("https://"):
                    url = "https://" + url
            elif not url.startswith("http://") and not url.startswith("https://"):
                url = "http://" + url
        
        self.base_url = url
        logger.info(f"🔗 PaddleOCR client initialized with base_url: {self.base_url}")

    async def get_line_bboxes(self, document_id: str, page_index: int, image_base64: str):
        """
        Calls the PaddleOCR service to get precise line-level bounding boxes.
        """
        try:
            # Increased timeout to 120s for slow ngrok uploads
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/ocr/lines",
                    json={
                        "document_id": document_id,
                        "page_index": page_index,
                        "image_data": image_base64
                    },
                    headers={"ngrok-skip-browser-warning": "69420"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"⚠️ PaddleOCR service call failed: {str(e)}")
            return None

paddle_client = PaddleOCRClient()
