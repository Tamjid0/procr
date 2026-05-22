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
            url = url.strip().strip('"').strip("'")
            if not url.startswith("http://") and not url.startswith("https://"):
                # Default to https for ngrok tunnels, http otherwise
                if "ngrok" in url or ("localhost" not in url and not url.startswith("127.0.0.1")):
                    url = "https://" + url
                else:
                    url = "http://" + url
        
        self.base_url = url
        logger.info(f"🔗 PaddleOCR client initialized with base_url: {self.base_url}")

    async def get_line_bboxes(self, document_id: str, page_index: int, image_base64: str):
        """
        Calls the PaddleOCR service to get precise line-level bounding boxes.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/ocr/lines",
                    json={
                        "document_id": document_id,
                        "page_index": page_index,
                        "image_data": image_base64
                    }
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"⚠️ PaddleOCR service call failed: {str(e)}")
            return None

paddle_client = PaddleOCRClient()
