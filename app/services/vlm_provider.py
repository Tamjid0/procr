"""
VlmProvider — Serverless VLM drop-in for MinerU.

Implements the same two_step_extract(image) -> list[block] interface as
MinerUClient, but routes to Cloudflare Workers AI (Qwen) with Gemini 2.5
Flash failover. Returned blocks use the same schema as MinerU so
MinerUAdapter.transform() stays unchanged.
"""
import base64
import io
import json
import logging
import os
import time
from typing import Optional

import httpx
from PIL import Image

logger = logging.getLogger("procr")

# ── Constants ──────────────────────────────────────────────────────────
CF_TIMEOUT_S = 30
GEMINI_TIMEOUT_S = 60
MAX_SIDE = 1280
CF_MAX_RETRIES = 1
GEMINI_MAX_RETRIES = 1

# MinerU-compatible categories — VLM is forced to these
ALLOWED_TYPES = ["text", "title", "header", "table", "figure", "equation", "footer", "list", "code"]

# ── Prompt — forces exact MinerU JSON schema with area targeting ──────
SYSTEM_PROMPT = """You are a document layout analyzer. Analyze the page image and return ONLY valid JSON matching this exact schema:

{
  "blocks": [
    {
      "type": "text|title|header|table|figure|equation|footer|list|code",
      "bbox": [x0, y0, x1, y1],
      "content": "text content",
      "confidence": 0.95
    }
  ]
}

Rules:
1. bbox is normalized [0-1000] grid: x0,y0 top-left, x1,y1 bottom-right. Example [120,80,880,200].
2. Target AREAS/BLOCKS, not individual lines. Group adjacent lines of same paragraph/table into one block.
3. Use categories: text (paragraph), title/header (heading), table, figure, equation (math), footer, list, code.
4. content must be the transcribed text in the ORIGINAL LANGUAGE. Do NOT translate. Preserve markdown for tables.
5. Include ALL text — do not skip fine print, footnotes, or watermarks.
6. If text is multilingual (CJK, Arabic, etc), transcribe exactly as seen.
7. Return ONLY the JSON object, no markdown fences, no commentary.
"""


def _resize_if_needed(image: Image.Image) -> Image.Image:
    """Resize to MAX_SIDE longest edge, matching procr main.py:31."""
    w, h = image.size
    if max(w, h) > MAX_SIDE:
        ratio = MAX_SIDE / max(w, h)
        return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return image


def _image_to_base64(image: Image.Image) -> str:
    """Encode PIL image to base64 PNG."""
    image = _resize_if_needed(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_vlm_json(raw: str) -> list[dict]:
    """Extract blocks[] from VLM response, handling fences and whitespace."""
    text = raw.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        # Remove ```json ... ``` wrapper
        lines = text.split("\n")
        # Drop first fence line and last fence line
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(f"[VlmProvider] JSON parse failed: {exc}. Raw head: {text[:500]}")
        raise ValueError(f"VLM returned invalid JSON: {exc}") from exc

    blocks = data.get("blocks", data.get("extracted_regions", []))
    if not isinstance(blocks, list):
        raise ValueError("VLM JSON missing 'blocks' array")

    # Normalize to MinerU shape
    normalized = []
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type", block.get("region_type", "text"))).lower()
        if btype not in ALLOWED_TYPES:
            btype = "text"
        bbox = block.get("bbox", block.get("coordinates", [0, 0, 0, 0]))
        if not isinstance(bbox, list) or len(bbox) != 4:
            logger.warning(f"[VlmProvider] Block {idx} invalid bbox: {bbox}")
            continue
        content = str(block.get("content", block.get("text", "")) or "")
        confidence = block.get("confidence", block.get("confidence_score", 0.95))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.95

        normalized.append({
            "type": btype,
            "bbox": [float(v) for v in bbox],
            "content": content,
            "confidence": confidence,
        })

    return normalized


class VlmClient:
    """Drop-in replacement for MinerUClient with VLM backends."""

    def __init__(self) -> None:
        self.cf_account_id = os.getenv("CF_ACCOUNT_ID", "")
        self.cf_api_token = os.getenv("CF_API_TOKEN", "")
        self.cf_model = os.getenv("CF_VLM_MODEL", "@cf/qwen/qwen2.5-vl-32b-instruct")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_VLM_MODEL", "gemini-2.5-flash")

        if not self.cf_api_token and not self.gemini_api_key:
            logger.warning("[VlmProvider] No CF_API_TOKEN or GEMINI_API_KEY set — VLM will fail")

    def _call_cloudflare(self, b64_image: str) -> str:
        """Call Cloudflare Workers AI Qwen. Returns raw response text."""
        if not self.cf_api_token or not self.cf_account_id:
            raise RuntimeError("CF_API_TOKEN or CF_ACCOUNT_ID not configured")

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.cf_account_id}/ai/run/{self.cf_model}"
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this document page and return the JSON."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    ],
                },
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        last_err: Optional[Exception] = None
        for attempt in range(CF_MAX_RETRIES + 1):
            try:
                resp = httpx.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.cf_api_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=CF_TIMEOUT_S,
                )
                if resp.status_code == 429:
                    raise RuntimeError(f"CF rate limited (429): {resp.text[:500]}")
                resp.raise_for_status()
                data = resp.json()
                # CF returns {result: {response: "..."}} or {result: "..."}
                result = data.get("result", {})
                if isinstance(result, dict):
                    text = result.get("response", result.get("text", ""))
                    if not text and "choices" in result:
                        # OpenAI-compat shape
                        choices = result.get("choices", [])
                        if choices:
                            text = choices[0].get("message", {}).get("content", "")
                elif isinstance(result, str):
                    text = result
                else:
                    text = str(result)
                if not text:
                    raise ValueError(f"CF empty response: {json.dumps(data)[:800]}")
                return text
            except Exception as exc:
                last_err = exc
                if attempt < CF_MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        raise last_err or RuntimeError("CF call failed")

    def _call_gemini(self, b64_image: str) -> str:
        """Call Gemini 2.5 Flash (direct or via AI Gateway if available)."""
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not configured for failover")

        # Use AI Gateway if configured (same as Node langchain-factory.ts:87)
        gateway_id = os.getenv("AI_GATEWAY_ID", "")
        account_id = os.getenv("CF_ACCOUNT_ID", "")
        if gateway_id and account_id:
            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.gemini_api_key}",
                "Content-Type": "application/json",
                "cf-aig-gateway-id": gateway_id,
            }
            model = self.gemini_model
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            model = self.gemini_model

        # Use AI Gateway OpenAI-compat shape when gateway, otherwise Gemini native
        if gateway_id and account_id:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this document page and return the JSON."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                        ],
                    },
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            }
        else:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": SYSTEM_PROMPT + "\n\nAnalyze this document page and return the JSON."},
                            {"inline_data": {"mime_type": "image/png", "data": b64_image}},
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
            }

        last_err: Optional[Exception] = None
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_S)
                resp.raise_for_status()
                data = resp.json()
                if gateway_id and account_id:
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    raise ValueError(f"Gateway empty response: {json.dumps(data)[:800]}")
                else:
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    raise ValueError(f"Gemini empty response: {json.dumps(data)[:800]}")
            except Exception as exc:
                last_err = exc
                if attempt < GEMINI_MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        raise last_err or RuntimeError("Gemini call failed")

    def _extract_with_fallback(self, image: Image.Image) -> list[dict]:
        """Try CF primary, then Gemini failover."""
        b64 = _image_to_base64(image)

        cf_error: Optional[str] = None
        try:
            logger.info("[VlmProvider] Calling Cloudflare Workers AI (Qwen)...")
            raw = self._call_cloudflare(b64)
            blocks = _parse_vlm_json(raw)
            logger.info(f"[VlmProvider] CF success: {len(blocks)} blocks")
            return blocks
        except Exception as exc:
            cf_error = str(exc)
            logger.warning(f"[VlmProvider] CF failed: {cf_error} — trying Gemini failover")

        try:
            logger.info("[VlmProvider] Calling Gemini 2.5 Flash failover...")
            raw = self._call_gemini(b64)
            blocks = _parse_vlm_json(raw)
            logger.info(f"[VlmProvider] Gemini success: {len(blocks)} blocks (CF error was: {cf_error})")
            return blocks
        except Exception as exc:
            logger.error(f"[VlmProvider] Gemini also failed: {exc} (CF error was: {cf_error})")
            raise RuntimeError(f"Both VLM backends failed. CF: {cf_error} | Gemini: {exc}") from exc

    # ── MinerU-compatible API ──────────────────────────────────────────
    def two_step_extract(self, image: Image.Image) -> list[dict]:
        """Single image — matches MinerUClient.two_step_extract."""
        return self._extract_with_fallback(image)

    def batch_two_step_extract(self, images: list[Image.Image]) -> list[list[dict]]:
        """Batch images — matches MinerUClient.batch_two_step_extract.

        Calls VLM sequentially (CF has rate limits; parallel would burst).
        Each image is independent for failover.
        """
        results: list[list[dict]] = []
        for idx, img in enumerate(images):
            logger.info(f"[VlmProvider] Batch item {idx+1}/{len(images)}")
            try:
                blocks = self._extract_with_fallback(img)
            except Exception as exc:
                logger.error(f"[VlmProvider] Batch item {idx} failed: {exc} — returning empty block")
                blocks = []
            results.append(blocks)
        return results
