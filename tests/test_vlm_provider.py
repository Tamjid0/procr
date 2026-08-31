"""Tests for VlmProvider — area-targeted JSON, failover."""
import json
import base64
import io
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from app.services.vlm_provider import _parse_vlm_json, _image_to_base64, SYSTEM_PROMPT


class TestParseVlmJson(unittest.TestCase):
    def test_valid_blocks(self):
        raw = json.dumps({"blocks": [{"type": "text", "bbox": [10, 20, 100, 50], "content": "hello", "confidence": 0.99}]})
        blocks = _parse_vlm_json(raw)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[0]["bbox"], [10.0, 20.0, 100.0, 50.0])

    def test_markdown_fence_stripped(self):
        inner = json.dumps({"blocks": [{"type": "title", "bbox": [0, 0, 1000, 100], "content": "Title"}]})
        raw = f"```json\n{inner}\n```"
        blocks = _parse_vlm_json(raw)
        self.assertEqual(blocks[0]["type"], "title")

    def test_invalid_bbox_skipped(self):
        raw = json.dumps({"blocks": [{"type": "text", "bbox": [1, 2], "content": "bad"}]})
        blocks = _parse_vlm_json(raw)
        self.assertEqual(len(blocks), 0)

    def test_unknown_type_normalized_to_text(self):
        raw = json.dumps({"blocks": [{"type": "unknown_cat", "bbox": [0, 0, 100, 100], "content": "x"}]})
        blocks = _parse_vlm_json(raw)
        self.assertEqual(blocks[0]["type"], "text")

    def test_empty_blocks(self):
        raw = json.dumps({"blocks": []})
        blocks = _parse_vlm_json(raw)
        self.assertEqual(blocks, [])

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            _parse_vlm_json("not json")

    def test_system_prompt_has_area_rule(self):
        self.assertIn("AREAS/BLOCKS", SYSTEM_PROMPT)


class TestImageToBase64(unittest.TestCase):
    def test_resize_large_image(self):
        img = Image.new("RGB", (2000, 3000), color="white")
        b64 = _image_to_base64(img)
        decoded = base64.b64decode(b64)
        out = Image.open(io.BytesIO(decoded))
        self.assertLessEqual(max(out.size), 1280)

    def test_small_image_not_resized(self):
        img = Image.new("RGB", (100, 100), color="white")
        b64 = _image_to_base64(img)
        decoded = base64.b64decode(b64)
        out = Image.open(io.BytesIO(decoded))
        self.assertEqual(out.size, (100, 100))


class TestVlmClientFailover(unittest.TestCase):
    @patch("app.services.vlm_provider.httpx.post")
    def test_cf_success_no_fallback(self, mock_post):
        from app.services.vlm_provider import VlmClient
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {"response": json.dumps({"blocks": [{"type": "text", "bbox": [0, 0, 500, 100], "content": "hi"}]})}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = VlmClient()
        client.cf_account_id = "test"
        client.cf_api_token = "tok"
        blocks = client.two_step_extract(Image.new("RGB", (100, 100), color="white"))
        self.assertEqual(len(blocks), 1)
        # Only one POST (CF), no Gemini
        self.assertEqual(mock_post.call_count, 1)

    @patch("app.services.vlm_provider.httpx.post")
    def test_cf_fails_then_gemini(self, mock_post):
        from app.services.vlm_provider import VlmClient

        def side_effect(url, **kwargs):
            if "cloudflare" in url:
                raise RuntimeError("CF 429")
            # Gemini call
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = {"candidates": [{"content": {"parts": [{"text": json.dumps({"blocks": [{"type": "text", "bbox": [0, 0, 100, 100], "content": "fallback"}]})}]}}]}
            m.raise_for_status = MagicMock()
            return m

        mock_post.side_effect = side_effect
        client = VlmClient()
        client.cf_account_id = "test"
        client.cf_api_token = "tok"
        client.gemini_api_key = "g-key"
        blocks = client.two_step_extract(Image.new("RGB", (100, 100), color="white"))
        self.assertEqual(blocks[0]["content"], "fallback")


if __name__ == "__main__":
    unittest.main()
