import logging

logger = logging.getLogger("procr")

class MinerUAdapter:
    @staticmethod
    def _remove_stutter(text):
        """Detects and prunes verbatim text repetitions (VLM stutter)."""
        if not text or len(text) < 20:
            return text
        
        # Look for long repeated phrases (at least 15 chars)
        # We check if the string ends with a repeat of a preceding segment
        # e.g. "Line 1 Line 2 Line 1 Line 2" -> "Line 1 Line 2"
        for length in range(len(text) // 2, 10, -1):
            suffix = text[-length:]
            preceding = text[-(2*length):-length]
            if suffix == preceding:
                logger.info(f"✂️ Pruning stutter: '{suffix[:30]}...'")
                return text[:-length].strip()
        return text

    @staticmethod
    def transform(mineru_output, page_width, page_height, mineru_width=None, mineru_height=None):
        """
        Transforms MinerU 2.5 Pro JSON output into standardized DocumentGraph nodes.
        
        Args:
            mineru_output (list): List of elements from MinerUClient.
            page_width (int): Original image width.
            page_height (int): Original image height.
            mineru_width (int): Downscaled width used for VLM inference.
            mineru_height (int): Downscaled height used for VLM inference.
        """
        # If thumbnail dimensions aren't provided, fallback to original dimensions
        if mineru_width is None:
            mineru_width = page_width
        if mineru_height is None:
            mineru_height = page_height

        extracted_regions = []
        
        # In MinerU 2.5, the output is often a list of blocks.
        # Each block might contain text, tables, or formulas.
        
        for idx, element in enumerate(mineru_output):
            # Element could be a dict or a ContentBlock object
            etype = getattr(element, "type", None) or (element.get("type", "text") if isinstance(element, dict) else "text")
            bbox = getattr(element, "bbox", None) or (element.get("bbox", [0, 0, 0, 0]) if isinstance(element, dict) else [0, 0, 0, 0])
            
            # Robustly get content and confidence
            if isinstance(element, dict):
                content = element.get("content", "")
                confidence = element.get("confidence", 0.95)
            else:
                content = getattr(element, "content", "")
                confidence = getattr(element, "confidence", 0.95)
            
            # Ensure content is a string and prune stuttering
            if content is None: content = ""
            content = MinerUAdapter._remove_stutter(str(content).strip())
            
            # Smart Scaling Logic: Detect if VLM is using 0-1000 normalized coords (common for Qwen2-VL)
            # or raw/downscaled pixels.
            is_normalized = all(v <= 1000 for v in bbox) and any(v > 1 for v in bbox)
            
            if is_normalized and any(v > mineru_width and v > mineru_height for v in bbox):
                # Scale 0-1000 -> Original Pixels
                x0 = round((bbox[0] / 1000) * page_width)
                y0 = round((bbox[1] / 1000) * page_height)
                x1 = round((bbox[2] / 1000) * page_width)
                y1 = round((bbox[3] / 1000) * page_height)
            elif all(v <= 1.0 for v in bbox):
                # Scale 0-1 -> Original Pixels
                x0 = round(bbox[0] * page_width)
                y0 = round(bbox[1] * page_height)
                x1 = round(bbox[2] * page_width)
                y1 = round(bbox[3] * page_height)
            else:
                # Scale from downscaled/thumbnail pixels -> Original Pixels
                x0 = round((bbox[0] / mineru_width) * page_width)
                y0 = round((bbox[1] / mineru_height) * page_height)
                x1 = round((bbox[2] / mineru_width) * page_width)
                y1 = round((bbox[3] / mineru_height) * page_height)

            # --- MATH TUNING ---
            # If it's a math/equation block, shift it up and tighten it
            is_math = any(m in etype.lower() for m in ["equation", "formula", "math"])
            if is_math:
                h_orig = y1 - y0
                offset = int(page_height * 0.01) # 1% upward shift
                y0 = max(0, y0 - offset)
                # Tighten the box by 20% to center it better on the symbols
                y1 = max(0, y1 - offset - int(h_orig * 0.2))
            
            # Create a region block
            region = {
                "region_id": f"reg-{idx}",
                "region_index": int(idx),
                "region_type": str(etype),
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "confidence_score": float(confidence),
                "extracted_lines": []
            }
            
            # ── PILLAR C V6.1: Raw Block Passing ──
            # We stop pre-slicing into lines here. We pass the whole text block to the merger.
            # merger.py now handles character-level splitting based on PHYSICAL PaddleOCR lines.
            if content.strip():
                region["extracted_lines"].append({
                    "text": content.strip(),
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "confidence_score": float(confidence),
                    "style": {
                        "font_size": 12, # Placeholder, merger will refine
                        "is_bold": etype in ["header", "title"]
                    }
                })
                
            extracted_regions.append(region)
            
        return {
            "page_width": page_width,
            "page_height": page_height,
            "extracted_regions": extracted_regions
        }
