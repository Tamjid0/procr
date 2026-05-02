import logging

logger = logging.getLogger("procr")

class MinerUAdapter:
    @staticmethod
    def transform(mineru_output, page_width, page_height):
        """
        Transforms MinerU 2.5 Pro JSON output into standardized DocumentGraph nodes.
        
        Args:
            mineru_output (list): List of elements from MinerUClient.
            page_width (int): Original image width.
            page_height (int): Original image height.
        """
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
            
            # Ensure content is a string
            if content is None: content = ""
            content = str(content)
            
            # Smart Scaling Logic: Detect if VLM is using 0-1000 normalized coords (common for Qwen2-VL)
            # or raw pixels.
            is_normalized = all(v <= 1000 for v in bbox) and any(v > 1 for v in bbox)
            
            if is_normalized:
                # Scale 0-1000 -> Pixels
                x0 = round((bbox[0] / 1000) * page_width)
                y0 = round((bbox[1] / 1000) * page_height)
                x1 = round((bbox[2] / 1000) * page_width)
                y1 = round((bbox[3] / 1000) * page_height)
            else:
                # Keep as raw pixels (or scale if they were 0-1)
                if all(v <= 1 for v in bbox):
                    x0 = round(bbox[0] * page_width)
                    y0 = round(bbox[1] * page_height)
                    x1 = round(bbox[2] * page_width)
                    y1 = round(bbox[3] * page_height)
                else:
                    x0, y0, x1, y1 = [round(v) for v in bbox]

            # --- MATH TUNING ---
            # If it's a math/equation block, shift it up slightly to fix the "downward" drift
            is_math = any(m in etype.lower() for m in ["equation", "formula", "math"])
            if is_math:
                offset = int(page_height * 0.005) # 0.5% upward shift
                y0 = max(0, y0 - offset)
                y1 = max(0, y1 - offset)
            
            # Create a region block
            region = {
                "region_id": f"reg-{idx}",
                "region_index": int(idx),
                "region_type": str(etype),
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "confidence_score": float(confidence),
                "extracted_lines": []
            }
            
            # If the block has text, split by newline to create atomic lines for citations
            if content.strip():
                lines = content.split('\n')
                for l_idx, line_text in enumerate(lines):
                    if not line_text.strip():
                        continue
                        
                    line_count = max(1, len(lines))
                    line_h = (y1 - y0) / line_count
                    line_y0 = round(y0 + (l_idx * line_h))
                    line_y1 = round(line_y0 + line_h)
                    
                    region["extracted_lines"].append({
                        "text": line_text.strip(),
                        "bbox": {"x0": x0, "y0": line_y0, "x1": x1, "y1": line_y1},
                        "confidence_score": float(confidence),
                        "style": {
                            "font_size": round(line_h * 0.8, 2),
                            "is_bold": etype in ["header", "title"]
                        }
                    })
                
            extracted_regions.append(region)
            
        return {
            "page_width": page_width,
            "page_height": page_height,
            "extracted_regions": extracted_regions
        }
