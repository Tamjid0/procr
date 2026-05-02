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
            etype = getattr(element, "type", None) or element.get("type", "text")
            bbox = getattr(element, "bbox", None) or element.get("bbox", [0, 0, 0, 0])
            content = getattr(element, "content", None) or element.get("content", "")
            confidence = getattr(element, "confidence", 0.95) if hasattr(element, "confidence") else element.get("confidence", 0.95)
            
            # Normalize coordinates to 0-100 (pdfx standard)
            x0, y0, x1, y1 = bbox
            x = (x0 / page_width) * 100
            y = (y0 / page_height) * 100
            w = ((x1 - x0) / page_width) * 100
            h = ((y1 - y0) / page_height) * 100
            
            # Create a region block
            region = {
                "region_id": f"reg-{idx}",
                "region_type": etype,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "extracted_lines": []
            }
            
            # If the block has text, split by newline to create atomic lines for citations
            lines = content.split('\n')
            for l_idx, line_text in enumerate(lines):
                if not line_text.strip():
                    continue
                    
                # Approximate line height if multiple lines exist in one block
                line_count = max(1, len(lines))
                line_h = (y1 - y0) / line_count
                line_y0 = y0 + (l_idx * line_h)
                line_y1 = line_y0 + line_h
                
                region["extracted_lines"].append({
                    "text": line_text.strip(),
                    "bbox": {"x0": x0, "y0": line_y0, "x1": x1, "y1": line_y1},
                    "confidence_score": confidence
                })
                
            extracted_regions.append(region)
            
        return {
            "page_width": page_width,
            "page_height": page_height,
            "extracted_regions": extracted_regions
        }
