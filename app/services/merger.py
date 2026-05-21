import logging

logger = logging.getLogger("procr")

class OCRMerger:
    @staticmethod
    def merge(mineru_data, paddle_data):
        """
        Maps PaddleOCR lines into MinerU blocks.
        
        Args:
            mineru_data (dict): Standardized MinerU output from MinerUAdapter.
            paddle_data (dict): Line-level output from PaddleOCR service.
        """
        if not paddle_data or "lines" not in paddle_data:
            logger.warning("No PaddleOCR data to merge, returning original MinerU data.")
            return mineru_data

        merged_regions = []
        paddle_lines = paddle_data["lines"]
        
        # Track which paddle lines have been assigned to a block
        assigned_paddle_indices = set()

        for region in mineru_data["extracted_regions"]:
            block_bbox = region["bbox"]
            
            # Find paddle lines that fall within this block
            # We use a simple intersection-over-area or "center point in block" approach
            matched_lines = []
            
            for i, p_line in enumerate(paddle_lines):
                p_bbox = p_line["bbox"]
                
                # Calculate center point of paddle line
                cx = (p_bbox["x0"] + p_bbox["x1"]) / 2
                cy = (p_bbox["y0"] + p_bbox["y1"]) / 2
                
                # Check if center point is inside the MinerU block
                if (block_bbox["x0"] <= cx <= block_bbox["x1"] and 
                    block_bbox["y0"] <= cy <= block_bbox["y1"]):
                    
                    matched_lines.append({
                        "text": p_line["text"],
                        "bbox": p_bbox,
                        "confidence_score": p_line["confidence"],
                        "style": {
                            "font_size": round((p_bbox["y1"] - p_bbox["y0"]) * 0.8, 2),
                            "is_bold": region["region_type"] in ["header", "title"]
                        }
                    })
                    assigned_paddle_indices.add(i)

            # If we found matches, replace the "fake" lines in the region
            if matched_lines:
                # Sort matched lines by Y coordinate
                matched_lines.sort(key=lambda x: x["bbox"]["y0"])
                region["extracted_lines"] = matched_lines
                
            merged_regions.append(region)

        # Handle "Orphan" Paddle lines (text found by Paddle but not by MinerU blocks)
        # This can happen if MinerU misses a text region entirely.
        for i, p_line in enumerate(paddle_lines):
            if i not in assigned_paddle_indices:
                p_bbox = p_line["bbox"]
                merged_regions.append({
                    "region_id": f"reg-orphan-{i}",
                    "region_index": 999 + i, # Put orphans at the end
                    "region_type": "text",
                    "bbox": p_bbox,
                    "confidence_score": p_line["confidence"],
                    "extracted_lines": [{
                        "text": p_line["text"],
                        "bbox": p_bbox,
                        "confidence_score": p_line["confidence"],
                        "style": {
                            "font_size": round((p_bbox["y1"] - p_bbox["y0"]) * 0.8, 2),
                            "is_bold": False
                        }
                    }]
                })

        return {
            "page_width": mineru_data["page_width"],
            "page_height": mineru_data["page_height"],
            "extracted_regions": merged_regions
        }
