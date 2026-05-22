import logging
import re
import difflib

logger = logging.getLogger("procr")

class OCRMerger:
    @staticmethod
    def _clean_text(text):
        if not text:
            return ""
        # Remove markdown characters like | - * # [ ] ( ) ` _ +
        cleaned = re.sub(r'[\|`\-*#_\+\[\]\(\)\{\}]', ' ', text)
        cleaned = ' '.join(cleaned.split())
        return cleaned.lower()

    @staticmethod
    def _calculate_iou(boxA, boxB):
        # box format: {x0, y0, x1, y1}
        xA = max(boxA['x0'], boxB['x0'])
        yA = max(boxA['y0'], boxB['y0'])
        xB = min(boxA['x1'], boxB['x1'])
        yB = min(boxA['y1'], boxB['y1'])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA['x1'] - boxA['x0'] + 1) * (boxA['y1'] - boxA['y0'] + 1)
        boxBArea = (boxB['x1'] - boxB['x0'] + 1) * (boxB['y1'] - boxB['y0'] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    @staticmethod
    def _is_contained(inner, outer, threshold=0.7):
        """Checks if inner box is significantly contained within outer box."""
        inter_x0 = max(inner['x0'], outer['x0'])
        inter_y0 = max(inner['y0'], outer['y0'])
        inter_x1 = min(inner['x1'], outer['x1'])
        inter_y1 = min(inner['y1'], outer['y1'])

        inter_w = max(0, inter_x1 - inter_x0)
        inter_h = max(0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h
        
        inner_area = (inner['x1'] - inner['x0']) * (inner['y1'] - inner['y0'])
        if inner_area <= 0: return False
        
        return (inter_area / inner_area) >= threshold

    @staticmethod
    def merge(mineru_data, paddle_data):
        """
        Maps PaddleOCR lines into MinerU blocks, updating the bboxes of MinerU lines
        while preserving MinerU's high-fidelity layout and markdown formatting.
        
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
            region_type = region["region_type"]
            mineru_lines = region.get("extracted_lines", [])

            # --- ROBUST ANCHORING ---
            # Find paddle lines that are significantly contained within this block
            paddle_lines_in_block = []
            for i, p_line in enumerate(paddle_lines):
                p_bbox = p_line["bbox"]
                
                # Check containment (70% threshold) or center-point fallback
                is_inside = OCRMerger._is_contained(p_bbox, block_bbox, threshold=0.6)
                
                if not is_inside:
                    cx = (p_bbox["x0"] + p_bbox["x1"]) / 2
                    cy = (p_bbox["y0"] + p_bbox["y1"]) / 2
                    is_inside = (block_bbox["x0"] <= cx <= block_bbox["x1"] and 
                                 block_bbox["y0"] <= cy <= block_bbox["y1"])
                
                if is_inside:
                    paddle_lines_in_block.append((i, p_line))

            # Case: Empty MinerU block (fallback to PaddleOCR lines)
            if not mineru_lines:
                if paddle_lines_in_block:
                    # Sort and Cluster (Standard flow)
                    sorted_by_y = sorted(paddle_lines_in_block, key=lambda x: x[1]["bbox"]["y0"])
                    rows = []
                    for idx, p_line in sorted_by_y:
                        bbox = p_line["bbox"]
                        h = bbox["y1"] - bbox["y0"]
                        placed = False
                        for row in rows:
                            row_bbox = row[0][1]["bbox"]
                            row_h = row_bbox["y1"] - row_bbox["y0"]
                            overlap = min(bbox["y1"], row_bbox["y1"]) - max(bbox["y0"], row_bbox["y0"])
                            min_h = min(h, row_h)
                            if min_h > 0 and overlap > 0.5 * min_h:
                                # COLUMN GUARD: Don't merge if horizontal gap is too large
                                row.sort(key=lambda x: x[1]["bbox"]["x0"])
                                left_x = row[0][1]["bbox"]["x0"]
                                right_x = row[-1][1]["bbox"]["x1"]
                                gap = max(0, bbox["x0"] - right_x)
                                if gap < (block_bbox["x1"] - block_bbox["x0"]) * 0.2: # 20% width gap limit
                                    row.append((idx, p_line))
                                    placed = True
                                    break
                        if not placed:
                            rows.append([(idx, p_line)])
                    
                    for row in rows: row.sort(key=lambda x: x[1]["bbox"]["x0"])
                    rows.sort(key=lambda r: sum(x[1]["bbox"]["y0"] for x in r) / len(r))
                    
                    final_lines = []
                    for row in rows:
                        for idx, p_line in row:
                            final_lines.append({
                                "text": p_line["text"],
                                "bbox": p_line["bbox"],
                                "confidence_score": p_line["confidence"],
                                "style": {
                                    "font_size": round((p_line["bbox"]["y1"] - p_line["bbox"]["y0"]) * 0.8, 2),
                                    "is_bold": region_type in ["header", "title"]
                                }
                            })
                            assigned_paddle_indices.add(idx)
                    region["extracted_lines"] = final_lines
                merged_regions.append(region)
                continue

            if not paddle_lines_in_block:
                merged_regions.append(region)
                continue

            # --- COLUMN-AWARE VISUAL CLUSTERING ---
            sorted_by_y = sorted(paddle_lines_in_block, key=lambda x: x[1]["bbox"]["y0"])
            rows = []
            for idx, p_line in sorted_by_y:
                bbox = p_line["bbox"]
                h = bbox["y1"] - bbox["y0"]
                placed = False
                for row in rows:
                    row_bbox = row[0][1]["bbox"]
                    row_h = row_bbox["y1"] - row_bbox["y0"]
                    overlap = min(bbox["y1"], row_bbox["y1"]) - max(bbox["y0"], row_bbox["y0"])
                    min_h = min(h, row_h)
                    if min_h > 0 and overlap > 0.5 * min_h:
                        row.sort(key=lambda x: x[1]["bbox"]["x0"])
                        right_x = row[-1][1]["bbox"]["x1"]
                        gap = max(0, bbox["x0"] - right_x)
                        # Only merge if gap is small (prevents multi-column merging)
                        if gap < (block_bbox["x1"] - block_bbox["x0"]) * 0.15:
                            row.append((idx, p_line))
                            placed = True
                            break
                if not placed:
                    rows.append([(idx, p_line)])
            
            for row in rows: row.sort(key=lambda x: x[1]["bbox"]["x0"])
            rows.sort(key=lambda r: sum(x[1]["bbox"]["y0"] for x in r) / len(r))
            
            # 5. Build list of merged paddle rows with IOU-based deduplication
            paddle_rows = []
            for row in rows:
                # Deduplicate lines in the row that have high IOU (over 80%)
                unique_row_lines = []
                for idx, p_line in row:
                    is_duplicate = False
                    for _, existing_p in unique_row_lines:
                        if OCRMerger._calculate_iou(p_line["bbox"], existing_p["bbox"]) > 0.8:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_row_lines.append((idx, p_line))
                
                if not unique_row_lines: continue
                
                x0 = min(x[1]["bbox"]["x0"] for x in unique_row_lines)
                y0 = min(x[1]["bbox"]["y0"] for x in unique_row_lines)
                x1 = max(x[1]["bbox"]["x1"] for x in unique_row_lines)
                y1 = max(x[1]["bbox"]["y1"] for x in unique_row_lines)
                row_text = " ".join(x[1]["text"] for x in unique_row_lines)
                row_conf = sum(x[1]["confidence"] for x in unique_row_lines) / len(unique_row_lines)
                paddle_rows.append({
                    "text": row_text,
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "confidence": row_conf,
                    "indices": [x[0] for x in unique_row_lines]
                })

            # --- RESILIENT TEXT ALIGNMENT ---
            cleaned_mineru = [OCRMerger._clean_text(m["text"]) for m in mineru_lines]
            cleaned_paddle = [OCRMerger._clean_text(p["text"]) for p in paddle_rows]
            
            mapping = {} # mineru_idx -> paddle_row_idx
            
            # Use a more flexible scoring for short strings and allow slight out-of-order 
            # for multi-column jitter if needed, but maintain sequential priority.
            last_p_idx = -1
            for m_idx, m_text in enumerate(cleaned_mineru):
                if len(m_text) < 2: continue # Skip single chars
                
                best_p_idx = None
                best_score = 0.0
                
                # Search window: check a few rows ahead to prevent getting stuck
                search_limit = min(last_p_idx + 4, len(paddle_rows))
                for p_idx in range(last_p_idx + 1, search_limit):
                    p_text = cleaned_paddle[p_idx]
                    
                    # Exact or Substring match
                    if m_text == p_text or m_text in p_text or p_text in m_text:
                        score = 0.9 + 0.1 * (min(len(m_text), len(p_text)) / max(len(m_text), len(p_text)))
                    else:
                        score = difflib.SequenceMatcher(None, m_text, p_text).ratio()
                        
                    # Heuristic boost for short strings if they are visually close
                    if len(m_text) < 5 and score > 0.6:
                        score += 0.2

                    if score > best_score and score >= 0.45:
                        best_score = score
                        best_p_idx = p_idx
                        
                if best_p_idx is not None:
                    mapping[m_idx] = best_p_idx
                    last_p_idx = best_p_idx
                    for idx in paddle_rows[best_p_idx]["indices"]:
                        assigned_paddle_indices.add(idx)

            # --- VERTICAL INTERPOLATION & FALLBACK ---
            for m_idx, m_line in enumerate(mineru_lines):
                if m_idx in mapping:
                    p_row = paddle_rows[mapping[m_idx]]
                    m_line["bbox"] = p_row["bbox"]
                    m_line["confidence_score"] = p_row["confidence"]
                else:
                    # Interpolate using surrounding matched lines
                    above_indices = [i for i in mapping if i < m_idx]
                    below_indices = [i for i in mapping if i > m_idx]
                    
                    if above_indices and below_indices:
                        a = max(above_indices)
                        b = min(below_indices)
                        bbox_a = mineru_lines[a]["bbox"]
                        bbox_b = mineru_lines[b]["bbox"]
                        factor = (m_idx - a) / (b - a)
                        y0 = round(bbox_a["y1"] + factor * (bbox_b["y0"] - bbox_a["y1"]))
                        h = (bbox_a["y1"] - bbox_a["y0"] + bbox_b["y1"] - bbox_b["y0"]) / 2
                        m_line["bbox"] = {"x0": block_bbox["x0"], "y0": y0, "x1": block_bbox["x1"], "y1": round(y0 + h)}
                    elif above_indices:
                        bbox_a = mineru_lines[max(above_indices)]["bbox"]
                        h = bbox_a["y1"] - bbox_a["y0"]
                        y0 = bbox_a["y1"] + 2
                        m_line["bbox"] = {"x0": block_bbox["x0"], "y0": y0, "x1": block_bbox["x1"], "y1": round(y0 + h)}
                    elif below_indices:
                        bbox_b = mineru_lines[min(below_indices)]["bbox"]
                        h = bbox_b["y1"] - bbox_b["y0"]
                        y1 = bbox_b["y0"] - 2
                        m_line["bbox"] = {"x0": block_bbox["x0"], "y0": round(y1 - h), "x1": block_bbox["x1"], "y1": y1}

            region["extracted_lines"] = mineru_lines
            merged_regions.append(region)

        # Orphan Harvesting (Unchanged)
        for i, p_line in enumerate(paddle_lines):
            if i not in assigned_paddle_indices:
                p_bbox = p_line["bbox"]
                merged_regions.append({
                    "region_id": f"reg-orphan-{i}",
                    "region_index": 999 + i,
                    "region_type": "text",
                    "bbox": p_bbox,
                    "confidence_score": p_line["confidence"],
                    "extracted_lines": [{
                        "text": p_line["text"],
                        "bbox": p_bbox,
                        "confidence_score": p_line["confidence"],
                        "style": {"font_size": round((p_bbox["y1"] - p_bbox["y0"]) * 0.8, 2), "is_bold": False}
                    }]
                })

        return {
            "page_width": mineru_data["page_width"],
            "page_height": mineru_data["page_height"],
            "extracted_regions": merged_regions
        }
