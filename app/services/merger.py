import logging
import re
import difflib

logger = logging.getLogger("procr")

class OCRMerger:
    @staticmethod
    def _clean_text(text):
        if not text:
            return ""
        # ── PILLAR C V9.0: Production HTML-to-Markdown Simplifier ──
        # 1. Convert structural table tags to readable Markdown pipes
        text = re.sub(r'<(?:tr|th|td)[^>]*>', ' | ', text)
        # 2. Strip all other HTML tags
        text = re.sub(r'<[^>]*>', ' ', text)
        # 3. Standardize whitespace and remove legacy artifacts
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
        Maps PaddleOCR lines into MinerU blocks using an Elastic Spatial Matcher.
        Prioritizes MinerU text truth and PaddleOCR visual truth.
        """
        if not paddle_data or "lines" not in paddle_data:
            logger.warning("No PaddleOCR data to merge, returning original MinerU data.")
            return mineru_data

        merged_regions = []
        paddle_lines = paddle_data["lines"]
        assigned_paddle_indices = set()

        # ── Step 1: Visual Row Clustering (Global) ──
        # We pre-cluster all PaddleOCR lines into physical rows for the entire page.
        sorted_by_y = sorted(enumerate(paddle_lines), key=lambda x: x[1]["bbox"]["y0"])
        visual_rows = []
        for idx, p_line in sorted_by_y:
            bbox = p_line["bbox"]
            placed = False
            for row in visual_rows:
                r_bbox = row[0][1]["bbox"]
                overlap = min(bbox["y1"], r_bbox["y1"]) - max(bbox["y0"], r_bbox["y0"])
                h = min(bbox["y1"] - bbox["y0"], r_bbox["y1"] - r_bbox["y0"])
                if h > 0 and overlap > 0.5 * h:
                    # Column Guard: check horizontal gap
                    row.sort(key=lambda x: x[1]["bbox"]["x0"])
                    right_x = row[-1][1]["bbox"]["x1"]
                    if (bbox["x0"] - right_x) < (mineru_data["page_width"] * 0.1):
                        row.append((idx, p_line))
                        placed = True
                        break
            if not placed:
                visual_rows.append([(idx, p_line)])
        
        # Build flattened physical rows with merged text and unified bboxes
        physical_rows = []
        for row in visual_rows:
            row.sort(key=lambda x: x[1]["bbox"]["x0"])
            x0 = min(x[1]["bbox"]["x0"] for x in row)
            y0 = min(x[1]["bbox"]["y0"] for x in row)
            x1 = max(x[1]["bbox"]["x1"] for x in row)
            y1 = max(x[1]["bbox"]["y1"] for x in row)
            physical_rows.append({
                "text": " ".join(x[1]["text"] for x in row),
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "indices": [x[0] for x in row],
                "confidence": sum(x[1]["confidence"] for x in row) / len(row)
            })

        # ── Step 2: Semantic Alignment (Elastic) ──
        for region in mineru_data["extracted_regions"]:
            block_bbox = region["bbox"]
            region_type = region["region_type"]
            mineru_lines = region.get("extracted_lines", [])
            if not mineru_lines:
                merged_regions.append(region)
                continue

            # ── PILLAR C V9.1: Mega-Block Table Slicer ──
            # If a block is a giant table, MinerU often outputs it as one line.
            # we force-split it into rows using structural tags for better citation granularity.
            refined_mineru_lines = []
            for m_line in mineru_lines:
                raw_txt = m_line["text"]
                if "<tr" in raw_txt or "<td" in raw_txt:
                    # Structural Split: each table row becomes a separate logical line
                    rows = re.split(r'</tr>|<tr>', raw_txt)
                    for r_txt in rows:
                        clean_r = OCRMerger._clean_text(r_txt)
                        if len(clean_r) > 2:
                            refined_mineru_lines.append({"text": clean_r, "bbox": m_line["bbox"]})
                else:
                    m_line["text"] = OCRMerger._clean_text(raw_txt)
                    refined_mineru_lines.append(m_line)
            
            mineru_lines = refined_mineru_lines

            # Tokenize block text to find physical anchors
            full_block_text = " ".join(m["text"] for m in mineru_lines)
            tokens = [t for t in full_block_text.split() if len(t) > 3] # Unique anchors
            
            # Find physical rows that contain these tokens
            potential_row_indices = []
            for i, p_row in enumerate(physical_rows):
                p_text_lower = p_row["text"].lower()
                matches = sum(1 for t in tokens if t.lower() in p_text_lower)
                if matches > 0:
                    potential_row_indices.append((i, matches))
            
            # Identify the visual cluster (True Location)
            # We look for a contiguous "Envelope" of rows that contains our words.
            if potential_row_indices:
                # 1. Find the median visual center of all matched rows (the 'Anchor Point')
                all_matched_indices = [idx for idx, count in potential_row_indices]
                avg_y = sum(physical_rows[i]["bbox"]["y0"] for i in all_matched_indices) / len(all_matched_indices)
                
                # 2. Find the contiguous range of physical rows closest to this Anchor Point
                # This ensures we don't pick up rows from the top of the page if the block is at the bottom.
                start_row_idx = min(all_matched_indices, key=lambda i: abs(physical_rows[i]["bbox"]["y0"] - avg_y))
                
                # Select an envelope around the start point (± N rows based on MinerU line count)
                buffer = 2
                range_start = max(0, start_row_idx - buffer)
                range_end = min(len(physical_rows), start_row_idx + len(mineru_lines) + buffer)
                target_rows = physical_rows[range_start:range_end]
            else:
                # FALLBACK: Use MinerU's bbox if no text anchors are found (e.g. symbols only)
                target_rows = [r for r in physical_rows if OCRMerger._is_contained(r["bbox"], block_bbox, threshold=0.4)]
                if not target_rows: # Last resort: nearest neighbor
                    m_cy = (block_bbox["y0"] + block_bbox["y1"]) / 2
                    target_rows = sorted(physical_rows, key=lambda r: abs(((r["bbox"]["y0"] + r["bbox"]["y1"])/2) - m_cy))[:1]

            # ── Step 3: Sequential-Spatial Alignment (v8.0 - Anti-Shattering) ──
            # We map MinerU text to the physical rows in the 'Envelope' found in Step 2.
            # This preserves 100% of the text order and prevents repetitive 'Word Theft'.
            m_full_text = " ".join(m["text"] for m in mineru_lines)
            p_full_text = " ".join(r["text"] for r in target_rows)
            
            # Map target rows to character offsets in p_full_text
            p_offsets = []
            curr_offset = 0
            for r in target_rows:
                start = p_full_text.find(r["text"], curr_offset)
                if start == -1: start = curr_offset
                end = start + len(r["text"])
                p_offsets.append((start, end))
                curr_offset = end

            # Use character-level sequence matcher ONLY within the verified cluster
            matcher = difflib.SequenceMatcher(None, m_full_text, p_full_text, autojunk=False)
            matching_blocks = matcher.get_matching_blocks()
            
            new_lines = []
            for i, (p_start, p_end) in enumerate(p_offsets):
                p_row = target_rows[i]
                m_start, m_end = None, None
                
                for b in matching_blocks:
                    # find intersection between this match and the current paddle row range
                    overlap_p_start = max(b.b, p_start)
                    overlap_p_end = min(b.b + b.size, p_end)
                    if overlap_p_start < overlap_p_end:
                        offset_in_block = overlap_p_start - b.b
                        overlap_len = overlap_p_end - overlap_p_start
                        curr_m_start = b.a + offset_in_block
                        curr_m_end = curr_m_start + overlap_len
                        if m_start is None or curr_m_start < m_start: m_start = curr_m_start
                        if m_end is None or curr_m_end > m_end: m_end = curr_m_end
                
                # Extract the high-fidelity MinerU slice for this physical line
                if m_start is not None and m_end is not None and (m_end - m_start) > 2:
                    text = m_full_text[m_start:m_end].strip()
                    if len(text) < 2: text = p_row["text"] # Fallback if slice is junk
                else:
                    text = p_row["text"]

                new_lines.append({
                    "text": text,
                    "bbox": p_row["bbox"],
                    "parent_bbox": block_bbox,
                    "confidence_score": p_row["confidence"],
                    "style": {
                        "font_size": round((p_row["bbox"]["y1"] - p_row["bbox"]["y0"]) * 0.8, 2),
                        "is_bold": region_type in ["header", "title"]
                    }
                })
                for idx in p_row["indices"]: assigned_paddle_indices.add(idx)

            region["extracted_lines"] = new_lines
            merged_regions.append(region)

        # ── Step 4: Orphan Harvesting with Context ──
        for i, p_line in enumerate(paddle_lines):
            if i not in assigned_paddle_indices:
                p_bbox = p_line["bbox"]
                # Guard: find logical parent box (nearest neighbor region)
                if mineru_data["extracted_regions"]:
                    nearest_region = min(mineru_data["extracted_regions"], 
                                         key=lambda r: abs(((r["bbox"]["y0"] + r["bbox"]["y1"])/2) - ((p_bbox["y0"] + p_bbox["y1"])/2)))
                    parent_bbox = nearest_region["bbox"]
                else:
                    parent_bbox = p_bbox
                    
                merged_regions.append({
                    "region_id": f"reg-orphan-{i}",
                    "region_index": 1000 + i,
                    "region_type": "text",
                    "bbox": p_bbox,
                    "confidence_score": p_line["confidence"],
                    "extracted_lines": [{
                        "text": p_line["text"],
                        "bbox": p_line["bbox"],
                        "parent_bbox": parent_bbox,
                        "confidence_score": p_line["confidence"],
                        "style": {"font_size": 12, "is_bold": False}
                    }]
                })

        return {
            "page_width": mineru_data["page_width"],
            "page_height": mineru_data["page_height"],
            "extracted_regions": merged_regions
        }
