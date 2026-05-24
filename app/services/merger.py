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
    def _filter_redundant_regions(regions):
        """
        Removes overlapping MinerU regions by prioritizing structured content.
        Priority: table, formula, header > text
        """
        if not regions: return []
        
        # Sort by priority and then by index
        def get_priority(r_type):
            if r_type in ["table", "formula", "header", "title"]: return 0
            return 1
            
        sorted_regions = sorted(regions, key=lambda x: (get_priority(x["region_type"]), x["region_index"]))
        keep_indices = set(range(len(sorted_regions)))
        
        for i in range(len(sorted_regions)):
            if i not in keep_indices: continue
            for j in range(i + 1, len(sorted_regions)):
                if j not in keep_indices: continue
                
                iou = OCRMerger._calculate_iou(sorted_regions[i]["bbox"], sorted_regions[j]["bbox"])
                # If they overlap significantly (> 70%), discard the lower priority one
                if iou > 0.7:
                    # Since we sorted by priority, j is always lower or equal priority to i
                    keep_indices.remove(j)
        
        return [sorted_regions[i] for i in sorted(list(keep_indices))]

    @staticmethod
    def merge(mineru_data, paddle_data):
        """
        Maps PaddleOCR lines into MinerU blocks using Global Spatial Competition.
        """
        if not paddle_data or "lines" not in paddle_data:
            logger.warning("No PaddleOCR data to merge, returning original MinerU data.")
            return mineru_data

        paddle_lines = paddle_data["lines"]
        assigned_paddle_indices = set()

        # ── Step 1: Visual Row Clustering (Global) ──
        # Build physical rows for the entire page first
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
                    row.sort(key=lambda x: x[1]["bbox"]["x0"])
                    if (bbox["x0"] - row[-1][1]["bbox"]["x1"]) < (mineru_data["page_width"] * 0.1):
                        row.append((idx, p_line))
                        placed = True
                        break
            if not placed: visual_rows.append([(idx, p_line)])
        
        physical_rows = []
        for row in visual_rows:
            row.sort(key=lambda x: x[1]["bbox"]["x0"])
            row_indices = [x[0] for x in row]
            physical_rows.append({
                "text": " ".join(x[1]["text"] for x in row),
                "bbox": {
                    "x0": min(x[1]["bbox"]["x0"] for x in row),
                    "y0": min(x[1]["bbox"]["y0"] for x in row),
                    "x1": max(x[1]["bbox"]["x1"] for x in row),
                    "y1": max(x[1]["bbox"]["y1"] for x in row)
                },
                "indices": row_indices,
                "confidence": sum(x[1]["confidence"] for x in row) / len(row)
            })

        # ── Step 2: Quality Filtering ──
        # Filter out redundant MinerU regions (Table > Text)
        valid_regions = OCRMerger._filter_redundant_regions(mineru_data["extracted_regions"])

        # ── Step 3: Global Competition (Physical -> Logical) ──
        # Map every physical row to the SINGLE best MinerU region
        region_assignments = [[] for _ in range(len(valid_regions))]
        
        for p_idx, p_row in enumerate(physical_rows):
            best_r_idx = -1
            best_score = -1.0
            
            p_words = set(p_row["text"].lower().split())
            
            for r_idx, region in enumerate(valid_regions):
                m_bbox = region["bbox"]
                # 1. Spatial Score (Intersection over Physical Row Area)
                p_bbox = p_row["bbox"]
                ix0, iy0 = max(p_bbox["x0"], m_bbox["x0"]), max(p_bbox["y0"], m_bbox["y0"])
                ix1, iy1 = min(p_bbox["x1"], m_bbox["x1"]), min(p_bbox["y1"], m_bbox["y1"])
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                p_area = (p_bbox["x1"] - p_bbox["x0"]) * (p_bbox["y1"] - p_bbox["y0"])
                spatial_score = inter / float(p_area) if p_area > 0 else 0
                
                # 2. Text Score (Keyword overlap)
                m_lines = region.get("extracted_lines", [])
                m_text = " ".join(l["text"] for l in m_lines).lower()
                m_words = set(m_text.split())
                text_score = len(p_words.intersection(m_words)) / float(len(p_words)) if p_words else 0
                
                # Weighted compatibility score
                score = (spatial_score * 0.7) + (text_score * 0.3)
                if score > best_score and score > 0.4:
                    best_score = score
                    best_r_idx = r_idx
            
            if best_r_idx != -1:
                region_assignments[best_r_idx].append(p_row)
                for idx in p_row["indices"]: assigned_paddle_indices.add(idx)

        # ── Step 4: Internal Alignment (v10.0 - Visual Row Assembly) ──
        merged_regions = []
        for r_idx, region in enumerate(valid_regions):
            assigned_rows = sorted(region_assignments[r_idx], key=lambda x: x["bbox"]["y0"])
            mineru_lines = region.get("extracted_lines", [])
            
            if not mineru_lines:
                merged_regions.append(region)
                continue

            # 1. Structural Row Splicing (Table Row Splitter)
            refined_m_lines = []
            for m_line in mineru_lines:
                raw = m_line["text"]
                if "<tr" in raw or "<td" in raw:
                    for row_txt in re.split(r'</tr>|<tr>', raw):
                        clean = OCRMerger._clean_text(row_txt)
                        if len(clean) > 2: refined_m_lines.append({"text": clean, "bbox": m_line["bbox"]})
                else:
                    m_line["text"] = OCRMerger._clean_text(raw)
                    refined_m_lines.append(m_line)
            
            if not assigned_rows:
                region["extracted_lines"] = refined_m_lines
                merged_regions.append(region)
                continue

            # 2. Word Tokenization (MinerU) with Spatial Anchoring
            m_words = []
            for m_line in refined_m_lines:
                words = m_line["text"].split()
                if not words: continue
                
                # Estimate word boxes for spatial anchoring
                lb = m_line["bbox"]
                w_width = (lb["x1"] - lb["x0"]) / len(words)
                for j, w_text in enumerate(words):
                    w_bbox = {
                        "x0": lb["x0"] + (j * w_width),
                        "y0": lb["y0"],
                        "x1": lb["x0"] + ((j + 1) * w_width),
                        "y1": lb["y1"]
                    }
                    m_words.append({"text": w_text, "bbox": w_bbox})

            # 3. Geometric Bucket Assignment (Pouring words into physical rows)
            row_buckets = [[] for _ in range(len(assigned_rows))]
            for m_word in m_words:
                best_p_idx = -1
                best_ioa = 0.0
                
                for p_idx, p_row in enumerate(assigned_rows):
                    p_bbox = p_row["bbox"]
                    # Calculate IoA (Intersection over m_word Area)
                    ix0, iy0 = max(m_word["bbox"]["x0"], p_bbox["x0"]), max(m_word["bbox"]["y0"], p_bbox["y0"])
                    ix1, iy1 = min(m_word["bbox"]["x1"], p_bbox["x1"]), min(m_word["bbox"]["y1"], p_bbox["y1"])
                    
                    inter_w = max(0, ix1 - ix0)
                    inter_h = max(0, iy1 - iy0)
                    inter_area = inter_w * inter_h
                    word_area = (m_word["bbox"]["x1"] - m_word["bbox"]["x0"]) * (m_word["bbox"]["y1"] - m_word["bbox"]["y0"])
                    
                    ioa = inter_area / float(word_area) if word_area > 0 else 0
                    if ioa > best_ioa:
                        best_ioa = ioa
                        best_p_idx = p_idx
                
                # If a word is 40% inside a Paddle row, assign it
                if best_p_idx != -1 and best_ioa > 0.4:
                    row_buckets[best_p_idx].append(m_word)
                else:
                    # Vertical Proximity Fallback for small shifts
                    m_cy = (m_word["bbox"]["y0"] + m_word["bbox"]["y1"]) / 2
                    closest_p_idx = -1
                    min_dist = 99999
                    for p_idx, p_row in enumerate(assigned_rows):
                        p_cy = (p_row["bbox"]["y0"] + p_row["bbox"]["y1"]) / 2
                        dist = abs(m_cy - p_cy)
                        if dist < min_dist:
                            min_dist = dist
                            closest_p_idx = p_idx
                    if closest_p_idx != -1 and min_dist < 15:
                        row_buckets[closest_p_idx].append(m_word)

            # 4. Physical Row Assembly (Ordering and text reconstruction)
            final_lines = []
            for i, p_row in enumerate(assigned_rows):
                bucket = row_buckets[i]
                if bucket:
                    # Sort words within row LEFT-TO-RIGHT (fixes sequence desync)
                    bucket.sort(key=lambda w: w["bbox"]["x0"])
                    text = " ".join(w["text"] for w in bucket)
                else:
                    text = p_row["text"] # Fallback if no words matched
                
                if len(text.strip()) < 2: continue

                final_lines.append({
                    "text": text,
                    "bbox": p_row["bbox"],
                    "parent_bbox": region["bbox"],
                    "confidence_score": p_row["confidence"],
                    "style": {
                        "font_size": round((p_row["bbox"]["y1"] - p_row["bbox"]["y0"]) * 0.8, 2),
                        "is_bold": region["region_type"] in ["header", "title"]
                    }
                })
            
            region["extracted_lines"] = final_lines
            merged_regions.append(region)

        # ── Step 5: Orphan Harvesting ──
        for i, p_line in enumerate(paddle_lines):
            if i not in assigned_paddle_indices:
                nearest_region = min(valid_regions, key=lambda r: abs(((r["bbox"]["y0"] + r["bbox"]["y1"])/2) - ((p_line["bbox"]["y0"] + p_line["bbox"]["y1"])/2))) if valid_regions else None
                merged_regions.append({
                    "region_id": f"reg-orphan-{i}", "region_index": 2000 + i, "region_type": "text", "bbox": p_line["bbox"], "confidence_score": p_line["confidence"],
                    "extracted_lines": [{"text": p_line["text"], "bbox": p_line["bbox"], "parent_bbox": nearest_region["bbox"] if nearest_region else p_line["bbox"], "confidence_score": p_line["confidence"], "style": {"font_size": 12, "is_bold": False}}]
                })

        return {"page_width": mineru_data["page_width"], "page_height": mineru_data["page_height"], "extracted_regions": merged_regions}
