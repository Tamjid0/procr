import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger("procr")

MAX_INK_GAP_PIXELS = 2
MIN_INK_RATIO = 0.04
LINE_INK_RATIO = 0.15


def _runs(values: list[bool], max_gap: int = 0) -> list[tuple[int, int]]:
    """Return contiguous true-runs, bridging short antialiasing gaps."""
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    gap = 0
    for index, active in enumerate(values):
        if active:
            if start is None:
                start = index
            gap = 0
            continue
        if start is None:
            continue
        gap += 1
        if gap > max_gap:
            runs.append((start, index - gap + 1))
            start = None
            gap = 0
    if start is not None:
        runs.append((start, len(values) - gap))
    return runs


def _line_bboxes_from_pixels(
    image: Optional[Image.Image],
    bbox: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
) -> list[dict[str, int]]:
    """Detect text rows and their horizontal bounds inside a model block."""
    if image is None or page_width <= 0 or page_height <= 0:
        return []

    x0, y0, x1, y1 = bbox
    left = max(0, min(image.width, round(x0 / page_width * image.width)))
    top = max(0, min(image.height, round(y0 / page_height * image.height)))
    right = max(left + 1, min(image.width, round(x1 / page_width * image.width)))
    bottom = max(top + 1, min(image.height, round(y1 / page_height * image.height)))
    crop = image.crop((left, top, right, bottom)).convert("L")
    pixels = list(crop.getdata())
    if not pixels:
        return []

    sorted_pixels = sorted(pixels)
    background = sorted_pixels[min(len(sorted_pixels) - 1, int(len(sorted_pixels) * 0.8))]
    # The page image may have gray paper texture. Keep only dark text ink;
    # using the background minus a large margin prevents texture from making
    # every row appear active.
    threshold = max(70, min(150, background - 100))
    width, height = crop.size
    dark = [pixel < threshold for pixel in pixels]
    row_counts = [sum(dark[row * width:(row + 1) * width]) for row in range(height)]
    row_threshold = max(2, round(width * MIN_INK_RATIO))
    row_runs = _runs([count >= row_threshold for count in row_counts], MAX_INK_GAP_PIXELS)
    if not row_runs:
        return []

    scale_x = page_width / image.width
    scale_y = page_height / image.height
    detected: list[dict[str, int]] = []
    for run_top, run_bottom in row_runs:
        run_height = max(1, run_bottom - run_top)
        column_counts = [
            sum(dark[row * width + column] for row in range(run_top, run_bottom))
            for column in range(width)
        ]
        column_threshold = max(2, round(run_height * LINE_INK_RATIO))
        column_runs = _runs([count >= column_threshold for count in column_counts], 1)
        if not column_runs:
            continue
        text_left = min(run[0] for run in column_runs)
        text_right = max(run[1] for run in column_runs)
        detected.append({
            "x0": round((left + text_left) * scale_x),
            "y0": round((top + run_top) * scale_y),
            "x1": round((left + text_right) * scale_x),
            "y1": round((top + run_bottom) * scale_y),
        })
    return detected


def _wrap_text_to_lines(text: str, line_bboxes: list[dict[str, int]]) -> list[str]:
    """Assign block words to measured rows using each row's pixel width."""
    line_count = len(line_bboxes)
    explicit_lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(explicit_lines) == line_count:
        return explicit_lines

    words = text.replace("\n", " ").split()
    if line_count <= 1 or not words:
        return [text.strip()]

    lines: list[str] = []
    remaining = words[:]
    total_text_length = sum(len(word) for word in words) + max(0, len(words) - 1)
    total_line_width = max(1, sum(max(1, bbox["x1"] - bbox["x0"]) for bbox in line_bboxes))
    for line_index in range(line_count):
        lines_left = line_count - line_index
        if lines_left == 1:
            lines.append(" ".join(remaining))
            break

        line_width = max(1, line_bboxes[line_index]["x1"] - line_bboxes[line_index]["x0"])
        target = max(1, round(total_text_length * line_width / total_line_width))
        current: list[str] = []
        current_length = 0
        while remaining and (not current or current_length + len(remaining[0]) + 1 <= target):
            if len(remaining) <= lines_left - 1:
                break
            word = remaining.pop(0)
            current.append(word)
            current_length += len(word) + 1
        lines.append(" ".join(current))
    return lines

class MinerUAdapter:
    @staticmethod
    def transform(mineru_output, page_width, page_height, page_image: Optional[Image.Image] = None):
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
            
            block_bbox = (x0, y0, x1, y1)
            line_bboxes = _line_bboxes_from_pixels(page_image, block_bbox, page_width, page_height)

            # Map block text to measured visual rows. If the image projection
            # cannot find rows, retain one accurate block bbox instead of
            # inventing equal-height line geometry.
            if content.strip():
                lines = _wrap_text_to_lines(content, line_bboxes) if line_bboxes else [content.strip()]
                geometry_source = "pixel_projection" if line_bboxes else "block"
                geometries = line_bboxes or [{"x0": x0, "y0": y0, "x1": x1, "y1": y1}]
                for line_text, line_bbox in zip(lines, geometries):
                    line_h = max(1, line_bbox["y1"] - line_bbox["y0"])
                    
                    region["extracted_lines"].append({
                        "text": line_text,
                        "bbox": line_bbox,
                        "confidence_score": float(confidence),
                        "geometry_source": geometry_source,
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
