import os
import sys

# Add app folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.adapter import MinerUAdapter
from app.services.merger import OCRMerger

def test_mineru_adapter_scaling():
    print("🧪 Testing MinerUAdapter coordinate scaling...")
    
    # Mock MinerU 2.5 Pro VLM output using downscaled coordinates
    # thumbnail size: 512x680, original size: 1024x1360 (exactly 2x scaling)
    mock_mineru_output = [
        {
            "type": "title",
            "bbox": [50, 100, 250, 150], # in 512x680 space
            "content": "Document Title",
            "confidence": 0.99
        },
        {
            "type": "text",
            "bbox": [50, 200, 450, 300], # in 512x680 space
            "content": "Paragraph line 1\nParagraph line 2",
            "confidence": 0.95
        }
    ]
    
    page_width, page_height = 1024, 1360
    mineru_width, mineru_height = 512, 680
    
    result = MinerUAdapter.transform(
        mock_mineru_output, 
        page_width, 
        page_height, 
        mineru_width=mineru_width, 
        mineru_height=mineru_height
    )
    
    # Assertions
    title_region = result["extracted_regions"][0]
    text_region = result["extracted_regions"][1]
    
    # Title region bbox must be scaled by 2x
    assert title_region["bbox"] == {"x0": 100, "y0": 200, "x1": 500, "y1": 300}, f"Failed scaling title: {title_region['bbox']}"
    # Text region bbox must be scaled by 2x
    assert text_region["bbox"] == {"x0": 100, "y0": 400, "x1": 900, "y1": 600}, f"Failed scaling text: {text_region['bbox']}"
    
    print("✅ MinerUAdapter coordinate scaling works perfectly!")

def test_ocr_merger_sorting():
    print("🧪 Testing OCRMerger reading order sorting...")
    
    # Original page size
    page_width, page_height = 1000, 1000
    
    # Create a single large block covering the page
    mineru_data = {
        "page_width": page_width,
        "page_height": page_height,
        "extracted_regions": [
            {
                "region_id": "reg-1",
                "region_index": 0,
                "region_type": "text",
                "bbox": {"x0": 50, "y0": 50, "x1": 950, "y1": 950},
                "confidence_score": 0.95,
                "extracted_lines": []
            }
        ]
    }
    
    # PaddleOCR lines (Shuffled to test sorting robustness)
    # We have:
    # 1. Centered header: y=100-120, x=400-600
    # 2. Left indent bullet: y=150-170, x=150-500
    # 3. Normal paragraph: y=200-220, x=100-900
    # 4. Multi-part inline line (e.g. key-value on same row):
    #    4a (Left key): y=250-270, x=100-300
    #    4b (Right value): y=250-270, x=500-700
    paddle_data = {
        "page_width": page_width,
        "page_height": page_height,
        "lines": [
            {
                "text": "Paragraph text line.",
                "bbox": {"x0": 100, "y0": 200, "x1": 900, "y1": 220},
                "confidence": 0.95
            },
            {
                "text": "Key:",
                "bbox": {"x0": 100, "y0": 250, "x1": 300, "y1": 270},
                "confidence": 0.94
            },
            {
                "text": "Value Description",
                "bbox": {"x0": 500, "y0": 250, "x1": 700, "y1": 270},
                "confidence": 0.93
            },
            {
                "text": "Centered Main Title Header",
                "bbox": {"x0": 400, "y0": 100, "x1": 600, "y1": 120},
                "confidence": 0.99
            },
            {
                "text": "- Bullet item list",
                "bbox": {"x0": 150, "y0": 150, "x1": 500, "y1": 170},
                "confidence": 0.97
            }
        ]
    }
    
    merged_result = OCRMerger.merge(mineru_data, paddle_data)
    
    lines = merged_result["extracted_regions"][0]["extracted_lines"]
    
    # Verify exact length
    assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"
    
    # Print the sorted output
    print("--- Sorted Result ---")
    for i, line in enumerate(lines):
        print(f"[{i}] {line['text']} (x0={line['bbox']['x0']}, y0={line['bbox']['y0']})")
    print("---------------------")
    
    # Expected ordering:
    # 0: Centered Main Title Header (y0=100)
    # 1: - Bullet item list (y0=150)
    # 2: Paragraph text line. (y0=200)
    # 3: Key: (y0=250, x0=100)
    # 4: Value Description (y0=250, x0=500)
    
    assert lines[0]["text"] == "Centered Main Title Header"
    assert lines[1]["text"] == "- Bullet item list"
    assert lines[2]["text"] == "Paragraph text line."
    assert lines[3]["text"] == "Key:"
    assert lines[4]["text"] == "Value Description"
    
    print("✅ OCRMerger reading order sorting works perfectly!")

def test_filter_redundant_regions_rules():
    print("🧪 Testing OCRMerger._filter_redundant_regions strict rules...")
    
    # 1. Test Rule 1: Strict Overlap Purge
    # Table (high priority) at [100, 100, 300, 300] (area 40000)
    # Text (low priority) at [150, 150, 250, 250] (area 10000), which is completely inside Table, so overlap >= 30%
    regions_r1 = [
        {
            "region_id": "table-1",
            "region_index": 0,
            "region_type": "table",
            "bbox": {"x0": 100, "y0": 100, "x1": 300, "y1": 300},
            "extracted_lines": []
        },
        {
            "region_id": "text-overlap",
            "region_index": 1,
            "region_type": "text",
            "bbox": {"x0": 150, "y0": 150, "x1": 250, "y1": 250},
            "extracted_lines": []
        }
    ]
    filtered_r1 = OCRMerger._filter_redundant_regions(regions_r1)
    # The text region should be purged, leaving only the table
    assert len(filtered_r1) == 1, f"Rule 1 failed: expected 1 region, got {len(filtered_r1)}"
    assert filtered_r1[0]["region_id"] == "table-1", f"Rule 1 failed: expected table-1, got {filtered_r1[0]['region_id']}"
    
    # 2. Test Rule 2: Aggressive Content Hash
    # Two text regions with similar content (similarity >= 0.7)
    regions_r2 = [
        {
            "region_id": "text-first",
            "region_index": 0,
            "region_type": "text",
            "bbox": {"x0": 100, "y0": 100, "x1": 200, "y1": 150},
            "extracted_lines": [{"text": "this is a very similar piece of text content for testing"}]
        },
        {
            "region_id": "text-second",
            "region_index": 1,
            "region_type": "text",
            "bbox": {"x0": 500, "y0": 500, "x1": 600, "y1": 550},
            "extracted_lines": [{"text": "this is a very similar piece of text content for testing here"}]
        }
    ]
    filtered_r2 = OCRMerger._filter_redundant_regions(regions_r2)
    # The second one should be removed due to content similarity >= 0.7
    assert len(filtered_r2) == 1, f"Rule 2 failed: expected 1 region, got {len(filtered_r2)}"
    assert filtered_r2[0]["region_id"] == "text-first", f"Rule 2 failed: expected text-first, got {filtered_r2[0]['region_id']}"
    
    # 3. Test Rule 3: Horizontal Containment
    # Table at [100, 100, 300, 200] (width 200)
    # Layout block in same vertical column (shares x coordinates [100, 300]) but different y [500, 600]
    regions_r3 = [
        {
            "region_id": "table-col",
            "region_index": 0,
            "region_type": "table",
            "bbox": {"x0": 100, "y0": 100, "x1": 300, "y1": 200},
            "extracted_lines": []
        },
        {
            "region_id": "layout-container",
            "region_index": 1,
            "region_type": "layout",
            "bbox": {"x0": 90, "y0": 500, "x1": 310, "y1": 600}, # width 220, overlap is 200 (>= 30% of table width)
            "extracted_lines": []
        }
    ]
    filtered_r3 = OCRMerger._filter_redundant_regions(regions_r3)
    # The layout container region should be purged, leaving only the table
    assert len(filtered_r3) == 1, f"Rule 3 failed: expected 1 region, got {len(filtered_r3)}"
    assert filtered_r3[0]["region_id"] == "table-col", f"Rule 3 failed: expected table-col, got {filtered_r3[0]['region_id']}"
    
    print("✅ OCRMerger._filter_redundant_regions strict rules work perfectly!")

if __name__ == "__main__":
    try:
        test_mineru_adapter_scaling()
        print()
        test_ocr_merger_sorting()
        print()
        test_filter_redundant_regions_rules()
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        sys.exit(1)
