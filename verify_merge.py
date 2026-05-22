import asyncio
import json
import base64
import os
import sys

# Add current dir to path to import services
sys.path.append(os.path.join(os.getcwd(), 'app'))

from app.services.paddle_client import paddle_client
from app.services.merger import OCRMerger

async def verify_merger(image_path):
    print(f"🔍 Verifying OCR Merger with image: {image_path}")
    
    # 1. Get PaddleOCR data (Real call to local service)
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    print("📡 Calling PaddleOCR service...")
    paddle_data = await paddle_client.get_line_bboxes("verify-doc", 0, img_base64)
    
    if not paddle_data:
        print("❌ Failed to get data from PaddleOCR service. Is it running on port 9002?")
        return

    # 2. Create MOCK MinerU data
    # We create one large block that covers the whole page to see all lines mapped inside it.
    mock_mineru_data = {
        "page_width": paddle_data["page_width"],
        "page_height": paddle_data["page_height"],
        "extracted_regions": [
            {
                "region_id": "mineru-block-1",
                "region_index": 0,
                "region_type": "text",
                "bbox": {
                    "x0": 0,
                    "y0": 0,
                    "x1": paddle_data["page_width"],
                    "y1": paddle_data["page_height"]
                },
                "confidence_score": 0.9,
                "extracted_lines": [] # This is what we expect to be filled
            }
        ]
    }

    # 3. Perform the Merge
    print("🎯 Running OCRMerger.merge...")
    merged_result = OCRMerger.merge(mock_mineru_data, paddle_data)

    # 4. Display Results
    region = merged_result["extracted_regions"][0]
    lines = region.get("extracted_lines", [])
    
    print(f"\n✅ MERGE COMPLETE")
    print(f"📊 Original Paddle Lines: {len(paddle_data['lines'])}")
    print(f"📊 Mapped Lines in Block: {len(lines)}")
    
    if lines:
        print("\n📝 Sample of Mapped Lines (First 3):")
        for i, line in enumerate(lines[:3]):
            print(f"  [{i+1}] '{line['text']}'")
            print(f"      BBox: {line['bbox']}")
            print(f"      Style: {line['style']}")

    # Save to file for manual inspection
    output_path = "merge_verification_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_result, f, indent=2)
    print(f"\n💾 Full merged data saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_merge.py <path_to_image>")
    else:
        asyncio.run(verify_merger(sys.argv[1]))
