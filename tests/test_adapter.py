import unittest

from PIL import Image, ImageDraw

from app.services.adapter import MinerUAdapter, _line_bboxes_from_pixels


def make_page() -> Image.Image:
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 180, 29), fill="black")
    draw.rectangle((20, 40, 150, 49), fill="black")
    return image


class AdapterLineGeometryTests(unittest.TestCase):
    def test_projection_detects_rows_and_text_widths(self) -> None:
        rows = _line_bboxes_from_pixels(make_page(), (0, 0, 200, 100), 200, 100)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"x0": 20, "y0": 20, "x1": 181, "y1": 30})
        self.assertEqual(rows[1], {"x0": 20, "y0": 40, "x1": 151, "y1": 50})

    def test_transform_maps_text_to_measured_rows(self) -> None:
        result = MinerUAdapter.transform(
            [{"type": "text", "bbox": [0, 0, 1, 1], "content": "First visual line Second visual line"}],
            200,
            100,
            make_page(),
        )

        lines = result["extracted_regions"][0]["extracted_lines"]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["geometry_source"], "pixel_projection")
        self.assertEqual(lines[0]["bbox"], {"x0": 20, "y0": 20, "x1": 181, "y1": 30})
        self.assertEqual(lines[1]["bbox"], {"x0": 20, "y0": 40, "x1": 151, "y1": 50})

    def test_transform_preserves_block_when_image_is_unavailable(self) -> None:
        result = MinerUAdapter.transform(
            [{"type": "text", "bbox": [0, 0, 1, 1], "content": "A long paragraph"}],
            200,
            100,
        )

        lines = result["extracted_regions"][0]["extracted_lines"]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["geometry_source"], "block")
        self.assertEqual(lines[0]["bbox"], {"x0": 0, "y0": 0, "x1": 200, "y1": 100})


if __name__ == "__main__":
    unittest.main()
