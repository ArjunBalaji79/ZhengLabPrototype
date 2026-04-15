"""Image manager for loading, annotating, and generating visual feedback.

Handles:
- Loading sample images and their metadata
- Drawing bounding box annotations on images
- Generating heatmap overlays for visual feedback
- Side-by-side comparison views
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO

from config import SAMPLE_IMAGES_DIR, DATA_DIR


@dataclass
class MedicalImage:
    """Represents a medical image with its metadata."""

    image_id: str
    filepath: Path
    category: str
    subcategory: str
    ground_truth: str
    annotations: list[dict]  # List of bounding boxes: {x, y, w, h, label}
    difficulty: str = "medium"

    def load(self) -> Image.Image:
        return Image.open(self.filepath).convert("RGB")



class ImageManager:
    """Manages medical image loading and visual feedback generation."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.sample_dir = data_dir / "sample_images"
        self.images: list[MedicalImage] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load image metadata from JSON file or generate from directory."""
        metadata_path = self.data_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
            for item in data.get("images", []):
                filepath = self.sample_dir / item["filename"]
                if filepath.exists():
                    self.images.append(
                        MedicalImage(
                            image_id=item["id"],
                            filepath=filepath,
                            category=item["category"],
                            subcategory=item.get("subcategory", "general"),
                            ground_truth=item["ground_truth"],
                            annotations=item.get("annotations", []),
                            difficulty=item.get("difficulty", "medium"),
                        )
                    )

        # If no metadata, create demo entries from any images found
        if not self.images and self.sample_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in sorted(self.sample_dir.glob(ext)):
                    self.images.append(
                        MedicalImage(
                            image_id=img_path.stem,
                            filepath=img_path,
                            category="fracture" if "fracture" in img_path.stem else
                                     "dental" if "dental" in img_path.stem else
                                     "chest_pathology",
                            subcategory="general",
                            ground_truth="Potential pathology — radiologist review required",
                            annotations=[],
                            difficulty="medium",
                        )
                    )

    def get_images_by_category(self, category: str) -> list[MedicalImage]:
        """Get all images in a specific category."""
        return [img for img in self.images if img.category == category]

    def get_random_image(self, category: Optional[str] = None) -> Optional[MedicalImage]:
        """Get a random image, optionally filtered by category."""
        pool = self.images
        if category:
            pool = self.get_images_by_category(category)
        if not pool:
            return None
        return np.random.choice(pool)

    def get_available_categories(self) -> list[str]:
        """Get list of categories that have images available."""
        return list(set(img.category for img in self.images))

    def annotate_image(
        self,
        image: MedicalImage,
        show_boxes: bool = True,
        show_labels: bool = True,
        box_color: str = "lime",
        box_width: int = 3,
    ) -> Image.Image:
        """Draw bounding box annotations on an image.

        Returns a new PIL Image with annotations overlaid.
        """
        img = image.load()
        draw = ImageDraw.Draw(img)

        if show_boxes and image.annotations:
            for ann in image.annotations:
                x, y, w, h = ann["x"], ann["y"], ann["w"], ann["h"]
                label = ann.get("label", "Finding")

                # Draw bounding box
                draw.rectangle(
                    [x, y, x + w, y + h],
                    outline=box_color,
                    width=box_width,
                )

                if show_labels:
                    # Draw label background
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                    except (OSError, IOError):
                        font = ImageFont.load_default()

                    bbox = draw.textbbox((x, y - 20), label, font=font)
                    draw.rectangle(bbox, fill=box_color)
                    draw.text((x, y - 20), label, fill="black", font=font)

        return img

    def generate_heatmap_overlay(
        self,
        image: MedicalImage,
        regions: list[dict] = None,
        intensity: float = 0.4,
    ) -> Image.Image:
        """Generate a heatmap overlay highlighting key regions.

        Args:
            image: The medical image
            regions: List of dicts with {x, y, w, h, weight} for focus areas
            intensity: Overlay opacity (0-1)

        Returns a PIL Image with heatmap overlay.
        """
        img = image.load()
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # Create heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)

        if regions:
            for region in regions:
                rx, ry = region.get("x", w // 4), region.get("y", h // 4)
                rw, rh = region.get("w", w // 2), region.get("h", h // 2)
                weight = region.get("weight", 1.0)

                # Create Gaussian blob for each region
                cy, cx = ry + rh // 2, rx + rw // 2
                Y, X = np.ogrid[:h, :w]
                sigma_y, sigma_x = rh / 2, rw / 2
                gaussian = np.exp(
                    -((X - cx) ** 2 / (2 * sigma_x ** 2) + (Y - cy) ** 2 / (2 * sigma_y ** 2))
                )
                heatmap += gaussian * weight
        elif image.annotations:
            # Use annotations as regions
            for ann in image.annotations:
                cx = ann["x"] + ann["w"] // 2
                cy = ann["y"] + ann["h"] // 2
                sigma_x, sigma_y = ann["w"] / 2, ann["h"] / 2
                Y, X = np.ogrid[:h, :w]
                gaussian = np.exp(
                    -((X - cx) ** 2 / (2 * sigma_x ** 2) + (Y - cy) ** 2 / (2 * sigma_y ** 2))
                )
                heatmap += gaussian
        else:
            # Default: center-weighted heatmap
            Y, X = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            heatmap = np.exp(
                -((X - cx) ** 2 / (2 * (w / 4) ** 2) + (Y - cy) ** 2 / (2 * (h / 4) ** 2))
            )

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Apply colormap
        colored_heatmap = cm.jet(heatmap)[:, :, :3]  # RGB only
        colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

        # Blend with original
        blended = (
            img_array * (1 - intensity) + colored_heatmap * intensity
        ).astype(np.uint8)

        return Image.fromarray(blended)

    def score_student_annotations(
        self,
        image: MedicalImage,
        student_rects: list[dict],
    ) -> dict:
        """Score student-drawn rectangles against ground truth bounding boxes.

        Uses IoU (Intersection over Union) to measure overlap.
        Returns a dict with per-box scores, overall score, and hit/miss details.
        """
        gt_boxes = image.annotations
        if not gt_boxes:
            # No ground truth boxes — score based on whether student drew nothing
            if not student_rects:
                return {
                    "annotation_score": 1.0,
                    "hits": [],
                    "misses": [],
                    "false_positives": len(student_rects),
                    "detail": "No abnormalities present — correct to mark nothing.",
                }
            return {
                "annotation_score": max(0.0, 1.0 - 0.25 * len(student_rects)),
                "hits": [],
                "misses": [],
                "false_positives": len(student_rects),
                "detail": "This image has no abnormalities, but you marked regions.",
            }

        if not student_rects:
            return {
                "annotation_score": 0.0,
                "hits": [],
                "misses": [b.get("label", "Finding") for b in gt_boxes],
                "false_positives": 0,
                "detail": "You did not mark any regions. Try clicking on areas you think are abnormal.",
            }

        # Match each GT box to the best overlapping student rect
        hits = []
        misses = []
        matched_student = set()

        for gt in gt_boxes:
            gx, gy, gw, gh = gt["x"], gt["y"], gt["w"], gt["h"]
            best_iou = 0.0
            best_idx = -1
            for i, sr in enumerate(student_rects):
                sx, sy, sw, sh = sr["x"], sr["y"], sr["w"], sr["h"]
                iou = self._compute_iou(gx, gy, gw, gh, sx, sy, sw, sh)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            label = gt.get("label", "Finding")
            # Threshold: any overlap > 0.1 counts as a hit (generous for education)
            if best_iou > 0.1:
                hits.append({"label": label, "iou": round(best_iou, 2)})
                matched_student.add(best_idx)
            else:
                misses.append(label)

        false_positives = len(student_rects) - len(matched_student)

        # Score: proportion of GT boxes hit, penalize false positives slightly
        hit_score = len(hits) / len(gt_boxes) if gt_boxes else 1.0
        fp_penalty = min(0.2, 0.05 * false_positives)
        annotation_score = max(0.0, round(hit_score - fp_penalty, 2))

        parts = []
        if hits:
            labels = ", ".join(h["label"] for h in hits)
            parts.append(f"You correctly identified: {labels}.")
        if misses:
            parts.append(f"You missed: {', '.join(misses)}.")
        if false_positives > 0:
            parts.append(f"{false_positives} region(s) you marked did not match any finding.")

        return {
            "annotation_score": annotation_score,
            "hits": hits,
            "misses": misses,
            "false_positives": false_positives,
            "detail": " ".join(parts),
        }

    @staticmethod
    def _compute_iou(x1, y1, w1, h1, x2, y2, w2, h2) -> float:
        """Compute Intersection over Union between two rectangles."""
        # Convert to (left, top, right, bottom)
        l1, t1, r1, b1 = x1, y1, x1 + w1, y1 + h1
        l2, t2, r2, b2 = x2, y2, x2 + w2, y2 + h2

        inter_l = max(l1, l2)
        inter_t = max(t1, t2)
        inter_r = min(r1, r2)
        inter_b = min(b1, b2)

        if inter_r <= inter_l or inter_b <= inter_t:
            return 0.0

        inter_area = (inter_r - inter_l) * (inter_b - inter_t)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def annotate_with_student_marks(
        self,
        image: MedicalImage,
        student_rects: list[dict],
    ) -> Image.Image:
        """Draw both ground truth (green) and student marks (yellow) on the image."""
        img = self.annotate_image(image, box_color="lime")
        draw = ImageDraw.Draw(img)

        for sr in student_rects:
            x, y, w, h = sr["x"], sr["y"], sr["w"], sr["h"]
            draw.rectangle(
                [x, y, x + w, y + h],
                outline="yellow",
                width=2,
            )
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except (OSError, IOError):
                font = ImageFont.load_default()
            draw.text((x, y - 15), "Your mark", fill="yellow", font=font)

        return img

    def create_comparison_view(
        self,
        image: MedicalImage,
        title_left: str = "Original",
        title_right: str = "Key Findings",
    ) -> Image.Image:
        """Create a side-by-side comparison: original vs annotated.

        Returns a PIL Image with both views.
        """
        original = image.load()
        annotated = self.annotate_image(image)

        # Create side-by-side canvas
        w, h = original.size
        canvas = Image.new("RGB", (w * 2 + 20, h + 40), color=(30, 30, 30))

        # Paste images
        canvas.paste(original, (0, 40))
        canvas.paste(annotated, (w + 20, 40))

        # Add titles
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw.text((w // 2 - 30, 10), title_left, fill="white", font=font)
        draw.text((w + 20 + w // 2 - 50, 10), title_right, fill="lime", font=font)

        return canvas
