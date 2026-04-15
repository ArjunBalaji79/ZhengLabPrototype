"""
MedTeach AI — Image Processor & Metadata Generator

Processes locally available medical X-ray datasets into the format
needed by the prototype app. Uses real annotations from dataset CSVs
and YOLO label files to generate accurate metadata.json with bounding boxes.

Datasets:
1. VinBigData Chest X-ray (DICOM + train.csv with bounding boxes)
2. GRAZPEDWRI-DX (PNG + YOLO labels with pixel-perfect bounding boxes)

Usage:
    python download_images.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_images"
METADATA_PATH = DATA_DIR / "metadata.json"
VINBIG_DIR = BASE_DIR / "vinbigdata"
GRAPE_DIR = BASE_DIR / "grape"

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Max dimension — images are scaled to fit within this while preserving
# aspect ratio. No squashing.
MAX_DIM = 600


# ── Helpers ─────────────────────────────────────────────────────────────────

def resize_preserve_aspect(img: Image.Image) -> Image.Image:
    """Resize image so its longest side is MAX_DIM, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= MAX_DIM:
        return img
    if w >= h:
        new_w = MAX_DIM
        new_h = int(h * MAX_DIM / w)
    else:
        new_h = MAX_DIM
        new_w = int(w * MAX_DIM / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def dicom_to_jpeg(dicom_path: Path, dest: Path) -> tuple[int, int, int, int]:
    """Convert DICOM to JPEG preserving aspect ratio.
    Returns (orig_w, orig_h, new_w, new_h)."""
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(np.float32)

    # Normalize to 0-255
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0

    # Invert if MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255.0 - arr

    orig_h, orig_w = arr.shape
    img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    img = resize_preserve_aspect(img)
    new_w, new_h = img.size
    img.save(dest, "JPEG", quality=92)
    return orig_w, orig_h, new_w, new_h


def png_to_jpeg(png_path: Path, dest: Path) -> tuple[int, int, int, int]:
    """Convert PNG to resized JPEG preserving aspect ratio.
    Handles 16-bit grayscale medical images correctly.
    Returns (orig_w, orig_h, new_w, new_h)."""
    img = Image.open(png_path)
    orig_w, orig_h = img.size

    # Handle 16-bit grayscale (common in medical imaging)
    if img.mode in ("I;16", "I"):
        arr = np.array(img, dtype=np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max() * 255.0
        img = Image.fromarray(arr.astype(np.uint8), mode="L")

    img = img.convert("RGB")
    img = resize_preserve_aspect(img)
    new_w, new_h = img.size
    img.save(dest, "JPEG", quality=92)
    return orig_w, orig_h, new_w, new_h


def scale_bbox_abs(x_min, y_min, x_max, y_max, orig_w, orig_h, new_w, new_h):
    """Scale bounding box from original pixel coords to new pixel coords.
    Returns dict with {x, y, w, h}."""
    sx = new_w / orig_w
    sy = new_h / orig_h
    x = int(x_min * sx)
    y = int(y_min * sy)
    w = int((x_max - x_min) * sx)
    h = int((y_max - y_min) * sy)
    return {"x": x, "y": y, "w": w, "h": h}


def yolo_to_bbox(class_id, x_center, y_center, width, height, img_w, img_h):
    """Convert YOLO normalized coords to pixel {x, y, w, h} in display coords.
    YOLO format: center_x, center_y, width, height (all 0-1 normalized).
    """
    # Convert normalized to pixel coords in display image
    cx = x_center * img_w
    cy = y_center * img_h
    bw = width * img_w
    bh = height * img_h
    x = int(cx - bw / 2)
    y = int(cy - bh / 2)
    return {"x": max(0, x), "y": max(0, y), "w": int(bw), "h": int(bh)}


# ── YOLO class mapping for GRAZPEDWRI-DX ───────────────────────────────────

YOLO_CLASSES = {
    0: "Bone anomaly",
    1: "Bone lesion",
    2: "Foreign body",
    3: "Fracture",
    4: "Metal",
    5: "Periosteal reaction",
    6: "Pronator sign",
    7: "Soft tissue",
    8: "Text",  # burned-in text overlay, skip this
}

# Only show clinically relevant classes (skip text overlays)
SKIP_CLASSES = {8}


# ── VinBigData Processing ──────────────────────────────────────────────────

def load_vinbig_annotations() -> dict:
    """Load VinBigData train.csv into {image_id: [annotations]}."""
    csv_path = VINBIG_DIR / "train.csv"
    annotations = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append(row)
    return annotations


VINBIG_DESCRIPTIONS = {
    "Cardiomegaly": "Cardiomegaly — enlarged cardiac silhouette with cardiothoracic ratio exceeding 0.5.",
    "Consolidation": "Consolidation — dense opacification with air bronchograms, suggestive of pneumonia.",
    "Lung Opacity": "Lung opacity — increased radiodensity within the lung field, may indicate infiltrate, mass, or fluid.",
    "Nodule/Mass": "Pulmonary nodule or mass — focal rounded opacity requiring further evaluation.",
    "Pleural thickening": "Pleural thickening — increased density along the pleural surface indicating chronic inflammation.",
    "Pulmonary fibrosis": "Pulmonary fibrosis — reticular interstitial pattern with volume loss.",
    "Aortic enlargement": "Aortic enlargement — widened aortic contour suggesting aneurysm or unfolding.",
    "Infiltration": "Pulmonary infiltration — ill-defined opacity suggesting inflammatory process.",
    "ILD": "Interstitial lung disease — diffuse reticulonodular pattern through the lung fields.",
    "Pleural effusion": "Pleural effusion — blunting of costophrenic angle with meniscus sign.",
    "Pneumothorax": "Pneumothorax — visible visceral pleural line with absent lung markings peripherally.",
    "Atelectasis": "Atelectasis — volume loss with increased opacity, suggesting partial lung collapse.",
    "Calcification": "Calcification — focal area of high density suggesting calcified granuloma.",
    "No finding": "Normal chest radiograph. Clear lung fields bilaterally. Normal cardiac silhouette. No pleural effusion or pneumothorax.",
}


def process_vinbigdata() -> list[dict]:
    """Process VinBigData DICOM images with real radiologist annotations."""
    if not VINBIG_DIR.exists():
        print("  ✗ vinbigdata/ directory not found, skipping")
        return []

    annotations = load_vinbig_annotations()
    results = []
    dicom_files = sorted(VINBIG_DIR.glob("*.dicom"))

    for i, dicom_path in enumerate(dicom_files, 1):
        img_id = dicom_path.stem
        filename = f"chest_{i:03d}.jpg"
        dest = SAMPLE_DIR / filename

        print(f"  Processing {img_id} → {filename}...")
        orig_w, orig_h, new_w, new_h = dicom_to_jpeg(dicom_path, dest)

        img_anns = annotations.get(img_id, [])
        bboxes = []
        findings = []
        is_normal = all(a["class_name"] == "No finding" for a in img_anns)

        if not is_normal:
            seen = set()
            for ann in img_anns:
                cls = ann["class_name"]
                if cls == "No finding" or not ann["x_min"]:
                    continue
                if cls not in seen:
                    seen.add(cls)
                    bbox = scale_bbox_abs(
                        float(ann["x_min"]), float(ann["y_min"]),
                        float(ann["x_max"]), float(ann["y_max"]),
                        orig_w, orig_h, new_w, new_h,
                    )
                    bbox["label"] = cls
                    bboxes.append(bbox)
                    findings.append(cls)

        if is_normal:
            ground_truth = VINBIG_DESCRIPTIONS["No finding"]
            subcategory = "normal"
            difficulty = "easy"
        else:
            desc_parts = [VINBIG_DESCRIPTIONS.get(f, f) for f in findings]
            ground_truth = " ".join(desc_parts)
            subcategory = findings[0].lower().replace(" ", "_").replace("/", "_")
            difficulty = "easy" if len(findings) == 1 else "hard" if len(findings) >= 3 else "medium"

        results.append({
            "id": f"chest_{i:03d}",
            "filename": filename,
            "category": "chest_pathology",
            "subcategory": subcategory,
            "ground_truth": ground_truth,
            "annotations": bboxes,
            "difficulty": difficulty,
        })
        print(f"  ✓ {filename} ({new_w}x{new_h}): {', '.join(findings) if findings else 'Normal'} ({len(bboxes)} boxes)")

    return results


# ── GRAZPEDWRI-DX Processing ───────────────────────────────────────────────

def load_grape_metadata() -> dict:
    csv_path = GRAPE_DIR / "dataset.csv"
    metadata = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["filestem"]] = row
    return metadata


def load_yolo_labels(stem: str) -> list[tuple]:
    """Load YOLO label file, return list of (class_id, xc, yc, w, h)."""
    label_path = GRAPE_DIR / "yolov5" / "labels" / f"{stem}.txt"
    if not label_path.exists():
        return []
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                labels.append((cls_id, xc, yc, w, h))
    return labels


AO_DESCRIPTIONS = {
    "23r-M/2.1": "Distal radius metaphyseal torus/buckle fracture. Cortical buckling without complete break.",
    "23r-M/3.1": "Distal radius metaphyseal fracture with complete cortical disruption and angulation.",
    "23r-E/2.1": "Distal radius epiphyseal (Salter-Harris) fracture. Growth plate involvement.",
    "23u-E/7": "Distal ulna epiphyseal fracture.",
}


def process_grazpedwri() -> list[dict]:
    """Process GRAZPEDWRI-DX images with YOLO bounding boxes."""
    if not GRAPE_DIR.exists():
        print("  ✗ grape/ directory not found, skipping")
        return []

    metadata = load_grape_metadata()
    results = []
    png_files = sorted(GRAPE_DIR.glob("*.png"))

    for i, png_path in enumerate(png_files, 1):
        stem = png_path.stem
        filename = f"fracture_{i:03d}.jpg"
        dest = SAMPLE_DIR / filename

        print(f"  Processing {stem} → {filename}...")
        orig_w, orig_h, new_w, new_h = png_to_jpeg(png_path, dest)

        # Load YOLO labels
        yolo_labels = load_yolo_labels(stem)

        # Convert YOLO to display bboxes, skip text overlays
        bboxes = []
        finding_names = []
        for cls_id, xc, yc, w, h in yolo_labels:
            if cls_id in SKIP_CLASSES:
                continue
            label = YOLO_CLASSES.get(cls_id, f"Class {cls_id}")
            bbox = yolo_to_bbox(cls_id, xc, yc, w, h, new_w, new_h)
            bbox["label"] = label
            bboxes.append(bbox)
            finding_names.append(label)

        # Get CSV metadata for ground truth text
        meta = metadata.get(stem, {})
        ao_class = meta.get("ao_classification", "")
        fracture_visible = meta.get("fracture_visible", "") == "1"
        has_cast = meta.get("cast", "") == "1"
        age = meta.get("age", "unknown")
        gender = meta.get("gender", "")
        laterality = meta.get("laterality", "")

        side = "left" if laterality == "L" else "right" if laterality == "R" else ""
        patient_info = f"Pediatric patient (age {age}, {gender}). {side.title()} wrist radiograph."

        if fracture_visible and bboxes:
            ao_parts = [p.strip() for p in ao_class.split(";")] if ao_class else []
            ao_descs = [AO_DESCRIPTIONS.get(p, f"Fracture ({p}).") for p in ao_parts]
            ground_truth = f"{patient_info} {' '.join(ao_descs)}"
            if has_cast:
                ground_truth += " Cast present — follow-up image."
            subcategory = "displaced" if "M/3" in ao_class else "non-displaced"
            difficulty = "medium" if len(bboxes) == 1 else "hard"
        elif fracture_visible:
            ground_truth = f"{patient_info} Fracture diagnosed ({ao_class}) but may be subtle on this view."
            subcategory = "hairline"
            difficulty = "hard"
        else:
            ground_truth = f"{patient_info} No fracture. Normal bone alignment and cortical margins."
            subcategory = "normal"
            difficulty = "easy"

        results.append({
            "id": f"fracture_{i:03d}",
            "filename": filename,
            "category": "fracture",
            "subcategory": subcategory,
            "ground_truth": ground_truth,
            "annotations": bboxes,
            "difficulty": difficulty,
        })

        box_info = f"{len(bboxes)} boxes: {', '.join(finding_names)}" if bboxes else "no clinical boxes"
        print(f"  ✓ {filename} ({new_w}x{new_h}): {box_info}")

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MedTeach AI — Image Processor")
    print("=" * 60)

    all_images = []

    print("\n[1/2] VinBigData Chest X-rays (DICOM → JPEG, real radiologist bboxes)")
    chest = process_vinbigdata()
    all_images.extend(chest)

    print("\n[2/2] GRAZPEDWRI-DX Fractures (PNG → JPEG, real YOLO bboxes)")
    fractures = process_grazpedwri()
    all_images.extend(fractures)

    # Write metadata.json
    print(f"\n{'=' * 60}")
    print("Generating metadata.json...")

    metadata = {"images": all_images}
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✓ Saved metadata for {len(all_images)} images")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    cats: dict[str, int] = {}
    bbox_count = 0
    for img in all_images:
        cats[img["category"]] = cats.get(img["category"], 0) + 1
        bbox_count += len(img["annotations"])

    print(f"  Total images:        {len(all_images)}")
    print(f"  Real bounding boxes: {bbox_count}")
    print(f"  Categories:")
    for cat, count in sorted(cats.items()):
        print(f"    - {cat}: {count} images")

    print(f"\n  Files saved to: {SAMPLE_DIR.resolve()}")
    print(f"  Metadata saved to: {METADATA_PATH.resolve()}")
    print(f"\n  Next step: streamlit run app.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
