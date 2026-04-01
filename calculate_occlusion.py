import os
import cv2
import json
import numpy as np
from pathlib import Path


def load_mask(mask_path):
    """
    Load a mask image and convert it to binary (0 or 1).
    Works for grayscale or color mask images.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    # If mask has multiple channels, convert to grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Convert to binary
    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask


def compute_iou(mask_a, mask_b):
    """
    Compute IoU (%) between two binary masks.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    iou = (intersection / union) * 100.0
    return iou


def categorize_occlusion(iou_percent):
    """
    Categorize occlusion/contact level based on paper thresholds.
    """
    if iou_percent < 10:
        return "Low"
    elif iou_percent <= 30:
        return "Moderate"
    else:
        return "High"


def process_camera_sequence(mask_a_dir, mask_b_dir, output_json=None):
    """
    Compute per-frame occlusion IoU for one camera folder.

    Assumes:
      - mask_a_dir contains masks for subject A
      - mask_b_dir contains masks for subject B
      - filenames match frame-by-frame
    """
    mask_a_dir = Path(mask_a_dir)
    mask_b_dir = Path(mask_b_dir)

    files_a = sorted([f for f in mask_a_dir.iterdir() if f.is_file()])
    files_b = sorted([f for f in mask_b_dir.iterdir() if f.is_file()])

    names_a = {f.name for f in files_a}
    names_b = {f.name for f in files_b}
    common_files = sorted(list(names_a.intersection(names_b)))

    if not common_files:
        raise ValueError("No matching mask filenames found between A and B folders.")

    results = []

    for fname in common_files:
        path_a = mask_a_dir / fname
        path_b = mask_b_dir / fname

        mask_a = load_mask(path_a)
        mask_b = load_mask(path_b)

        if mask_a.shape != mask_b.shape:
            raise ValueError(f"Shape mismatch for {fname}: {mask_a.shape} vs {mask_b.shape}")

        iou = compute_iou(mask_a, mask_b)
        level = categorize_occlusion(iou)

        results.append({
            "frame": fname,
            "iou_percent": round(iou, 4),
            "contact_level": level
        })

    mean_iou = float(np.mean([r["iou_percent"] for r in results]))
    peak_iou = float(np.max([r["iou_percent"] for r in results]))

    summary = {
        "num_frames": len(results),
        "mean_iou_percent": round(mean_iou, 4),
        "peak_iou_percent": round(peak_iou, 4),
        "overall_contact_level": categorize_occlusion(mean_iou),
        "frames": results
    }

    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=4)

    return summary


if __name__ == "__main__":
    # Example paths
    mask_a_dir = r"Instance_Mask/A"
    mask_b_dir = r"Instance_Mask/B"
    output_json = "occlusion_results.json"

    summary = process_camera_sequence(mask_a_dir, mask_b_dir, output_json)

    print("Occlusion Summary")
    print(f"Frames processed     : {summary['num_frames']}")
    print(f"Mean IoU (%)         : {summary['mean_iou_percent']}")
    print(f"Peak IoU (%)         : {summary['peak_iou_percent']}")
    print(f"Overall contact level: {summary['overall_contact_level']}")
    print(f"Saved results to     : {output_json}")