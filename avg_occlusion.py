import os
import cv2
import json
import numpy as np
from pathlib import Path


def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return (mask > 0).astype(np.uint8)


def compute_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return 0.0 if union == 0 else (intersection / union) * 100.0


def categorize_occlusion(iou_percent):
    if iou_percent < 10:
        return "Low"
    elif iou_percent <= 30:
        return "Moderate"
    return "High"


def compute_camera_framewise_iou(camera_dir):
    camera_dir = Path(camera_dir)
    a_dir = camera_dir / "Instance Mask" / "A"
    b_dir = camera_dir / "Instance Mask" / "B"

    if not a_dir.exists() or not b_dir.exists():
        raise FileNotFoundError(f"Missing A/B instance mask folders in {camera_dir}")

    files_a = {f.name: f for f in a_dir.iterdir() if f.is_file()}
    files_b = {f.name: f for f in b_dir.iterdir() if f.is_file()}
    common = sorted(files_a.keys() & files_b.keys())

    frame_iou = {}

    for fname in common:
        mask_a = load_mask(files_a[fname])
        mask_b = load_mask(files_b[fname])

        if mask_a.shape != mask_b.shape:
            raise ValueError(f"Shape mismatch in {camera_dir.name}, frame {fname}")

        frame_iou[fname] = compute_iou(mask_a, mask_b)

    return frame_iou


def compute_multicamera_occlusion(dataset_root, output_json="multicamera_occlusion.json"):
    dataset_root = Path(dataset_root)
    camera_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    per_camera = {}
    all_frames = {}

    for cam_dir in camera_dirs:
        cam_result = compute_camera_framewise_iou(cam_dir)
        per_camera[cam_dir.name] = cam_result

        for frame_name, iou in cam_result.items():
            all_frames.setdefault(frame_name, []).append(iou)

    averaged_frames = []
    for frame_name in sorted(all_frames.keys()):
        mean_iou = float(np.mean(all_frames[frame_name]))
        averaged_frames.append({
            "frame": frame_name,
            "mean_iou_percent_across_cameras": round(mean_iou, 4),
            "contact_level": categorize_occlusion(mean_iou)
        })

    mean_all = float(np.mean([x["mean_iou_percent_across_cameras"] for x in averaged_frames])) if averaged_frames else 0.0
    peak_all = float(np.max([x["mean_iou_percent_across_cameras"] for x in averaged_frames])) if averaged_frames else 0.0

    summary = {
        "num_cameras": len(per_camera),
        "num_frames": len(averaged_frames),
        "dataset_mean_iou_percent": round(mean_all, 4),
        "dataset_peak_iou_percent": round(peak_all, 4),
        "overall_contact_level": categorize_occlusion(mean_all),
        "per_camera_framewise_iou": per_camera,
        "averaged_framewise_iou": averaged_frames
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=4)

    return summary


if __name__ == "__main__":
    dataset_root = r"HHID"   # folder containing Camera1, Camera2, ...
    summary = compute_multicamera_occlusion(dataset_root)

    print("Multi-camera Occlusion Summary")
    print(f"Number of cameras      : {summary['num_cameras']}")
    print(f"Number of frames       : {summary['num_frames']}")
    print(f"Dataset Mean IoU (%)   : {summary['dataset_mean_iou_percent']}")
    print(f"Dataset Peak IoU (%)   : {summary['dataset_peak_iou_percent']}")
    print(f"Overall contact level  : {summary['overall_contact_level']}")
    print("Saved to multicamera_occlusion.json")