import os
from pathlib import Path
import numpy as np
import cv2


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def compute_iou(m1, m2):
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else intersection / union


def get_common_files(dir1, dir2, dir3, dir4, exts=(".png", ".jpg", ".jpeg")):
    """
    Find filenames common to all 4 folders.
    """
    s1 = {f.name for f in Path(dir1).iterdir() if f.is_file() and f.suffix.lower() in exts}
    s2 = {f.name for f in Path(dir2).iterdir() if f.is_file() and f.suffix.lower() in exts}
    s3 = {f.name for f in Path(dir3).iterdir() if f.is_file() and f.suffix.lower() in exts}
    s4 = {f.name for f in Path(dir4).iterdir() if f.is_file() and f.suffix.lower() in exts}

    common = sorted(s1 & s2 & s3 & s4)
    return common


def compute_isr_from_folders(gt_a_dir, gt_b_dir, pred_a_dir, pred_b_dir, margin=0.0):
    """
    ISR = percentage of frames where predicted identities are swapped.

    gt_a_dir   : folder with GT masks for subject A
    gt_b_dir   : folder with GT masks for subject B
    pred_a_dir : folder with predicted masks for subject A
    pred_b_dir : folder with predicted masks for subject B
    margin     : optional safety margin for stricter switch detection
    """
    common_files = get_common_files(gt_a_dir, gt_b_dir, pred_a_dir, pred_b_dir)

    if len(common_files) == 0:
        raise ValueError("No common files found across GT and predicted folders.")

    switch_count = 0
    valid_frames = 0
    frame_results = []

    for fname in common_files:
        gt_a = load_mask(Path(gt_a_dir) / fname)
        gt_b = load_mask(Path(gt_b_dir) / fname)
        pr_a = load_mask(Path(pred_a_dir) / fname)
        pr_b = load_mask(Path(pred_b_dir) / fname)

        if gt_a.shape != gt_b.shape or gt_a.shape != pr_a.shape or gt_a.shape != pr_b.shape:
            raise ValueError(f"Shape mismatch in frame {fname}")

        iou_aa = compute_iou(pr_a, gt_a)
        iou_ab = compute_iou(pr_a, gt_b)
        iou_bb = compute_iou(pr_b, gt_b)
        iou_ba = compute_iou(pr_b, gt_a)

        switched = ((iou_ab - iou_aa) > margin) and ((iou_ba - iou_bb) > margin)

        if switched:
            switch_count += 1

        valid_frames += 1

        frame_results.append({
            "frame": fname,
            "IoU_AA": round(iou_aa, 4),
            "IoU_AB": round(iou_ab, 4),
            "IoU_BB": round(iou_bb, 4),
            "IoU_BA": round(iou_ba, 4),
            "switched": switched
        })

    isr = (switch_count / valid_frames) * 100.0

    return {
        "total_frames": valid_frames,
        "switch_count": switch_count,
        "isr_percent": round(isr, 4),
        "frame_results": frame_results
    }


if __name__ == "__main__":
    gt_a_dir = r"Data/Instance Mask/A"
    gt_b_dir = r"Data/Instance Mask/B"
    pred_a_dir = r"PRED/Instance Mask/A"
    pred_b_dir = r"PRED/Instance Mask/B"

    result = compute_isr_from_folders(
        gt_a_dir=gt_a_dir,
        gt_b_dir=gt_b_dir,
        pred_a_dir=pred_a_dir,
        pred_b_dir=pred_b_dir,
        margin=0.0
    )

    print("ISR Results")
    print(f"Total frames : {result['total_frames']}")
    print(f"Switch count : {result['switch_count']}")
    print(f"ISR (%)      : {result['isr_percent']}")