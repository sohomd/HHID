import numpy as np
import cv2
from pathlib import Path


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask > 0


def compute_iou(m1, m2):
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else np.logical_and(m1, m2).sum() / union


def get_common_files(dirs, exts=(".png", ".jpg", ".jpeg")):
    sets = [
        {f.name for f in Path(d).iterdir() if f.is_file() and f.suffix.lower() in exts}
        for d in dirs
    ]
    return sorted(set.intersection(*sets))


def compute_isr_from_folders(gt_a_dir, gt_b_dir, pred_a_dir, pred_b_dir, margin=0.0):
    """
    ISR = % of evaluable frames where the optimal prediction->GT assignment
    is the crossed one (A->B, B->A) rather than the identity one.

    A frame is evaluable only if both GT masks are non-empty; frames where a
    subject is absent cannot exhibit an identity swap and would otherwise
    dilute the rate.

    margin : the crossed assignment must beat the direct one by at least this
             much in summed IoU before the frame is flagged as a swap.
    """
    common_files = get_common_files([gt_a_dir, gt_b_dir, pred_a_dir, pred_b_dir])
    if not common_files:
        raise ValueError("No common files found across GT and predicted folders.")

    switch_count = 0
    skipped = 0
    frame_results = []

    for fname in common_files:
        gt_a = load_mask(Path(gt_a_dir) / fname)
        gt_b = load_mask(Path(gt_b_dir) / fname)
        pr_a = load_mask(Path(pred_a_dir) / fname)
        pr_b = load_mask(Path(pred_b_dir) / fname)

        if not (gt_a.shape == gt_b.shape == pr_a.shape == pr_b.shape):
            raise ValueError(f"Shape mismatch in frame {fname}")

        if not gt_a.any() or not gt_b.any():
            skipped += 1
            continue

        iou_aa = compute_iou(pr_a, gt_a)
        iou_ab = compute_iou(pr_a, gt_b)
        iou_bb = compute_iou(pr_b, gt_b)
        iou_ba = compute_iou(pr_b, gt_a)

        direct = iou_aa + iou_bb
        crossed = iou_ab + iou_ba
        switched = (crossed - direct) > margin

        switch_count += switched
        frame_results.append({
            "frame": fname,
            "IoU_AA": round(iou_aa, 4),
            "IoU_AB": round(iou_ab, 4),
            "IoU_BB": round(iou_bb, 4),
            "IoU_BA": round(iou_ba, 4),
            "direct_score": round(direct, 4),
            "crossed_score": round(crossed, 4),
            "switched": bool(switched),
        })

    valid_frames = len(frame_results)
    if valid_frames == 0:
        raise ValueError("No evaluable frames (both GT masks empty everywhere).")

    return {
        "total_frames": valid_frames,
        "skipped_frames": skipped,
        "switch_count": int(switch_count),
        "isr_percent": round(switch_count / valid_frames * 100.0, 4),
        "frame_results": frame_results,
    }


if __name__ == "__main__":
    result = compute_isr_from_folders(
        gt_a_dir=r"Data/Instance Mask/A",
        gt_b_dir=r"Data/Instance Mask/B",
        pred_a_dir=r"PRED/Instance Mask/A",
        pred_b_dir=r"PRED/Instance Mask/B",
        margin=0.0,
    )

    print("ISR Results")
    print(f"Total frames   : {result['total_frames']}")
    print(f"Skipped frames : {result['skipped_frames']}")
    print(f"Switch count   : {result['switch_count']}")
    print(f"ISR (%)        : {result['isr_percent']}")
