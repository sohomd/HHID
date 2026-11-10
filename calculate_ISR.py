import csv
import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# High-contact subset threshold on the Eq. (1) overlap ratio, in percent (Sec. 4.1).
HIGH_CONTACT_THRESHOLD = 30.0


# --------------------------------------------------------------------------- #
# Mask utilities
# --------------------------------------------------------------------------- #
def load_mask(path):
    """Read a mask as a binary uint8 array."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def compute_iou(m1, m2):
    """Pixel-wise IoU of two binary masks. Empty union -> 0.0."""
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else float(intersection) / float(union)


def get_common_files(*dirs, exts=IMG_EXTS):
    """Filenames present in every supplied directory."""
    sets = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            raise NotADirectoryError(f"Missing directory: {d}")
        sets.append({f.name for f in d.iterdir()
                     if f.is_file() and f.suffix.lower() in exts})
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


# --------------------------------------------------------------------------- #
# ISR
# --------------------------------------------------------------------------- #
def compute_isr_from_folders(gt_a_dir, gt_b_dir, pred_a_dir, pred_b_dir,
                             margin=0.0):
    """
    ISR for a single camera view.

    margin : optional strictness offset. The paper uses 0.0; a small positive
             value (e.g. 0.05) requires the wrong-identity IoU to win by a
             clear gap, suppressing flicker at the decision boundary.
             Leave at 0.0 to reproduce the published numbers.
    """
    files = get_common_files(gt_a_dir, gt_b_dir, pred_a_dir, pred_b_dir)
    if not files:
        raise ValueError("No common files across the four mask folders.")

    frames = []
    switch_count = 0
    a_only = b_only = both = 0

    for fname in files:
        gt_a = load_mask(Path(gt_a_dir) / fname)
        gt_b = load_mask(Path(gt_b_dir) / fname)
        pr_a = load_mask(Path(pred_a_dir) / fname)
        pr_b = load_mask(Path(pred_b_dir) / fname)

        if not (gt_a.shape == gt_b.shape == pr_a.shape == pr_b.shape):
            raise ValueError(f"Shape mismatch in frame {fname}")

        iou_aa = compute_iou(pr_a, gt_a)
        iou_ab = compute_iou(pr_a, gt_b)
        iou_bb = compute_iou(pr_b, gt_b)
        iou_ba = compute_iou(pr_b, gt_a)

        # Eq. (3): disjunction of the two per-identity tests.
        switch_a = (iou_ab - iou_aa) > margin
        switch_b = (iou_ba - iou_bb) > margin
        s_t = bool(switch_a or switch_b)

        if s_t:
            switch_count += 1
            if switch_a and switch_b:
                both += 1
            elif switch_a:
                a_only += 1
            else:
                b_only += 1

        # Eq. (1): inter-subject overlap ratio on the ground-truth masks,
        # used to select the high-contact subset.
        overlap = compute_iou(gt_a, gt_b) * 100.0

        frames.append({
            "frame": fname,
            "IoU_AA": round(iou_aa, 4),
            "IoU_AB": round(iou_ab, 4),
            "IoU_BB": round(iou_bb, 4),
            "IoU_BA": round(iou_ba, 4),
            "switch_A": switch_a,
            "switch_B": switch_b,
            "s_t": int(s_t),
            "overlap_percent": round(overlap, 4),
            "high_contact": overlap > HIGH_CONTACT_THRESHOLD,
        })

    total = len(frames)
    hc = [f for f in frames if f["high_contact"]]
    hc_switches = sum(f["s_t"] for f in hc)

    return {
        "total_frames": total,
        "switch_count": switch_count,
        "isr_percent": round(100.0 * switch_count / total, 4),
        "switch_events": count_switch_events(frames),
        "breakdown": {"A_only": a_only, "B_only": b_only, "both": both},
        "high_contact_frames": len(hc),
        "high_contact_switches": hc_switches,
        "high_contact_isr_percent": (round(100.0 * hc_switches / len(hc), 4)
                                     if hc else None),
        "mean_overlap_percent": round(
            float(np.mean([f["overlap_percent"] for f in frames])), 4),
        "frame_results": frames,
    }


def count_switch_events(frames):
    """
    Number of 0 -> 1 transitions in s_t.

    ISR counts frames in a swapped state; this counts how many times identity
    is lost. They are different quantities and are reported separately.
    """
    events = 0
    prev = 0
    for f in frames:
        if f["s_t"] == 1 and prev == 0:
            events += 1
        prev = f["s_t"]
    return events


def evaluate_camera(camera_dir, margin=0.0):
    """ISR for one CameraN folder using the README layout."""
    camera_dir = Path(camera_dir)
    return compute_isr_from_folders(
        gt_a_dir=camera_dir / "Instance Mask" / "A",
        gt_b_dir=camera_dir / "Instance Mask" / "B",
        pred_a_dir=camera_dir / "PRED" / "Instance Mask" / "A",
        pred_b_dir=camera_dir / "PRED" / "Instance Mask" / "B",
        margin=margin,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_csv(result, path):
    fields = ["frame", "IoU_AA", "IoU_AB", "IoU_BB", "IoU_BA",
              "switch_A", "switch_B", "s_t", "overlap_percent", "high_contact"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["frame_results"])


def print_camera_report(name, r):
    print(f"\n--- {name} ---")
    print(f"  Frames evaluated      : {r['total_frames']}")
    print(f"  Switched frames       : {r['switch_count']}")
    print(f"  ISR (%)               : {r['isr_percent']}")
    print(f"  Switch events (0->1)  : {r['switch_events']}")
    b = r["breakdown"]
    print(f"  Breakdown             : A-only {b['A_only']} | "
          f"B-only {b['B_only']} | both {b['both']}")
    print(f"  Mean GT overlap (%)   : {r['mean_overlap_percent']}")
    if r["high_contact_isr_percent"] is None:
        print(f"  High-contact (>{HIGH_CONTACT_THRESHOLD:.0f}%)   : no frames")
    else:
        print(f"  High-contact frames   : {r['high_contact_frames']}")
        print(f"  High-contact ISR (%)  : {r['high_contact_isr_percent']}")


def main():
    # Change these paths if needed.
    data_root = Path("Data")
    cameras = ["Camera1", "Camera2", "Camera3", "Camera4"]
    margin = 0.0
    write_per_frame_csv = True

    print("HHID - Identity Switching Rate (ISR)")
    print(f"Criterion: IoU(P_A,G_B) > IoU(P_A,G_A) OR IoU(P_B,G_A) > IoU(P_B,G_B)")
    print(f"Data root: {data_root.resolve()}")

    results = {}
    for cam in cameras:
        cam_dir = data_root / cam
        if not cam_dir.is_dir():
            print(f"\n[skip] {cam}: folder not found")
            continue
        try:
            r = evaluate_camera(cam_dir, margin=margin)
        except (FileNotFoundError, NotADirectoryError, ValueError) as err:
            print(f"\n[skip] {cam}: {err}")
            continue

        results[cam] = r
        print_camera_report(cam, r)

        if write_per_frame_csv:
            out_csv = cam_dir / "isr_per_frame.csv"
            write_csv(r, out_csv)
            print(f"  Per-frame CSV         : {out_csv}")

    if not results:
        print("\nNo cameras evaluated. Check the folder structure.")
        return

    # View-averaged ISR (Sec. 4.2: metrics computed independently per view).
    all_isr = [r["isr_percent"] for r in results.values()]
    hc_isr = [r["high_contact_isr_percent"] for r in results.values()
              if r["high_contact_isr_percent"] is not None]

    print("\n=== Summary over cameras ===")
    print(f"  Cameras evaluated     : {len(results)}")
    print(f"  Mean ISR (%)          : {round(float(np.mean(all_isr)), 4)}")
    if hc_isr:
        print(f"  Mean high-contact ISR : {round(float(np.mean(hc_isr)), 4)}")

    summary = {
        cam: {k: v for k, v in r.items() if k != "frame_results"}
        for cam, r in results.items()
    }
    summary["mean_isr_percent"] = round(float(np.mean(all_isr)), 4)
    if hc_isr:
        summary["mean_high_contact_isr_percent"] = round(float(np.mean(hc_isr)), 4)

    out_json = data_root / "isr_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Summary written to    : {out_json}")


if __name__ == "__main__":
    main()
