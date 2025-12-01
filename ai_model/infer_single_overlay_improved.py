"""
Microscopy candidate union + gated detection (224, no-norm by default)

This script runs the full CellCounter/DDG biomarker-detection pipeline on a single
microscopy image and writes:

- A PNG overlay with detections drawn as circles
- A CSV of detections (x, y, score)
- A CSV of artifact/debug paths
- A DDG-compatible .pnt file with point annotations for DotDotGoose

High-level pipeline
-------------------
1. **Load image**
2. **Optional edge/green/yellow gate** (structural + color gate)
3. **Strict HSV yellow candidates**
4. **Candidate union + small-radius NMS** (dedupe)
5. **Center crops around candidates**
6. **Optional pre-CNN FP filters** (compactness, blobness, red-edge, gate proximity)
7. **CNN scoring** (ResNet-18, 224x224 crops, 1 output logit per crop)
8. **Score thresholding**
9. **Radius-based NMS** (pre-snap)
10. **Optional snapping to yellow centroids** (HSV-based refinement with jump safety)
11. **Radius-based NMS** (post-snap)
12. **Tight-pair pruning** (remove very close duplicates)
13. **Write overlay + CSVs + PNT**

Usage
-----
python infer_single_overlay_improved.py \
    --image /path/to/image.png \
    --weights /path/to/resnet18_weights.pt \
    --out_dir experiments/quickcheck \
    [other optional flags...]

By default:
- Model expects 224x224 crops
- No normalization (unless --normalize is set)
- Color order is BGR->RGB before feeding the model (color_order="rgb")
"""

import sys
import os
import pkgutil
import argparse
import csv
import json

print("[DIAG] INFER PYTHON:", sys.executable)
print("[DIAG] PATH[0]:", os.environ.get("PATH", "").split(os.pathsep)[0])
print("[DIAG] Has cv2 in this interpreter?:", pkgutil.find_loader("cv2") is not None)

from pathlib import Path
from typing import List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


# -------------------- utils --------------------
def load_image_bgr(path: str) -> np.ndarray:
    """
    Load an image from disk in BGR color order.

    Args:
        path: Path to the input image.

    Returns:
        np.ndarray: Image array (H, W, 3) in BGR uint8 format.

    Raises:
        FileNotFoundError: If OpenCV cannot read the image from the path.
        """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_tensor_bchw_uint8(crops: List[np.ndarray], color_order: str, normalize: bool,
                         mean_str: str, std_str: str, device: torch.device) -> torch.Tensor:
    """
    Convert a list of BGR crops to a float32 tensor (N, C, H, W) for PyTorch.

    Pipeline:
      - Stack crops into a single array (N, H, W, C)
      - Optionally convert BGR -> RGB
      - Scale to [0, 1]
      - Optional channel-wise normalization
      - Transpose to (N, C, H, W)
      - Move to the requested device

    Args:
        crops: List of image crops in BGR uint8 format with equal size.
        color_order: Either "rgb" or "bgr". If "rgb", crops are converted BGR->RGB.
        normalize: Whether to apply per-channel normalization.
        mean_str: Comma-separated means per channel (e.g. "0.485,0.456,0.406").
        std_str: Comma-separated std per channel (e.g. "0.229,0.224,0.225").
        device: PyTorch device to move the tensor to.

    Returns:
        torch.Tensor: Float32 tensor of shape (N, C, H, W) on the specified device.
        """
    arr = np.stack(crops, axis=0)  # N,H,W,C (BGR)
    if color_order.lower() == "rgb":
        arr = arr[:, :, :, ::-1]   # BGR->RGB
    arr = arr.astype(np.float32) / 255.0
    if normalize:
        mean = np.array([float(v) for v in mean_str.split(",")], dtype=np.float32).reshape(1, 1, 1, 3)
        std  = np.array([float(v) for v in std_str.split(",")],  dtype=np.float32).reshape(1, 1, 1, 3)
        arr = (arr - mean) / (std + 1e-7)
    arr = arr.transpose(0, 3, 1, 2)  # N,C,H,W
    return torch.from_numpy(arr).to(device)


def strip_prefix_if_present(state_dict, prefix: str):
    """
    Strip a prefix from all keys in a state_dict if all keys start with that prefix.

    Useful when loading checkpoints saved with DataParallel or wrappers like "module.".

    Args:
        state_dict: A mapping of parameter names to tensors.
        prefix: Prefix string to strip (e.g. "module.").

    Returns:
        dict: Potentially modified state_dict with prefix removed.
        """
    keys = list(state_dict.keys())
    if keys and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def build_resnet18(num_classes: int) -> nn.Module:
    """
    Construct a vanilla torchvision ResNet-18 for classification.

    Args:
        num_classes: Number of output classes (here typically 1 for sigmoid).

    Returns:
        nn.Module: ResNet-18 model with final fc layer replaced for num_classes.
        """
    import torchvision.models as tvm
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def extract_state_dict(ckpt: Any, checkpoint_key: Optional[str]):
    """
    Extract a tensor-only state_dict from a flexible checkpoint structure.

    Handles checkpoints that may contain nested keys like "state_dict",
    "model_state", "model", "net", or "weights".

    Args:
        ckpt: Loaded checkpoint object (typically from torch.load).
        checkpoint_key: Optional explicit key to look for inside ckpt.

    Returns:
        dict: A mapping from parameter names to tensors, or {} if nothing found.
        """
    if isinstance(ckpt, dict):
        if checkpoint_key and checkpoint_key in ckpt and isinstance(ckpt[checkpoint_key], dict):
            return ckpt[checkpoint_key]
        for k in ("state_dict", "model_state", "model", "net", "weights"):
            v = ckpt.get(k)
            if isinstance(v, dict) and v and isinstance(next(iter(v.values())), torch.Tensor):
                return v
        if ckpt and isinstance(next(iter(ckpt.values())), torch.Tensor):
            return ckpt
    if hasattr(ckpt, "keys") and ckpt and isinstance(next(iter(ckpt.values())), torch.Tensor):
        return ckpt
    return {}


def load_model(weights_path: str, device: torch.device, num_classes: int) -> nn.Module:
    """
    Load a ResNet-18 model with a binary head from a checkpoint.

    This:
      - torch.load's the checkpoint
      - Extracts a state_dict from common patterns
      - Strips known prefixes (module., model., net.) if present
      - Loads weights into a fresh ResNet-18(num_classes)

    Args:
        weights_path: Path to the .pt or .pth weights file.
        device: Torch device ("cpu" or "cuda").
        num_classes: Number of classes for the ResNet head.

    Returns:
        nn.Module: Loaded ResNet-18 model in eval() mode.
        """
    ckpt = torch.load(weights_path, map_location=device)
    state = extract_state_dict(ckpt, None)
    model = build_resnet18(num_classes=num_classes).to(device)
    for pref in ("module.", "model.", "net."):
        if any(k.startswith(pref) for k in state.keys()):
            state = strip_prefix_if_present(state, pref)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("[debug] model.fc weight shape:", tuple(model.fc.weight.shape))
    return model


# -------------------- HSV candidate extraction --------------------
def find_yellow_candidates(img_bgr: np.ndarray,
                           h_lo=15, h_hi=40,
                           s_lo=80, v_lo=80,
                           min_area=5, max_area=500) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Find yellow-ish connected components in HSV space as candidate centroids.

    Steps:
      - Convert BGR -> HSV
      - Threshold using provided (H, S, V) bounds
      - Morphological cleanup (median blur, open/close)
      - Extract contours and compute centroids
      - Filter by contour area

    Args:
        img_bgr: Input image in BGR uint8 format.
        h_lo: Lower hue bound.
        h_hi: Upper hue bound.
        s_lo: Lower saturation bound.
        v_lo: Lower value/brightness bound.
        min_area: Minimum contour area to keep.
        max_area: Maximum contour area to keep.

    Returns:
        Tuple[List[Tuple[int, int]], np.ndarray]:
            - List of (x, y) centroids of candidate components.
            - Final binary mask (uint8) used to find the components.
        """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # cleanup pipeline
    mask = cv2.medianBlur(mask, 5)
    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    kernel_medium = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: List[Tuple[int, int]] = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    return centers, mask


# -------------------- Edge + Green + Yellow gate --------------------
def _float_rgb(img_bgr: np.ndarray):
    """
    Convert a BGR uint8 image to float RGB channels in [0, 1].

    Args:
        img_bgr: Input BGR image (H, W, 3) uint8.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            R, G, B float32 arrays each in [0, 1].
        """
    img = img_bgr.astype(np.float32) / 255.0
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    return R, G, B


def build_edge_green_yellow_gate(
    img_bgr: np.ndarray,
    edge_band_px=5, canny_lo=10, canny_hi=40,
    green_k=0.25,
    tR=0.55, tG=0.55, tB=0.35, delta_rg=0.20, rg_ratio=0.75,
    min_cc_area=6, max_cc_area=200,
    keep_border=False, border_px=8,
    debug=False, debug_prefix="debug"
):
    """
    Build a binary gate mask based on red edges, green positivity, and yellow co-mix.

    Logic (per our prior experiments):
      1. Use Canny edges on blurred red channel, dilate to get a "red edge band".
      2. Compute an adaptive green threshold (mean + k * std) and keep green-positive pixels.
      3. Enforce "yellow-ish" pixels where R and G are both high, B is low, and R~G.
      4. Combine: gate = edge_band & green_mask & yellow.
      5. Optionally remove border pixels (to avoid frame artifacts).
      6. Clean using morph operations and connected component area filtering.

    Args:
        img_bgr: Input BGR image, uint8.
        edge_band_px: Radius for dilating edges into a band.
        canny_lo: Lower Canny threshold on red.
        canny_hi: Upper Canny threshold on red.
        green_k: Multiplier for green_std when computing green threshold.
        tR: Min R value (0..1) for yellow.
        tG: Min G value (0..1) for yellow.
        tB: Max B value (0..1) for yellow.
        delta_rg: Max |R - G| allowed (co-mix closeness).
        rg_ratio: Min min(R,G)/max(R,G) allowed.
        min_cc_area: Minimum connected component area to keep.
        max_cc_area: Maximum connected component area to keep.
        keep_border: If False, suppress a border_px-wide frame around the image.
        border_px: Border width in pixels.
        debug: If True, write intermediate masks to disk.
        debug_prefix: Path prefix for debug mask outputs.

    Returns:
        Tuple[np.ndarray, dict]:
            - gate_mask: Final boolean mask of candidate locations.
            - debug_dict: Dict with raw masks: {"edge_band", "green", "yellow"}.
        """
    H, W = img_bgr.shape[:2]
    R, G, B = _float_rgb(img_bgr)

    # 1) Red edge band
    red_u8 = np.clip(R * 255, 0, 255).astype(np.uint8)
    red_blur = cv2.GaussianBlur(red_u8, (0, 0), 1.0)
    edges = cv2.Canny(red_blur, canny_lo, canny_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_band_px + 1, 2 * edge_band_px + 1))
    edge_band = cv2.dilate(edges, kernel) > 0

    # 2) Green positivity (adaptive)
    g_blur = cv2.GaussianBlur(G, (0, 0), 1.0)
    g_thr = float(np.mean(g_blur) + green_k * np.std(g_blur))
    green_mask = (g_blur > g_thr)

    # 3) Yellow co-mix
    yellow = (R > tR) & (G > tG) & (B < tB)
    yellow &= (np.abs(R - G) < delta_rg)
    rg_min = np.minimum(R, G) + 1e-6
    rg_max = np.maximum(R, G) + 1e-6
    yellow &= (rg_min / rg_max) > rg_ratio

    cand = edge_band & green_mask & yellow

    # Remove border if requested
    if not keep_border:
        inner = np.zeros_like(cand, dtype=bool)
        inner[border_px:H - border_px, border_px:W - border_px] = True
        cand &= inner

    # Clean small speckles and fill small gaps
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, k3)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k3).astype(bool)

    # Area filter on connected components
    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    good = np.zeros_like(cand, dtype=bool)
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_cc_area <= a <= max_cc_area:
            good |= (lab == i)

    if debug:
        def out(img, name):
            cv2.imwrite(f"{debug_prefix}_{name}.png", (255 * img.astype(np.uint8)))
        out(edge_band, "edgeband")
        out(green_mask, "green")
        out(yellow, "yellow")
        out(good, "gatemask")

    return good, {"edge_band": edge_band, "green": green_mask, "yellow": yellow}


# crop around a point with reflect padding to fixed size
def center_crop(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """
    Extract a fixed-size crop centered at (x, y), using reflect padding if needed.

    Args:
        img: Input image (H, W, C).
        x: X coordinate of crop center.
        y: Y coordinate of crop center.
        size: Desired crop size (size x size).

    Returns:
        np.ndarray: Cropped image (size, size, C) in same dtype as input.
        """
    h, w = img.shape[:2]
    r = size // 2
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(w, x + r), min(h, y + r)
    crop = img[y0:y1, x0:x1, :]
    if crop.shape[0] != size or crop.shape[1] != size:
        pad_top = (size - crop.shape[0]) // 2
        pad_bottom = size - crop.shape[0] - pad_top
        pad_left = (size - crop.shape[1]) // 2
        pad_right = size - crop.shape[1] - pad_left
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    return crop


# -------------------- snap-to-yellow refinement --------------------
def snap_to_yellow_centroids(img_bgr,
                             pts_xy: np.ndarray,
                             search_r=8,
                             h_lo=18, h_hi=42, s_lo=150, v_lo=170,
                             min_area=8, max_area=220) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optionally refine candidate positions by snapping them to nearby yellow centroids.

    For each candidate point:
      - Look in a (2*search_r+1) window around the point in a strict HSV yellow mask.
      - Run connectedComponentsWithStats to find blobs.
      - Choose the centroid that is closest to the original candidate.
      - Keep only blobs whose area is within [min_area, max_area].

    Args:
        img_bgr: Input BGR image.
        pts_xy: Array of candidate points (N, 2) in (x, y) order.
        search_r: Search radius (in pixels) around each candidate.
        h_lo: Lower hue for yellow region.
        h_hi: Upper hue for yellow region.
        s_lo: Lower saturation bound.
        v_lo: Lower value/brightness bound.
        min_area: Minimum blob area to consider.
        max_area: Maximum blob area to consider.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - snapped_xy: (M, 2) array of refined coordinates.
            - keep_idx: (M,) array of indices into the original pts_xy that were snapped.
        """
    if len(pts_xy) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))
    mask = cv2.medianBlur(mask, 3)

    snapped = []
    keep_idx = []
    for i, (x, y) in enumerate(pts_xy.astype(int)):
        x0, y0 = max(0, x - search_r), max(0, y - search_r)
        x1, y1 = min(W, x + search_r + 1), min(H, y + search_r + 1)
        sub = mask[y0:y1, x0:x1]

        num, lab, stats, centroids = cv2.connectedComponentsWithStats(sub, connectivity=8)
        best_d2, best_xy = None, None
        cx0, cy0 = x - x0, y - y0
        for j in range(1, num):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            cx, cy = centroids[j]
            d2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
            if (best_d2 is None) or (d2 < best_d2):
                best_d2, best_xy = d2, (x0 + cx, y0 + cy)

        if best_xy is not None:
            snapped.append(best_xy)
            keep_idx.append(i)

    if len(snapped) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.array(snapped, dtype=np.float32), np.array(keep_idx, dtype=np.int32)


# -------------------- FP filters (OFF by default) --------------------
def dot_compactness_ok(crop_bgr, min_ratio=0.003, max_ratio=0.030, blur_sigma=1.0):
    """
    Filter based on the fraction of foreground pixels in a binarized crop.

    Heuristically:
      - Convert to grayscale, optionally Gaussian-blur.
      - Otsu-threshold to get a binary mask.
      - Compute ratio of foreground pixels.
      - Keep if ratio is in [min_ratio, max_ratio].

    Args:
        crop_bgr: Crop image in BGR.
        min_ratio: Minimum foreground pixel fraction.
        max_ratio: Maximum foreground pixel fraction.
        blur_sigma: Gaussian sigma for smoothing before threshold.

    Returns:
        bool: True if the compactness ratio lies within the specified bounds.
        """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0,0), blur_sigma)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ratio = float((thr > 0).sum()) / thr.size
    return (min_ratio <= ratio <= max_ratio)


def log_blobness_score(crop_bgr, sigma=1.2):
    """
    Compute a simple Laplacian-based "blobness" score for a crop.

    Rough idea:
      - Blur grayscale crop.
      - Apply Laplacian.
      - Measure positive and negative responses.
      - Score = mean(pos) - 0.6 * mean(neg)

    Higher scores indicate blob-like bright center structures.

    Args:
        crop_bgr: Crop image in BGR.
        sigma: Gaussian blur sigma.

    Returns:
        float: Blobness score (higher is more blob-like).
        """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    g = cv2.GaussianBlur(gray, (0,0), sigma)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    s_pos = float(np.clip(lap, 0, 1).mean())
    s_neg = float(np.clip(-lap, 0, 1).mean())
    return s_pos - 0.6*s_neg


def red_edge_ok(crop_bgr, canny_lo=10, canny_hi=40, min_edge_pct=0.010):
    """
    Filter based on red-channel edge content.

    Args:
        crop_bgr: Crop image in BGR.
        canny_lo: Lower Canny threshold.
        canny_hi: Upper Canny threshold.
        min_edge_pct: Minimum fraction of edge pixels required.

    Returns:
        bool: True if the fraction of red-channel edges exceeds min_edge_pct.
        """
    R = crop_bgr[..., 2]
    rb = cv2.GaussianBlur(R, (0,0), 1.0)
    e = cv2.Canny(rb, canny_lo, canny_hi)
    return (e > 0).mean() > min_edge_pct


def require_gate_proximity(points: np.ndarray, gate_mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """
    Keep only candidate points that lie within a dilated gate mask.

    Args:
        points: (N, 2) array of (x, y) points.
        gate_mask: Boolean or uint8 gate mask (H, W).
        radius: Structuring element radius for dilating the gate.

    Returns:
        np.ndarray: Indices of points that fall within the dilated gate.
        """
    if gate_mask is None or len(points) == 0:
        return np.arange(len(points), dtype=np.int32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    near = cv2.dilate(gate_mask.astype(np.uint8), k) > 0
    keep = []
    for i,(x,y) in enumerate(points.astype(int)):
        if 0 <= y < near.shape[0] and 0 <= x < near.shape[1] and near[y,x]:
            keep.append(i)
    return np.array(keep, dtype=np.int32)


def prune_tight_pairs(points: np.ndarray, scores: np.ndarray, floor=8):
    """
    Greedy pruning of detections that are too close to each other.

    Algorithm:
      - Sort points by descending score.
      - Iterate; for each kept point, suppress any other point within floor pixels.

    Args:
        points: (N, 2) array of (x, y) positions.
        scores: (N,) array of scores.
        floor: Distance threshold in pixels (Euclidean).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - pruned_points: Remaining points.
            - pruned_scores: Corresponding scores.
        """
    if len(points) == 0:
        return points, scores
    keep = []
    used = np.zeros(len(points), bool)
    order = np.argsort(-scores)
    f2 = floor*floor
    for i in order:
        if used[i]: continue
        keep.append(i)
        d = (points[:,0]-points[i,0])**2 + (points[:,1]-points[i,1])**2
        used |= (d <= f2)
        used[i] = True
    keep = np.array(keep, dtype=np.int32)
    return points[keep], scores[keep]


# -------------------- CSV helpers --------------------
def save_detections_csv(csv_path: str, coords_xy: np.ndarray, scores: np.ndarray):
    """
    Save detections to CSV with columns [x, y, score].

    Args:
        csv_path: Output CSV path.
        coords_xy: (N, 2) array of detections.
        scores: (N,) array of detection scores.
        """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "score"])
        if len(coords_xy):
            for (x, y), s in zip(coords_xy, scores):
                writer.writerow([int(x), int(y), float(s)])


def save_artifacts_csv(csv_path: str, artifact_rows: List[Tuple[str, str]]):
    """
    Save artifact paths (debug masks, overlay, csv, etc.) to CSV.

    Each row:
        artifact_type, path

    Args:
        csv_path: Output CSV path.
        artifact_rows: List of (artifact_type, path) pairs.
        """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["artifact_type", "path"])
        for t, p in artifact_rows:
            writer.writerow([t, os.path.abspath(p)])


# -------------------- PNT writer (for DDG) --------------------
def save_detections_pnt(pnt_path: str, image_path: str, coords_xy: np.ndarray):
    """
    Write a DotDotGoose-compatible .pnt (JSON) with detections.

    Args:
        pnt_path: Path to .pnt file to write.
        image_path: Original image path used as key in the .pnt file.
        coords_xy: (N, 2) array of detection coordinates.
        """
    os.makedirs(os.path.dirname(pnt_path), exist_ok=True)
    pkg = {
        "classes": ["output"],
        "points": {image_path: {"output": []}},
        "colors": {"output": [255, 255, 0]},  # yellow
        "metadata": {"survey_id": "", "coordinates": {}},
        "custom_fields": {"fields": [], "data": {}},
        "ui": {
            "grid": {"size": 200, "color": [255, 255, 255]},
            "point": {"radius": 6, "color": [255, 255, 0]}
        }
    }
    for i, (x, y) in enumerate(coords_xy):
        pkg["points"][image_path]["output"].append({"x": float(x), "y": float(y)})

    with open(pnt_path, "w") as f:
        json.dump(pkg, f, indent=2)


# -------------------- main --------------------
def main():
    """
    Entry point for running the microscopy candidate union + gated detection pipeline.

    Parses CLI arguments, runs:
      - Image loading
      - Optional gate construction
      - HSV candidate extraction
      - Candidate union + dedupe
      - Crop extraction
      - Optional pre-CNN FP filters
      - CNN scoring + thresholding + NMS
      - Optional snap-to-yellow refinement + jump safety
      - Post-snap NMS + tight-pair pruning
      - Overlay drawing + CSV + PNT writing

    Outputs are written into --out_dir.
        """
    ap = argparse.ArgumentParser("Microscopy candidate union + gated detection (224, no-norm by default)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", default="experiments/quickcheck")

    # model / preprocessing
    ap.add_argument("--window", type=int, default=224)
    ap.add_argument("--normalize", action="store_true", default=False)  # you trained w/o normalization
    ap.add_argument("--mean", default="0.485,0.456,0.406")
    ap.add_argument("--std", default="0.229,0.224,0.225")
    ap.add_argument("--color_order", choices=["rgb", "bgr"], default="rgb")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # scoring / drawing
    ap.add_argument("--threshold", type=float, default=0.935)
    ap.add_argument("--nms_radius", type=int, default=18)
    ap.add_argument("--circle_radius", type=int, default=8)
    ap.add_argument("--thickness", type=int, default=-1)

    # HSV params (strict by default)
    ap.add_argument("--h_lo", type=int, default=22)
    ap.add_argument("--h_hi", type=int, default=36)
    ap.add_argument("--s_lo", type=int, default=200)
    ap.add_argument("--v_lo", type=int, default=200)
    ap.add_argument("--min_area", type=int, default=10)
    ap.add_argument("--max_area", type=int, default=240)

    # Gate params
    ap.add_argument("--use_edge_gate", action="store_true")
    ap.add_argument("--edge_band_px", type=int, default=5)
    ap.add_argument("--canny_lo", type=int, default=10)
    ap.add_argument("--canny_hi", type=int, default=40)
    ap.add_argument("--green_k", type=float, default=0.25)
    ap.add_argument("--tR", type=float, default=0.55)
    ap.add_argument("--tG", type=float, default=0.55)
    ap.add_argument("--tB", type=float, default=0.40)
    ap.add_argument("--delta_rg", type=float, default=0.22)
    ap.add_argument("--rg_ratio", type=float, default=0.78)
    ap.add_argument("--min_cc_area_gate", type=int, default=6)
    ap.add_argument("--max_cc_area_gate", type=int, default=260)
    ap.add_argument("--keep_border", action="store_true")
    ap.add_argument("--border_px", type=int, default=8)
    ap.add_argument("--debug_masks", action="store_true")

    # Candidate union & snap
    ap.add_argument("--candidate_union_radius", type=int, default=8)
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_search_r", type=int, default=8)
    ap.add_argument("--snap_h_lo", type=int, default=18)
    ap.add_argument("--snap_h_hi", type=int, default=42)
    ap.add_argument("--snap_s_lo", type=int, default=150)
    ap.add_argument("--snap_v_lo", type=int, default=170)
    ap.add_argument("--snap_min_area", type=int, default=8)
    ap.add_argument("--snap_max_area", type=int, default=220)
    ap.add_argument("--snap_jump_limit", type=float, default=9.0)

    # FP filter toggles (OFF by default)
    ap.add_argument("--fp_compactness", action="store_true", default=False)
    ap.add_argument("--fp_blobness", action="store_true", default=False)
    ap.add_argument("--fp_rededge", action="store_true", default=False)
    ap.add_argument("--fp_gate_proximity", action="store_true", default=False)
    ap.add_argument("--post_tight_pair_floor", type=int, default=8)

    # Optional eval helpers
    ap.add_argument("--truth_csv", default=None, help="Optional GT CSV to compare (x,y[,score])")
    ap.add_argument("--match_tol_px", type=float, default=10.0)

    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    artifacts: List[Tuple[str, str]] = []

    img = load_image_bgr(args.image)
    H, W = img.shape[:2]
    print(f"[debug] image loaded: {H}x{W}")

    stem = Path(args.image).stem

    # Build gate
    gate_mask = None
    if args.use_edge_gate:
        dbg_prefix = os.path.join(args.out_dir, f"{stem}")
        gate_mask, _dbg = build_edge_green_yellow_gate(
            img_bgr=img,
            edge_band_px=args.edge_band_px, canny_lo=args.canny_lo, canny_hi=args.canny_hi,
            green_k=args.green_k,
            tR=args.tR, tG=args.tG, tB=args.tB, delta_rg=args.delta_rg, rg_ratio=args.rg_ratio,
            min_cc_area=args.min_cc_area_gate, max_cc_area=args.max_cc_area_gate,
            keep_border=args.keep_border, border_px=args.border_px,
            debug=args.debug_masks, debug_prefix=f"{dbg_prefix}"
        )
        if args.debug_masks:
            for suf in ("edgeband", "green", "yellow", "gatemask"):
                artifacts.append(("debug_mask", f"{dbg_prefix}_{suf}.png"))

    # Candidates = gate centroids U strict-HSV centroids (deduped by small NMS)
    gate_pts: List[Tuple[int,int]] = []
    if gate_mask is not None:
        num, lab, stats, cent = cv2.connectedComponentsWithStats(gate_mask.astype(np.uint8), connectivity=8)
        for i in range(1, num):
            a = stats[i, cv2.CC_STAT_AREA]
            if args.min_cc_area_gate <= a <= args.max_cc_area_gate:
                cx, cy = cent[i]
                gate_pts.append((int(round(cx)), int(round(cy))))

    hsv_pts, yellow_mask = find_yellow_candidates(
        img, h_lo=args.h_lo, h_hi=args.h_hi, s_lo=args.s_lo, v_lo=args.v_lo,
        min_area=args.min_area, max_area=args.max_area
    )

    print(f"[stage] gate centroids: {len(gate_pts)}")
    print(f"[stage] hsv centroids: {len(hsv_pts)}")

    all_pts = (gate_pts or []) + (hsv_pts or [])
    cand_xy: List[Tuple[int,int]] = []
    if len(all_pts):
        P = np.array(all_pts, dtype=np.int32)
        ones = np.ones(len(P), dtype=np.float32)

        # small union NMS to dedupe nearby points
        def nms_radius(points: np.ndarray, scores: np.ndarray, radius: int) -> List[int]:
            """
           Radius-based non-max suppression (NMS) for candidate union.

           Args:
               points: (N, 2) array of points.
               scores: (N,) array of scores.
               radius: Suppression radius in pixels.

           Returns:
               List[int]: Indices of kept points.
                        """
            if len(points) == 0:
                return []
            order = np.argsort(-scores)
            keep_idx, used = [], np.zeros(len(points), dtype=bool)
            rr = radius * radius
            for i in order:
                if used[i]:
                    continue
                keep_idx.append(i)
                dx = points[:, 0] - points[i, 0]
                dy = points[:, 1] - points[i, 1]
                used |= (dx * dx + dy * dy) <= rr
                used[i] = True
            return keep_idx

        keep = nms_radius(P, ones, args.candidate_union_radius)
        cand_xy = [tuple(map(int, P[i])) for i in keep]

    print(f"[stage] total candidates after union+dedupe: {len(cand_xy)}")

    # ---- debug saves ----
    if yellow_mask is not None:
        yellow_path = os.path.join(args.out_dir, f"{stem}_yellowmask.png")
        cv2.imwrite(yellow_path, yellow_mask)
        artifacts.append(("yellowmask", yellow_path))

    cand_vis = img.copy()
    for (x, y) in cand_xy:
        cv2.circle(cand_vis, (int(x), int(y)), 6, (0, 255, 255), -1)
    cand_path = os.path.join(args.out_dir, f"{stem}_candidates.png")
    cv2.imwrite(cand_path, cand_vis)
    artifacts.append(("candidates_vis", cand_path))

    detections_csv_path = os.path.join(args.out_dir, f"{stem}_detections.csv")
    artifacts_csv_path  = os.path.join(args.out_dir, f"{stem}_artifacts.csv")
    pnt_out             = os.path.join(args.out_dir, f"{stem}.pnt")

    if len(cand_xy) == 0:
        out_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
        cv2.imwrite(out_path, img)
        artifacts.append(("overlay", out_path))
        # write CSVs (empty)
        save_detections_csv(detections_csv_path, np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32))
        artifacts.append(("csv", detections_csv_path))
        save_artifacts_csv(artifacts_csv_path, artifacts)
        save_detections_pnt(pnt_out, args.image, np.empty((0, 2), dtype=np.float32))
        print("[OK] Saved (no candidates) and CSVs/PNT.")
        return

    # Build crops
    crops = [center_crop(img, x, y, args.window) for (x, y) in cand_xy]

    # Optional gentle FP filters BEFORE CNN (all OFF by default; enable via flags)
    idx = np.arange(len(crops), dtype=np.int32)

    if args.fp_compactness:
        keep = [i for i,c in enumerate(crops) if dot_compactness_ok(c, 0.003, 0.030, 1.0)]
        crops   = [crops[i] for i in keep]; cand_xy = [cand_xy[i] for i in keep]; idx = idx[keep]

    if args.fp_blobness:
        keep = [i for i,c in enumerate(crops) if (log_blobness_score(c) > 0.010)]
        crops   = [crops[i] for i in keep]; cand_xy = [cand_xy[i] for i in keep]; idx = idx[keep]

    if args.fp_rededge:
        keep = [i for i,c in enumerate(crops) if red_edge_ok(c, args.canny_lo, args.canny_hi, 0.010)]
        crops   = [crops[i] for i in keep]; cand_xy = [cand_xy[i] for i in keep]; idx = idx[keep]

    if args.fp_gate_proximity and gate_mask is not None and len(cand_xy):
        P = np.array(cand_xy, dtype=np.int32)
        keep = require_gate_proximity(P, gate_mask, radius=4)
        crops   = [crops[i] for i in keep]; cand_xy = [cand_xy[i] for i in keep]; idx = idx[keep]

    if len(crops) == 0:
        out_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
        cv2.imwrite(out_path, img)
        artifacts.append(("overlay", out_path))
        save_detections_csv(detections_csv_path, np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32))
        artifacts.append(("csv", detections_csv_path))
        save_artifacts_csv(artifacts_csv_path, artifacts)
        save_detections_pnt(pnt_out, args.image, np.empty((0, 2), dtype=np.float32))
        print("[OK] Saved (all candidates filtered) and CSVs/PNT.")
        return

    # 2) classify crops
    model = load_model(args.weights, device=device, num_classes=1)
    x_tensor = to_tensor_bchw_uint8(crops, args.color_order, args.normalize, args.mean, args.std, device)

    with torch.no_grad():
        logits = model(x_tensor).squeeze()
        if not torch.is_floating_point(logits):
            logits = logits.float()
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    probs = np.asarray(probs, dtype=np.float32)
    print(f"[stage] prob stats – min {probs.min():.3f} / max {probs.max():.3f} / mean {probs.mean():.3f}")
    keep = probs >= args.threshold
    sel_xy = np.array(cand_xy, dtype=np.int32)[keep]
    sel_sc = probs[keep]
    print(f"[stage] kept after threshold {args.threshold}: {len(sel_xy)}")

    # small radius NMS to avoid near-duplicates
    def nms_radius(points: np.ndarray, scores: np.ndarray, radius: int) -> List[int]:
        if len(points) == 0:
            return []
        order = np.argsort(-scores)
        keep_idx, used = [], np.zeros(len(points), dtype=bool)
        rr = radius * radius
        for i in order:
            if used[i]:
                continue
            keep_idx.append(i)
            dx = points[:, 0] - points[i, 0]
            dy = points[:, 1] - points[i, 1]
            used |= (dx * dx + dy * dy) <= rr
            used[i] = True
        return keep_idx

    if len(sel_xy) and args.nms_radius > 0:
        k = nms_radius(sel_xy, sel_sc, args.nms_radius)
        sel_xy = sel_xy[k]
        sel_sc = sel_sc[k]
        print(f"[stage] after pre-snap NMS (r={args.nms_radius}): {len(sel_xy)}")

    # 2b) optional snapping + jump safety
    if len(sel_xy) and args.snap:
        pre_xy = sel_xy.copy()
        snapped_xy, keep_idx = snap_to_yellow_centroids(
            img_bgr=img,
            pts_xy=sel_xy,
            search_r=args.snap_search_r,
            h_lo=args.snap_h_lo, h_hi=args.snap_h_hi,
            s_lo=args.snap_s_lo, v_lo=args.snap_v_lo,
            min_area=args.snap_min_area, max_area=args.snap_max_area
        )
        if len(snapped_xy):
            # discard big jumps
            jump = np.sqrt(((snapped_xy - pre_xy[keep_idx])**2).sum(1))
            safe = jump <= float(args.snap_jump_limit)
            snapped_xy = snapped_xy[safe]; keep_idx = keep_idx[safe]
            sel_xy = snapped_xy.astype(np.int32)
            sel_sc = sel_sc[keep_idx]
            print(f"[stage] after snap kept: {len(sel_xy)}")
        else:
            sel_xy = np.empty((0,2), dtype=np.int32)
            sel_sc = np.empty((0,), dtype=np.float32)
            print(f"[stage] after snap kept: 0")

    # second NMS after snap
    if len(sel_xy) and args.nms_radius > 0:
        k = nms_radius(sel_xy, sel_sc, args.nms_radius)
        sel_xy = sel_xy[k]
        sel_sc = sel_sc[k]
        print(f"[stage] after post-snap NMS (r={args.nms_radius}): {len(sel_xy)}")

    # final tight-pair pruning (8 px default)
    if len(sel_xy) and args.post_tight_pair_floor > 0:
        sel_xy, sel_sc = prune_tight_pairs(sel_xy, sel_sc, floor=args.post_tight_pair_floor)

    # 3) draw overlay
    overlay = img.copy()
    for (x, y) in sel_xy:
        cv2.circle(overlay, (int(x), int(y)), args.circle_radius, (255, 255, 255), args.thickness)

    out_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
    cv2.imwrite(out_path, overlay)
    artifacts.append(("overlay", out_path))

    # 4) write CSVs + PNT
    save_detections_csv(detections_csv_path, sel_xy, sel_sc)
    artifacts.append(("csv", detections_csv_path))
    save_artifacts_csv(artifacts_csv_path, artifacts)
    save_detections_pnt(pnt_out, args.image, sel_xy)

    print(f"[OK] Saved overlay -> {out_path}  (detections: {len(sel_xy)})")
    print(f"[OK] Wrote CSVs: {detections_csv_path}, {artifacts_csv_path}")
    print(f"[OK] Wrote PNT: {pnt_out}")


if __name__ == "__main__":
    main()
