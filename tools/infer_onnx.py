#!/usr/bin/env python3
"""
CenterPoint ONNX inference & visualization script.

Runs full CenterPoint inference (PFE -> Scatter -> RPN -> Decode -> NMS)
entirely in Python using onnxruntime, then visualises results with matplotlib.

All parameters are read from a YAML config (default: config/config_python.yaml).
Individual values can be overridden on the command line.

Usage:
  python infer_onnx.py --config config/config_python.yaml
  python infer_onnx.py --config config/config_python.yaml --data_dir lidars/ --save_dir out/

Output per input bin file:
  <save_dir>/<stem>.txt      - detection results (one box per line)
  <save_dir>/<stem>_bev.png  - BEV visualisation
"""

import os
import sys
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import yaml
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ============================================================
# Config – populated from YAML in main()
# ============================================================
class Cfg:
    """Flat configuration namespace, filled from the YAML file."""
    # point cloud range & grid
    X_MIN, X_MAX = -74.88, 74.88
    Y_MIN, Y_MAX = -74.88, 74.88
    Z_MIN, Z_MAX = -2.0, 4.0
    X_STEP, Y_STEP = 0.32, 0.32
    POINT_DIM = 5
    # pillar encoder
    MAX_PILLARS = 32000
    MAX_PTS_PILLAR = 20
    FEATURE_NUM = 10
    PFE_OUTPUT_DIM = 64
    # BEV grid
    BEV_W = 468
    BEV_H = 468
    # postprocess
    OUT_SIZE_FACTOR = 1.0
    SCORE_THRESH = 0.1
    NMS_THRESH = 0.7
    INPUT_NMS_MAX = 4096
    OUTPUT_NMS_MAX = 500
    OUTPUT_H = 468
    OUTPUT_W = 468
    # RPN output tensor names (role -> name)
    ROLE_MAP = {"246": "reg", "250": "height", "264": "dim",
                "258": "rot", "265": "score", "266": "cls"}
    # model / io / runtime
    PFE_PATH = "models/pfe_baseline32000.onnx"
    RPN_PATH = "models/rpn_baseline.onnx"
    DATA_DIR = "lidars"
    SAVE_DIR = "output/python"
    PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    SAVE_VIZ = True


def load_config(path: str):
    """Load YAML config into the Cfg namespace (missing keys keep defaults)."""
    if not path or not os.path.exists(path):
        print(f"[cfg] config '{path}' not found, using built-in defaults")
        return
    with open(path, "r") as f:
        y = yaml.safe_load(f) or {}

    pc = y.get("point_cloud", {})
    Cfg.X_MIN, Cfg.X_MAX = pc.get("x_min", Cfg.X_MIN), pc.get("x_max", Cfg.X_MAX)
    Cfg.Y_MIN, Cfg.Y_MAX = pc.get("y_min", Cfg.Y_MIN), pc.get("y_max", Cfg.Y_MAX)
    Cfg.Z_MIN, Cfg.Z_MAX = pc.get("z_min", Cfg.Z_MIN), pc.get("z_max", Cfg.Z_MAX)
    Cfg.X_STEP, Cfg.Y_STEP = pc.get("x_step", Cfg.X_STEP), pc.get("y_step", Cfg.Y_STEP)
    Cfg.POINT_DIM = pc.get("point_dim", Cfg.POINT_DIM)

    p = y.get("pillar", {})
    Cfg.MAX_PILLARS = p.get("max_pillars", Cfg.MAX_PILLARS)
    Cfg.MAX_PTS_PILLAR = p.get("max_points_per_pillar", Cfg.MAX_PTS_PILLAR)
    Cfg.FEATURE_NUM = p.get("feature_num", Cfg.FEATURE_NUM)
    Cfg.PFE_OUTPUT_DIM = p.get("pfe_output_dim", Cfg.PFE_OUTPUT_DIM)

    b = y.get("bev", {})
    Cfg.BEV_W = b.get("width", Cfg.BEV_W)
    Cfg.BEV_H = b.get("height", Cfg.BEV_H)

    pp = y.get("postprocess", {})
    Cfg.OUT_SIZE_FACTOR = pp.get("out_size_factor", Cfg.OUT_SIZE_FACTOR)
    Cfg.SCORE_THRESH = pp.get("score_threshold", Cfg.SCORE_THRESH)
    Cfg.NMS_THRESH = pp.get("nms_threshold", Cfg.NMS_THRESH)
    Cfg.INPUT_NMS_MAX = pp.get("input_nms_max_size", Cfg.INPUT_NMS_MAX)
    Cfg.OUTPUT_NMS_MAX = pp.get("output_nms_max_size", Cfg.OUTPUT_NMS_MAX)
    Cfg.OUTPUT_H = pp.get("output_h", Cfg.OUTPUT_H)
    Cfg.OUTPUT_W = pp.get("output_w", Cfg.OUTPUT_W)

    names = y.get("rpn_output_names", {})
    if names:
        Cfg.ROLE_MAP = {str(v): k for k, v in names.items()}

    m = y.get("model", {})
    Cfg.PFE_PATH = m.get("pfe_path", Cfg.PFE_PATH)
    Cfg.RPN_PATH = m.get("rpn_path", Cfg.RPN_PATH)

    rt = y.get("runtime", {})
    Cfg.PROVIDERS = rt.get("providers", Cfg.PROVIDERS)
    Cfg.SAVE_VIZ = rt.get("save_viz", Cfg.SAVE_VIZ)

    io = y.get("io", {})
    Cfg.DATA_DIR = io.get("data_dir", Cfg.DATA_DIR)
    Cfg.SAVE_DIR = io.get("save_dir", Cfg.SAVE_DIR)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="CenterPoint ONNX inference")
    p.add_argument("--config", default="config/config_python.yaml",
                   help="Path to YAML config")
    p.add_argument("--pfe_onnx", help="override model.pfe_path")
    p.add_argument("--rpn_onnx", help="override model.rpn_path")
    p.add_argument("--data_dir", help="override io.data_dir")
    p.add_argument("--save_dir", help="override io.save_dir")
    p.add_argument("--score_thresh", type=float, help="override postprocess.score_threshold")
    p.add_argument("--nms_thresh", type=float, help="override postprocess.nms_threshold")
    p.add_argument("--providers", nargs="+", help="override runtime.providers")
    p.add_argument("--no_viz", action="store_true", help="disable BEV PNG output")
    return p.parse_args()


# ============================================================
# I/O
# ============================================================
def read_bin(path: str) -> np.ndarray:
    """Load Waymo LiDAR binary file -> (N, point_dim) float32."""
    data = np.fromfile(path, dtype=np.float32)
    if data.size % Cfg.POINT_DIM:
        raise ValueError(f"File size {data.size} not divisible by POINT_DIM={Cfg.POINT_DIM}")
    return data.reshape(-1, Cfg.POINT_DIM)


# ============================================================
# Preprocessing (mirrors CenterPointDetector::voxelise)
# ============================================================
def voxelize(points: np.ndarray):
    """
    Convert raw point cloud to pillar features.

    Returns
    -------
    pillars  : (MAX_PILLARS, MAX_PTS_PILLAR, FEATURE_NUM) float32
    indices  : (MAX_PILLARS,) int32  - flat BEV index for each occupied pillar
    n_pillars: int  - number of occupied pillars (<= MAX_PILLARS)
    """
    mask = (
        (points[:, 0] >= Cfg.X_MIN) & (points[:, 0] < Cfg.X_MAX) &
        (points[:, 1] >= Cfg.Y_MIN) & (points[:, 1] < Cfg.Y_MAX) &
        (points[:, 2] >= Cfg.Z_MIN) & (points[:, 2] < Cfg.Z_MAX)
    )
    pts = points[mask]

    pillars = np.zeros((Cfg.MAX_PILLARS, Cfg.MAX_PTS_PILLAR, Cfg.FEATURE_NUM), dtype=np.float32)
    indices = np.zeros(Cfg.MAX_PILLARS, dtype=np.int32)

    if len(pts) == 0:
        return pillars, indices, 0

    bev_x = np.floor((pts[:, 0] - Cfg.X_MIN) / Cfg.X_STEP).astype(np.int32)
    bev_y = np.floor((pts[:, 1] - Cfg.Y_MIN) / Cfg.Y_STEP).astype(np.int32)
    bev_x = np.clip(bev_x, 0, Cfg.BEV_W - 1)
    bev_y = np.clip(bev_y, 0, Cfg.BEV_H - 1)
    flat = bev_y * Cfg.BEV_W + bev_x

    sort_order = np.argsort(flat, kind="stable")
    flat_s = flat[sort_order]
    bev_x_s = bev_x[sort_order]
    bev_y_s = bev_y[sort_order]
    pts_s = pts[sort_order]

    uniq, first, counts = np.unique(flat_s, return_index=True, return_counts=True)
    n_pillars = min(len(uniq), Cfg.MAX_PILLARS)

    for i in range(n_pillars):
        start = first[i]
        count = min(int(counts[i]), Cfg.MAX_PTS_PILLAR)
        p = pts_s[start: start + count]

        px = int(bev_x_s[start])
        py = int(bev_y_s[start])
        pc_x = (px + 0.5) * Cfg.X_STEP + Cfg.X_MIN
        pc_y = (py + 0.5) * Cfg.Y_STEP + Cfg.Y_MIN

        mx, my, mz = p[:, 0].mean(), p[:, 1].mean(), p[:, 2].mean()

        pillars[i, :count, 0] = p[:, 0]
        pillars[i, :count, 1] = p[:, 1]
        pillars[i, :count, 2] = p[:, 2]
        pillars[i, :count, 3] = p[:, 3]
        pillars[i, :count, 4] = p[:, 4] if Cfg.POINT_DIM > 4 else 0.0
        pillars[i, :count, 5] = p[:, 0] - mx
        pillars[i, :count, 6] = p[:, 1] - my
        pillars[i, :count, 7] = p[:, 2] - mz
        pillars[i, :count, 8] = p[:, 0] - pc_x
        pillars[i, :count, 9] = p[:, 1] - pc_y

        indices[i] = int(flat_s[start])

    return pillars, indices, n_pillars


def scatter(pfe_out: np.ndarray, indices: np.ndarray, n_pillars: int) -> np.ndarray:
    """Scatter PFE features onto the BEV grid -> (1, PFE_OUTPUT_DIM, BEV_H, BEV_W)."""
    bev = np.zeros((1, Cfg.PFE_OUTPUT_DIM, Cfg.BEV_H, Cfg.BEV_W), dtype=np.float32)
    for i in range(n_pillars):
        idx = int(indices[i])
        if 0 <= idx < Cfg.BEV_H * Cfg.BEV_W:
            y = idx // Cfg.BEV_W
            x = idx % Cfg.BEV_W
            bev[0, :, y, x] = pfe_out[i]
    return bev


# ============================================================
# Postprocessing
# ============================================================
def decode_boxes(reg, height, dim, rot, score, cls, score_thresh):
    """Decode RPN heads to a list of box dicts above `score_thresh`."""
    H, W = score.shape[1], score.shape[2]
    yg, xg = np.mgrid[0:H, 0:W]

    s = score[0]
    valid = s > score_thresh
    if not valid.any():
        return []

    xi = xg[valid]
    yi = yg[valid]
    sv = s[valid]

    x = (xi + reg[0, 0][valid]) * Cfg.OUT_SIZE_FACTOR * Cfg.X_STEP + Cfg.X_MIN
    y = (yi + reg[0, 1][valid]) * Cfg.OUT_SIZE_FACTOR * Cfg.Y_STEP + Cfg.Y_MIN
    z = height[0, 0][valid]
    # dim head channel order is (width, length, height)
    w = dim[0, 0][valid]
    l = dim[0, 1][valid]
    h_val = dim[0, 2][valid]
    # rot channels are (cos, sin) for this model -> heading = atan2(ch1, ch0)
    theta = np.arctan2(rot[0, 1][valid], rot[0, 0][valid])
    c = cls[0][valid].astype(np.int32)

    boxes = []
    for i in range(len(x)):
        boxes.append(dict(
            x=float(x[i]), y=float(y[i]), z=float(z[i]),
            l=float(l[i]), h=float(h_val[i]), w=float(w[i]),
            theta=float(theta[i]),
            score=float(sv[i]), cls=int(c[i]),
        ))
    return boxes


def _box_corners(box):
    """Return 4x2 corners of a box's oriented BEV footprint."""
    return _box_corners_bev(box)


def _poly_area(p):
    """Shoelace area of a polygon given as an (N,2) array."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_poly(subj, clip):
    """Clip convex polygon `subj` against convex quad `clip` (Sutherland-Hodgman)."""
    sign = 1.0 if _signed_area(clip) >= 0 else -1.0
    out = list(subj)
    for e in range(len(clip)):
        if not out:
            break
        a = clip[e]
        b = clip[(e + 1) % len(clip)]
        ex, ey = b[0] - a[0], b[1] - a[1]
        inp = out
        out = []
        n = len(inp)
        for i in range(n):
            p = inp[i]
            q = inp[(i + 1) % n]
            dp = sign * (ex * (p[1] - a[1]) - ey * (p[0] - a[0]))
            dq = sign * (ex * (q[1] - a[1]) - ey * (q[0] - a[0]))
            if dp >= 0:
                out.append(p)
            if (dp >= 0) != (dq >= 0):
                t = dp / (dp - dq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _signed_area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return s


def _iou_bev_rotated(a, b):
    """Oriented (rotated) BEV IoU between two boxes."""
    ca = _box_corners(a)
    cb = _box_corners(b)
    inter_poly = _clip_poly([tuple(p) for p in ca], [tuple(p) for p in cb])
    if len(inter_poly) < 3:
        return 0.0
    inter = _poly_area(np.asarray(inter_poly))
    if inter <= 0:
        return 0.0
    union = a["l"] * a["w"] + b["l"] * b["w"] - inter
    return inter / union if union > 1e-6 else 0.0


def nms_bev(boxes, iou_thresh, max_before, max_after):
    """Greedy BEV NMS using oriented (rotated) IoU."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)[:max_before]
    keep = []
    suppressed = [False] * len(boxes)
    for i, bi in enumerate(boxes):
        if suppressed[i]:
            continue
        keep.append(bi)
        if len(keep) >= max_after:
            break
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            if _iou_bev_rotated(bi, boxes[j]) > iou_thresh:
                suppressed[j] = True
    return keep


# ============================================================
# Visualisation
# ============================================================
CLS_COLORS = {
    0: "#FF4444",   # vehicle / car
    1: "#4444FF",   # pedestrian
    2: "#44CC44",   # cyclist
    3: "#FFCC00",   # truck
    4: "#CC44CC",   # bus
}
CLS_NAMES = {0: "veh", 1: "ped", 2: "cyc", 3: "truck", 4: "bus"}


def _box_corners_bev(box):
    """Return 4x2 corners of an oriented bounding box in BEV."""
    x, y, l, w, theta = box["x"], box["y"], box["l"], box["w"], box["theta"]
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = l / 2, w / 2
    offsets = np.array([[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]])
    rot = np.array([[c, -s], [s, c]])
    return (offsets @ rot.T) + np.array([[x, y]])


def visualize(points: np.ndarray, boxes: list, save_path: str, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # ---- left: BEV overview ----
    ax = axes[0]
    mask = (
        (points[:, 0] >= Cfg.X_MIN) & (points[:, 0] < Cfg.X_MAX) &
        (points[:, 1] >= Cfg.Y_MIN) & (points[:, 1] < Cfg.Y_MAX)
    )
    pts = points[mask]
    z_norm = np.clip((pts[:, 2] - Cfg.Z_MIN) / (Cfg.Z_MAX - Cfg.Z_MIN), 0, 1)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=z_norm, cmap="viridis",
                    s=0.15, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Height (norm.)")

    patches_bev, colors_bev = [], []
    for box in boxes:
        patches_bev.append(Polygon(_box_corners_bev(box), closed=True))
        colors_bev.append(CLS_COLORS.get(box["cls"], "#FFFFFF"))
    ax.add_collection(PatchCollection(patches_bev, facecolor="none",
                                      edgecolors=colors_bev, linewidths=1.5))

    ax.set_xlim(Cfg.X_MIN, Cfg.X_MAX)
    ax.set_ylim(Cfg.Y_MIN, Cfg.Y_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"BEV - {title}  ({len(boxes)} detections)")

    legend_handles = [
        mpatches.Patch(color=c, label=n)
        for n, c in [(CLS_NAMES.get(k, str(k)), v) for k, v in CLS_COLORS.items()]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    # ---- right: zoomed front view ----
    ax2 = axes[1]
    z_norm2 = np.clip((points[:, 2] - Cfg.Z_MIN) / (Cfg.Z_MAX - Cfg.Z_MIN), 0, 1)
    sc2 = ax2.scatter(points[:, 0], points[:, 1], c=z_norm2, cmap="viridis",
                      s=0.5, alpha=0.5, rasterized=True)
    plt.colorbar(sc2, ax=ax2, label="Height (norm.)")

    patches2, colors2 = [], []
    for box in boxes:
        patches2.append(Polygon(_box_corners_bev(box), closed=True))
        colors2.append(CLS_COLORS.get(box["cls"], "#FFFFFF"))
    ax2.add_collection(PatchCollection(patches2, facecolor="none",
                                       edgecolors=colors2, linewidths=2.0))
    for box in boxes:
        if -50 < box["x"] < 50 and 0 < box["y"] < 60:
            ax2.text(box["x"], box["y"], f"{box['score']:.2f}",
                     fontsize=5, color="white", ha="center")

    ax2.set_xlim(-50, 50)
    ax2.set_ylim(0, 60)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Front view (y in [0,60])")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] saved: {save_path}")


# ============================================================
# Save text results
# ============================================================
def save_txt(boxes: list, path: str):
    with open(path, "w") as f:
        for b in boxes:
            f.write(
                f"{b['x']:.4f} {b['y']:.4f} {b['z']:.4f} "
                f"{b['l']:.4f} {b['h']:.4f} {b['w']:.4f} "
                f"0.0 0.0 {b['theta']:.4f} {b['score']:.4f} {b['cls']}\n"
            )


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    load_config(args.config)

    # CLI overrides
    if args.pfe_onnx:     Cfg.PFE_PATH = args.pfe_onnx
    if args.rpn_onnx:     Cfg.RPN_PATH = args.rpn_onnx
    if args.data_dir:     Cfg.DATA_DIR = args.data_dir
    if args.save_dir:     Cfg.SAVE_DIR = args.save_dir
    if args.score_thresh is not None: Cfg.SCORE_THRESH = args.score_thresh
    if args.nms_thresh is not None:   Cfg.NMS_THRESH = args.nms_thresh
    if args.providers:    Cfg.PROVIDERS = args.providers
    if args.no_viz:       Cfg.SAVE_VIZ = False

    os.makedirs(Cfg.SAVE_DIR, exist_ok=True)

    avail = ort.get_available_providers()
    providers = [p for p in Cfg.PROVIDERS if p in avail] or ["CPUExecutionProvider"]
    print(f"[ort] using providers: {providers}")

    # Disable graph optimisation: some ORT versions apply an over-eager Gemm
    # fusion to the PFE graph that fails shape inference on the (P, N, C) input.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    pfe_sess = ort.InferenceSession(Cfg.PFE_PATH, sess_options=so, providers=providers)
    rpn_sess = ort.InferenceSession(Cfg.RPN_PATH, sess_options=so, providers=providers)

    pfe_in_name = pfe_sess.get_inputs()[0].name
    pfe_out_name = pfe_sess.get_outputs()[0].name
    rpn_in_name = rpn_sess.get_inputs()[0].name
    rpn_out_names = [o.name for o in rpn_sess.get_outputs()]
    print(f"[ort] PFE: {pfe_in_name} -> {pfe_out_name}")
    print(f"[ort] RPN: {rpn_in_name} -> {rpn_out_names}")

    bin_files = sorted(glob.glob(os.path.join(Cfg.DATA_DIR, "seq_*.bin")))
    if not bin_files:
        bin_files = sorted(glob.glob(os.path.join(Cfg.DATA_DIR, "*.bin")))
    if not bin_files:
        sys.exit(f"[error] no .bin files found in {Cfg.DATA_DIR}")
    print(f"[main] processing {len(bin_files)} files ...")

    total_t = {"pre": 0, "pfe": 0, "scatter": 0, "rpn": 0, "post": 0}

    for bin_path in bin_files:
        stem = Path(bin_path).stem
        print(f"\n[{stem}]")

        t0 = time.perf_counter()
        points = read_bin(bin_path)
        pillars, indices, n_pillars = voxelize(points)
        total_t["pre"] += time.perf_counter() - t0
        print(f"  points={len(points)}, pillars={n_pillars}")

        t0 = time.perf_counter()
        (pfe_out,) = pfe_sess.run([pfe_out_name], {pfe_in_name: pillars})
        total_t["pfe"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        bev_feat = scatter(pfe_out, indices, n_pillars)
        total_t["scatter"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        rpn_outs = rpn_sess.run(None, {rpn_in_name: bev_feat})
        total_t["rpn"] += time.perf_counter() - t0

        # map outputs to roles using config names; fall back to index order
        role2tensor = {}
        for name, tensor in zip(rpn_out_names, rpn_outs):
            role = Cfg.ROLE_MAP.get(name)
            if role:
                role2tensor[role] = tensor
        if len(role2tensor) < 6:
            for role, tensor in zip(["reg", "height", "dim", "rot", "score", "cls"], rpn_outs):
                role2tensor.setdefault(role, tensor)

        t0 = time.perf_counter()
        raw_boxes = decode_boxes(
            reg=role2tensor["reg"], height=role2tensor["height"],
            dim=role2tensor["dim"], rot=role2tensor["rot"],
            score=role2tensor["score"], cls=role2tensor["cls"],
            score_thresh=Cfg.SCORE_THRESH,
        )
        boxes = nms_bev(raw_boxes, Cfg.NMS_THRESH, Cfg.INPUT_NMS_MAX, Cfg.OUTPUT_NMS_MAX)
        total_t["post"] += time.perf_counter() - t0
        print(f"  raw={len(raw_boxes)}, after_nms={len(boxes)}")

        save_txt(boxes, os.path.join(Cfg.SAVE_DIR, f"{stem}.txt"))
        if Cfg.SAVE_VIZ:
            visualize(points, boxes, os.path.join(Cfg.SAVE_DIR, f"{stem}_bev.png"), title=stem)

    n = len(bin_files)
    if n:
        print("\n=== timing (ms/frame) ===")
        for k, v in total_t.items():
            print(f"  {k:<10}: {v / n * 1000:.2f} ms")


if __name__ == "__main__":
    main()
