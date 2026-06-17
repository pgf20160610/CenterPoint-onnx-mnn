#!/usr/bin/env python3
"""
CenterPoint model conversion script.

Supports:
  1. Auto-download a pretrained PyTorch checkpoint
  2. Export PyTorch → ONNX (PFE + RPN separately)
  3. Verify ONNX models with onnxruntime
  4. Convert ONNX → MNN via MNNConvert CLI or Python MNN SDK

Datasets: waymo (default) | kitti

Usage examples:
  # Waymo: export to ONNX from a local checkpoint
  python convert_model.py --ckpt weights/centerpoint.pth --to_onnx

  # KITTI: download + export to ONNX + convert to MNN
  python convert_model.py --dataset kitti --download --model_url <url> --to_onnx --to_mnn

  # KITTI: export from a local checkpoint
  python convert_model.py --dataset kitti --ckpt weights/kitti_centerpoint.pth --to_onnx

  # Only convert existing ONNX to MNN
  python convert_model.py --pfe_onnx models/pfe.onnx --rpn_onnx models/rpn.onnx --to_mnn
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import shutil
from pathlib import Path


def load_runtime_config(path: str):
    """Read output paths / max_pillars from the runtime YAML config.

    Returns a dict with keys pfe_path, rpn_path, max_pillars (any may be None).
    Missing file or missing pyyaml degrades gracefully to an empty dict.
    """
    out = {"pfe_path": None, "rpn_path": None, "max_pillars": None}
    if not path or not os.path.exists(path):
        return out
    try:
        import yaml
    except ImportError:
        print("[cfg] pyyaml not installed, ignoring runtime config")
        return out
    with open(path, "r") as f:
        y = yaml.safe_load(f) or {}
    model = y.get("model", {})
    out["pfe_path"] = model.get("pfe_path")
    out["rpn_path"] = model.get("rpn_path")
    out["max_pillars"] = y.get("pillar", {}).get("max_pillars")
    return out


# KITTI pretrained model: download from the CenterPoint model zoo
# https://github.com/tianweiy/CenterPoint  (see README → Model Zoo)
KITTI_DEFAULTS = {
    "config": "tools/kitti_centerpoint_pp.py",
    "runtime_config": "config/config_kitti.yaml",
    "max_pillars": 12000,
    "pfe_onnx_name": "kitti_pfe.onnx",
    "rpn_onnx_name": "kitti_rpn.onnx",
    "export_script": "export_onnx_kitti.py",   # relative to tools/
}

WAYMO_DEFAULTS = {
    "config": "tools/waymo_centerpoint_pp_two_pfn_stride1_3x.py",
    "runtime_config": "config/config_python.yaml",
    "max_pillars": 32000,
    "pfe_onnx_name": "pfe_baseline32000.onnx",
    "rpn_onnx_name": "rpn_baseline.onnx",
    "export_script": "export_onnx.py",         # relative to tools/
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CenterPoint model conversion tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Dataset preset
    parser.add_argument("--dataset", type=str, default="waymo",
                        choices=["waymo", "kitti"],
                        help="Dataset preset: sets default config, paths, and export script "
                             "(waymo | kitti). Individual flags override these defaults.")

    # Source
    src = parser.add_argument_group("source")
    src.add_argument("--ckpt", type=str, default="",
                     help="Path to existing PyTorch checkpoint (.pth)")
    src.add_argument("--download", action="store_true",
                     help="Download checkpoint from --model_url before converting")
    src.add_argument("--model_url", type=str, default="",
                     help="HTTP(S) URL to download checkpoint from")
    src.add_argument("--pfe_onnx", type=str, default="",
                     help="Use existing PFE ONNX file (skips PyTorch export)")
    src.add_argument("--rpn_onnx", type=str, default="",
                     help="Use existing RPN ONNX file (skips PyTorch export)")

    # Model config
    cfg = parser.add_argument_group("model config")
    cfg.add_argument("--config", type=str, default="",
                     help="Det3d config file for the model architecture "
                          "(default: derived from --dataset)")
    cfg.add_argument("--runtime_config", type=str, default="",
                     help="Runtime YAML config; supplies default ONNX output paths "
                          "and max_pillars so converted models match the inference pipeline "
                          "(default: derived from --dataset)")
    cfg.add_argument("--max_pillars", type=int, default=None,
                     help="Override max_pillars (default: from runtime config or dataset preset)")

    # Output
    out = parser.add_argument_group("output")
    out.add_argument("--output_dir", type=str, default="models",
                     help="Directory to write converted models into")
    out.add_argument("--to_onnx", action="store_true", default=False,
                     help="Export / verify ONNX models")
    out.add_argument("--to_mnn", action="store_true", default=False,
                     help="Convert ONNX models to MNN format")
    out.add_argument("--mnn_convert", type=str, default="MNNConvert",
                     help="Path or name of the MNNConvert executable")
    return parser.parse_args()


def _apply_dataset_defaults(args):
    """Fill in config / runtime_config / max_pillars from the dataset preset when not set."""
    preset = KITTI_DEFAULTS if args.dataset == "kitti" else WAYMO_DEFAULTS
    if not args.config:
        args.config = preset["config"]
    if not args.runtime_config:
        args.runtime_config = preset["runtime_config"]
    if args.max_pillars is None:
        args.max_pillars = preset["max_pillars"]
    args._preset = preset


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_checkpoint(url: str, save_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    print(f"[download] {url} → {save_path}")

    def _hook(count, block, total):
        pct = count * block * 100 // total if total > 0 else 0
        bar = "#" * (pct // 5)
        print(f"\r  [{bar:<20}] {min(pct, 100):3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, save_path, reporthook=_hook)
    except Exception as e:
        # Try wget as fallback
        print(f"\n[download] urllib failed ({e}), trying wget …")
        ret = subprocess.run(["wget", "-q", "--show-progress", "-O", save_path, url])
        if ret.returncode != 0:
            raise RuntimeError(f"Download failed: {url}")
    print(f"\n[download] saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(ckpt: str, config: str, max_pillars: int,
                pfe_path: str, rpn_path: str, export_script: str = ""):
    """Call the appropriate export_onnx*.py and return (pfe_path, rpn_path)."""
    for p in (pfe_path, rpn_path):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    if not export_script:
        export_script = os.path.join(os.path.dirname(__file__), "export_onnx.py")
    if not os.path.exists(export_script):
        raise FileNotFoundError(f"Export script not found: {export_script}")

    cmd = [
        sys.executable, export_script,
        "--config", config,
        "--ckpt", ckpt,
        "--pfe_save_path", pfe_path,
        "--rpn_save_path", rpn_path,
    ]
    # export_onnx_kitti.py accepts --max_pillars directly
    if "kitti" in export_script:
        cmd += ["--max_pillars", str(max_pillars)]

    print(f"[onnx-export] running: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError("ONNX export failed")

    return pfe_path, rpn_path


# ---------------------------------------------------------------------------
# ONNX verification
# ---------------------------------------------------------------------------

def verify_onnx(path: str) -> bool:
    """Check model integrity and print shapes using onnx + onnxruntime."""
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
    except ImportError:
        print("[verify] onnx not installed, skipping graph check")
    except Exception as e:
        print(f"[verify] ONNX graph check failed: {e}")
        return False

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print(f"[verify] {path}")
        for inp in sess.get_inputs():
            print(f"   IN  {inp.name}: {inp.shape} [{inp.type}]")
        for out in sess.get_outputs():
            print(f"   OUT {out.name}: {out.shape} [{out.type}]")
        return True
    except ImportError:
        print("[verify] onnxruntime not installed, skipping runtime check")
        return True
    except Exception as e:
        print(f"[verify] onnxruntime check failed: {e}")
        return False


# ---------------------------------------------------------------------------
# MNN conversion
# ---------------------------------------------------------------------------

def _mnn_convert_cli(onnx_path: str, mnn_path: str, tool: str):
    cmd = [
        tool,
        "-f", "ONNX",
        "--modelFile", onnx_path,
        "--MNNModel", mnn_path,
        "--bizCode", "MNN",
    ]
    print(f"[mnn] running: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"MNNConvert returned {ret.returncode}")


def _mnn_convert_python(onnx_path: str, mnn_path: str):
    """Fallback: use MNN Python package for conversion."""
    try:
        import MNN.tools.mnnconvert as mc  # type: ignore
        mc.convert(model_file=onnx_path, MNN_model_file=mnn_path, input_format="ONNX")
    except ImportError:
        raise RuntimeError(
            "MNN Python package not installed. "
            "Install it (pip install MNN) or provide --mnn_convert path."
        )


def convert_to_mnn(onnx_path: str, mnn_convert_tool: str) -> str:
    mnn_path = onnx_path.replace(".onnx", ".mnn")
    if shutil.which(mnn_convert_tool):
        _mnn_convert_cli(onnx_path, mnn_path, mnn_convert_tool)
    else:
        print(f"[mnn] '{mnn_convert_tool}' not found in PATH, trying Python MNN SDK …")
        _mnn_convert_python(onnx_path, mnn_path)
    print(f"[mnn] saved: {mnn_path}")
    return mnn_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _apply_dataset_defaults(args)
    os.makedirs(args.output_dir, exist_ok=True)

    rt = load_runtime_config(args.runtime_config)
    max_pillars = args.max_pillars or rt["max_pillars"] or 32000

    pfe_onnx = args.pfe_onnx
    rpn_onnx = args.rpn_onnx

    # ---- acquire checkpoint ----
    if not (pfe_onnx and rpn_onnx):
        ckpt = args.ckpt
        if args.download:
            if not args.model_url:
                sys.exit(
                    "[error] --download requires --model_url.\n"
                    "  Example: --model_url https://example.com/centerpoint.pth\n"
                    "  KITTI models: see https://github.com/tianweiy/CenterPoint (Model Zoo)"
                )
            ckpt_name = f"{args.dataset}_centerpoint.pth"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                print(f"[download] using cached checkpoint: {ckpt_path}")
            else:
                download_checkpoint(args.model_url, ckpt_path)
            ckpt = ckpt_path

        if not ckpt:
            sys.exit(
                "[error] No checkpoint provided.\n"
                "  Use --ckpt <path> or --download --model_url <url>.\n"
                "  If you already have ONNX files, use --pfe_onnx and --rpn_onnx."
            )

    # ---- ONNX export ----
    if args.to_onnx and not (pfe_onnx and rpn_onnx):
        # Prefer paths from the runtime config; fall back to dataset-preset filenames.
        preset = args._preset
        pfe_out = (rt["pfe_path"]
                   or os.path.join(args.output_dir, preset["pfe_onnx_name"]))
        rpn_out = (rt["rpn_path"]
                   or os.path.join(args.output_dir, preset["rpn_onnx_name"]))
        export_script = os.path.join(os.path.dirname(__file__), preset["export_script"])
        pfe_onnx, rpn_onnx = export_onnx(
            ckpt, args.config, max_pillars, pfe_out, rpn_out,
            export_script=export_script,
        )

    # ---- verify ----
    ok = True
    if pfe_onnx and os.path.exists(pfe_onnx):
        ok &= verify_onnx(pfe_onnx)
    if rpn_onnx and os.path.exists(rpn_onnx):
        ok &= verify_onnx(rpn_onnx)
    if not ok:
        print("[warn] one or more ONNX models failed verification")

    # ---- MNN conversion ----
    if args.to_mnn:
        for onnx_path in [pfe_onnx, rpn_onnx]:
            if onnx_path and os.path.exists(onnx_path):
                convert_to_mnn(onnx_path, args.mnn_convert)
            else:
                print(f"[mnn] skipping {onnx_path} (not found)")

    # ---- summary ----
    print("\n=== conversion complete ===")
    for label, path in [("PFE ONNX", pfe_onnx), ("RPN ONNX", rpn_onnx)]:
        if path:
            status = "OK" if os.path.exists(path) else "MISSING"
            print(f"  {label:<12}: {path}  [{status}]")
    if args.to_mnn:
        for label, path in [("PFE MNN", pfe_onnx), ("RPN MNN", rpn_onnx)]:
            if path:
                mnn = path.replace(".onnx", ".mnn")
                status = "OK" if os.path.exists(mnn) else "MISSING"
                print(f"  {label:<12}: {mnn}  [{status}]")


if __name__ == "__main__":
    main()
