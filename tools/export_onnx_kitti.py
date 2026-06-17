#!/usr/bin/env python3
"""
Export CenterPoint KITTI model to ONNX without requiring a KITTI dataset.

Dummy tensors of the correct shape are synthesized from the model config,
so this script can run on any machine that has the checkpoint and det3d installed.

Usage:
    python tools/export_onnx_kitti.py \\
        --config tools/kitti_centerpoint_pp.py \\
        --ckpt  weights/kitti_centerpoint.pth \\
        --pfe_save_path models/kitti_pfe.onnx \\
        --rpn_save_path models/kitti_rpn.onnx
"""

import argparse
import sys
import os

import torch
from torch import nn

# det3d must be on PYTHONPATH (see tools/setup_det3d.sh)
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint


class _RPN(nn.Module):
    """Wraps neck + bbox_head for a single ONNX-exportable forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bev_map):
        x = self.model.neck(bev_map)
        preds = self.model.bbox_head(x)
        out = {}
        for task_preds in preds:
            hm = torch.sigmoid(task_preds["hm"])
            scores, labels = torch.max(hm, dim=1)
            out["score"] = scores
            out["cls"] = labels
            out["reg"] = task_preds["reg"]
            out["height"] = task_preds["height"]
            out["dim"] = torch.exp(task_preds["dim"])
            out["rot"] = task_preds["rot"]
        return out["reg"], out["height"], out["dim"], out["rot"], out["score"], out["cls"]


def parse_args():
    p = argparse.ArgumentParser(description="Export KITTI CenterPoint to ONNX")
    p.add_argument("--config", default="tools/kitti_centerpoint_pp.py",
                   help="Det3d model config (kitti_centerpoint_pp.py)")
    p.add_argument("--ckpt", required=True, help="PyTorch checkpoint (.pth)")
    p.add_argument("--pfe_save_path", default="models/kitti_pfe.onnx")
    p.add_argument("--rpn_save_path", default="models/kitti_rpn.onnx")
    p.add_argument("--max_pillars", type=int, default=12000)
    p.add_argument("--max_points", type=int, default=100,
                   help="max_points_per_pillar")
    p.add_argument("--feature_num", type=int, default=9,
                   help="PFE input feature dim (9 for KITTI 4-dim points)")
    p.add_argument("--pfe_output_dim", type=int, default=64)
    p.add_argument("--bev_h", type=int, default=496)
    p.add_argument("--bev_w", type=int, default=432)
    return p.parse_args()


def export_pfe(model, args, device):
    """Export PillarFeatureNet (PFE) to ONNX."""
    os.makedirs(os.path.dirname(os.path.abspath(args.pfe_save_path)), exist_ok=True)

    pfe_input = torch.zeros(
        (args.max_pillars, args.max_points, args.feature_num),
        dtype=torch.float32, device=device,
    )
    num_points = torch.ones(args.max_pillars, dtype=torch.int32, device=device)
    coordinates = torch.zeros(
        (args.max_pillars, 4), dtype=torch.int32, device=device
    )

    torch.onnx.export(
        model.reader,
        (pfe_input, num_points, coordinates),
        args.pfe_save_path,
        opset_version=11,
        input_names=["pfe_input", "num_points", "coordinates"],
        output_names=["pfe_output"],
        dynamic_axes={
            "pfe_input": {0: "num_pillars"},
            "num_points": {0: "num_pillars"},
            "coordinates": {0: "num_pillars"},
            "pfe_output": {0: "num_pillars"},
        },
    )
    print(f"[pfe] saved: {args.pfe_save_path}")


def export_rpn(model, args, device):
    """Export Neck + BBoxHead (RPN) to ONNX with named outputs."""
    os.makedirs(os.path.dirname(os.path.abspath(args.rpn_save_path)), exist_ok=True)

    rpn_wrapper = _RPN(model).to(device).eval()
    bev_input = torch.zeros(
        (1, args.pfe_output_dim, args.bev_h, args.bev_w),
        dtype=torch.float32, device=device,
    )

    torch.onnx.export(
        rpn_wrapper,
        bev_input,
        args.rpn_save_path,
        opset_version=11,
        input_names=["bev_map"],
        output_names=["reg", "height", "dim", "rot", "score", "cls"],
    )
    print(f"[rpn] saved: {args.rpn_save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config.fromfile(args.config)
    cfg.EXPORT_ONNX = True

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model.eval().to(device)

    with torch.no_grad():
        export_pfe(model, args, device)
        export_rpn(model, args, device)

    print("\n=== KITTI ONNX export complete ===")
    print(f"  PFE : {args.pfe_save_path}")
    print(f"  RPN : {args.rpn_save_path}")


if __name__ == "__main__":
    main()
