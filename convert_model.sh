#!/usr/bin/env bash
# CenterPoint 模型转换：PyTorch -> ONNX -> MNN
# 支持 Waymo（默认）和 KITTI 数据集。
# 输出路径与 max_pillars 默认取自对应的 YAML 配置，
# 使转换出的模型正好落在推理流程预期的位置。
#
# ---- Waymo 用法 ----
#   # 从已有 checkpoint 导出 ONNX 并转 MNN
#   bash convert_model.sh --ckpt latest.pth --to_onnx --to_mnn
#
#   # 自动下载 checkpoint 后导出
#   bash convert_model.sh --download --model_url <URL> --to_onnx --to_mnn
#
#   # 仅把已有 ONNX 转为 MNN
#   bash convert_model.sh --pfe_onnx models/pfe_baseline32000.onnx \
#                         --rpn_onnx models/rpn_baseline.onnx --to_mnn
#
# ---- KITTI 用法 ----
#   # 从已有 KITTI checkpoint 导出 ONNX
#   bash convert_model.sh --dataset kitti \
#                         --ckpt weights/kitti_centerpoint.pth --to_onnx
#
#   # 自动下载 KITTI checkpoint 并导出 ONNX，再转 MNN
#   bash convert_model.sh --dataset kitti \
#                         --download --model_url <URL> --to_onnx --to_mnn
#
#   # 仅把已有 KITTI ONNX 转为 MNN
#   bash convert_model.sh --dataset kitti \
#                         --pfe_onnx models/kitti_pfe.onnx \
#                         --rpn_onnx models/kitti_rpn.onnx --to_mnn
#
# KITTI 预训练权重下载地址见官方 Model Zoo：
#   https://github.com/tianweiy/CenterPoint
#
# 其余参数将原样转发给 tools/convert_model.py。
set -e

PYTHON=${PYTHON:-python3}
# MNNConvert 可执行文件（若不在 PATH 中可用环境变量指定）
MNN_CONVERT=${MNN_CONVERT:-MNNConvert}

"$PYTHON" tools/convert_model.py \
    --mnn_convert "$MNN_CONVERT" \
    "$@"
