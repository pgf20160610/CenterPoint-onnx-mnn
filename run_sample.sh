#!/usr/bin/env bash
# CenterPoint 单样本推理 + 可视化（KITTI 格式：点云 + 相机图像 + 标定）
# 读取 data/ 中的 sample.bin / sample.png / sample_calib.txt / sample_label.txt，
# 推理后输出：
#   output/sample/sample.txt        检测框
#   output/sample/sample_bev.png    BEV 鸟瞰图（含预测框与 GT）
#   output/sample/sample_img3d.png  相机图像上的 3D 目标框投影
# 用法: bash run_sample.sh [传给 centerpoint 的额外参数]
set -e

ORT_ROOT=${ORT_ROOT:-/home/panguofeng/pgf_ai_deploy/PointPillars_ONNX_MNN_CPP/ThirdParty/onnxruntime-linux-x64-1.16.3}
CONDA_PREFIX=${CONDA_PREFIX:-/home/panguofeng/anaconda3/envs/mnn_toolkit}
export LD_LIBRARY_PATH="$ORT_ROOT/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

./build_mnn/centerpoint \
    --config config/config_sample.yaml \
    --backend mnn \
    --input data/sample.bin \
    --image data/sample.png \
    --calib data/sample_calib.txt \
    --label data/sample_label.txt \
    "$@"
