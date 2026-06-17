#!/usr/bin/env bash
# CenterPoint ONNX Runtime 推理
# 用法: bash run_onnx.sh [传给 centerpoint 的额外参数]
#   例: bash run_onnx.sh --data-dir lidars --save-dir output/cpp --no-viz
set -e

# 运行时库路径（可用环境变量覆盖）
ORT_ROOT=${ORT_ROOT:-/home/panguofeng/pgf_ai_deploy/PointPillars_ONNX_MNN_CPP/ThirdParty/onnxruntime-linux-x64-1.16.3}
CONDA_PREFIX=${CONDA_PREFIX:-/home/panguofeng/anaconda3/envs/mnn_toolkit}
export LD_LIBRARY_PATH="$ORT_ROOT/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

./build/centerpoint --config config/config_cpp.yaml --backend onnx "$@"
