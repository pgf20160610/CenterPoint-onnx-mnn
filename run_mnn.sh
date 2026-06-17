#!/usr/bin/env bash
# CenterPoint MNN 推理
# 用法: bash run_mnn.sh [传给 centerpoint 的额外参数]
#   例: bash run_mnn.sh --data-dir lidars --save-dir output/cpp --no-viz
# 前提: 编译时启用 MNN (-DUSE_MNN=ON)，且 config/config_cpp.yaml 中模型路径为 .mnn
set -e

# 运行时库路径（可用环境变量覆盖）
MNN_ROOT=${MNN_ROOT:-/home/panguofeng/project/MNN}
CONDA_PREFIX=${CONDA_PREFIX:-/home/panguofeng/anaconda3/envs/mnn_toolkit}
export LD_LIBRARY_PATH="$MNN_ROOT/lib:$MNN_ROOT/build:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

./build_mnn/centerpoint --config config/config_cpp.yaml --backend mnn "$@"
