#!/usr/bin/env bash
# ============================================================
# 配置本地 det3d 环境（用于 PyTorch -> ONNX 模型导出）
#
# det3d 仅在导出模型（tools/export_onnx.py）时需要；纯 ONNX/MNN 推理不依赖它。
# 本脚本完成三件事：
#   1.（可选）创建并激活 conda 环境
#   2. 安装 PyTorch 及最小 Python 依赖
#   3. 编译 det3d 的 CUDA 扩展（dcn、iou3d_nms）
#
# 用法:
#   bash tools/setup_det3d.sh                      # 在当前环境安装并编译
#   bash tools/setup_det3d.sh --create-env det3d   # 先创建名为 det3d 的 conda 环境
#   CUDA=cu113 bash tools/setup_det3d.sh           # 指定 torch 的 CUDA 版本
#
# 完成后将 tools/ 加入 PYTHONPATH 即可 import det3d：
#   export PYTHONPATH=$(pwd)/tools:$PYTHONPATH
# ============================================================
set -euo pipefail

# 脚本所在目录（tools/）与工程根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ---- 可配置项（可用环境变量覆盖）----
PY_VER=${PY_VER:-3.8}                 # conda 环境的 Python 版本
CUDA=${CUDA:-cu111}                   # torch 的 CUDA 后缀：cu111 | cu113 | cpu
TORCH_VER=${TORCH_VER:-1.8.0}
TORCHVISION_VER=${TORCHVISION_VER:-0.9.0}

CREATE_ENV=""
if [[ "${1:-}" == "--create-env" ]]; then
    CREATE_ENV="${2:-det3d}"
fi

# ---- 1. 可选：创建并激活 conda 环境 ----
if [[ -n "$CREATE_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[setup] 未找到 conda，请先安装 Anaconda/Miniconda 或去掉 --create-env"; exit 1
    fi
    echo "[setup] 创建 conda 环境: $CREATE_ENV (python=$PY_VER)"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda create -y -n "$CREATE_ENV" "python=$PY_VER"
    conda activate "$CREATE_ENV"
fi

PYTHON=${PYTHON:-python3}
echo "[setup] 使用 Python: $($PYTHON -c 'import sys; print(sys.executable)')"

# det3d 是 2021 年代的旧代码，需要 torch 1.x；而 torch 1.x 只提供到 cp39 的 wheel。
# 若当前 Python >= 3.10，无法安装 torch 1.x —— 提示创建独立的旧版环境。
PY_MM=$("$PYTHON" -c 'import sys; print(sys.version_info.major*100 + sys.version_info.minor)')
if [[ "$TORCH_VER" == 1.* && "$PY_MM" -ge 310 ]]; then
    echo "[setup] 错误：当前 Python 为 $($PYTHON -c 'import platform;print(platform.python_version())')，"
    echo "        det3d 需要 torch $TORCH_VER（仅提供 Python 3.6–3.9 的 wheel），无法在此 Python 上安装。"
    echo
    echo "  方案 A（推荐）：创建独立的 Python 3.8 环境后重试"
    echo "      bash tools/setup_det3d.sh --create-env det3d"
    echo
    echo "  方案 B：自行指定较新的 torch 组合（det3d 旧代码可能不兼容，需自测）"
    echo "      TORCH_VER=2.0.0 TORCHVISION_VER=0.15.1 CUDA=cu118 bash tools/setup_det3d.sh"
    exit 1
fi

# ---- 2. 安装 PyTorch 及最小依赖 ----
if [[ "$CUDA" == "cpu" ]]; then
    echo "[setup] 安装 CPU 版 torch $TORCH_VER"
    "$PYTHON" -m pip install "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER"
else
    echo "[setup] 安装 torch $TORCH_VER ($CUDA)"
    "$PYTHON" -m pip install \
        "torch==${TORCH_VER}+${CUDA}" "torchvision==${TORCHVISION_VER}+${CUDA}" \
        -f https://download.pytorch.org/whl/torch_stable.html
fi

echo "[setup] 安装最小 Python 依赖"
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements_det3d.txt"

# ---- 3. 编译 det3d 的 CUDA 扩展 ----
build_op() {
    local op_dir="$1"
    if [[ -f "$op_dir/setup.py" ]]; then
        echo "[setup] 编译扩展: $op_dir"
        ( cd "$op_dir" && "$PYTHON" setup.py build_ext --inplace )
    else
        echo "[setup] 跳过（无 setup.py）: $op_dir"
    fi
}
build_op "$SCRIPT_DIR/det3d/ops/dcn"
build_op "$SCRIPT_DIR/det3d/ops/iou3d_nms"

# ---- 完成提示 ----
cat <<EOF

[setup] 完成。验证：
  export PYTHONPATH=$SCRIPT_DIR:\$PYTHONPATH
  $PYTHON -c "import torch, det3d; print('torch', torch.__version__, 'det3d OK')"

之后即可导出模型：
  cd $ROOT_DIR
  bash convert_model.sh --ckpt latest.pth --to_onnx --to_mnn
EOF
