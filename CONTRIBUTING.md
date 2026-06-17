# 贡献指南

本文档记录 ONNX/MNN 部署重构引入的架构与改动，以及如何编译、运行和扩展本项目。

## 概述

CenterPoint 激光雷达 3D 检测通过两个可互换的推理后端部署——**ONNX Runtime** 与
**MNN**——它们共用同一套预处理 / 后处理流程。不依赖 TensorRT/CUDA。所有可调参数
都放在 YAML 配置文件中，C++ 程序和 Python 工具都从中读取。

单帧流程：

```
读取 .bin → 体素化(voxelise) → PFE 引擎 → 散射(scatter) → RPN 引擎 → 解码 + 旋转 NMS → TXT + BEV PNG
```

## 架构

各后端只在张量执行层不同，其余全部共用。

| 层 | 文件 | 职责 |
|-----|------|------|
| 引擎接口 | [include/runtime_engine.h](include/runtime_engine.h) | `IRuntimeEngine` 抽象基类：`Init / SetInput / Run / GetOutput / InputNames / OutputNames`。引擎只处理以名称为键的扁平 float 张量。 |
| ONNX 后端 | [include/onnx_engine.h](include/onnx_engine.h)、[src/onnx_engine.cpp](src/onnx_engine.cpp) | ONNX Runtime 实现。优先尝试 CUDA EP，失败回退 CPU。将整型输出（如 int64 类别头 `266`）转换为 float，使调用方与数据类型无关。 |
| MNN 后端 | [include/mnn_engine.h](include/mnn_engine.h)、[src/mnn_engine.cpp](src/mnn_engine.cpp) | MNN 实现。在 `Run()` 时按需 resize session 以支持动态 shape；张量按 float（CAFFE/NCHW）读回。 |
| 检测器 | [include/centerpoint_detector.h](include/centerpoint_detector.h)、[src/centerpoint_detector.cpp](src/centerpoint_detector.cpp) | 持有 PFE + RPN 两个引擎，并拥有**全部**共享逻辑：体素化、散射、框解码、旋转 BEV NMS。后端由 `cfg.backend` 经 `CreateEngine()` 选择。 |
| 配置 | [include/centerpoint_config.h](include/centerpoint_config.h)、[src/centerpoint_config.cpp](src/centerpoint_config.cpp) | 从 YAML 加载的 `Config` 结构体（yaml-cpp）。 |
| 可视化 | [include/visualization.h](include/visualization.h)、[src/visualization.cpp](src/visualization.cpp) | OpenCV BEV 渲染 + 相机 3D 框投影（KITTI 标定）。由 `HAVE_OPENCV` 守卫；无 OpenCV 时退化为空操作（仅输出 TXT）。KITTI 标定/标签解析封装在 .cpp 内，头文件不暴露 OpenCV 类型。 |
| 入口 | [src/main.cpp](src/main.cpp) | 配置驱动；支持命令行覆盖；两种模式——目录模式（枚举 `.bin`，输出 BEV）与单样本模式（点云 + 相机图 + 标定，输出 BEV + 相机投影）。 |

**新增后端的方式：** 实现 `IRuntimeEngine`，在 `CenterPointDetector::CreateEngine()`
中加分支，并在 CMake 里用 `HAVE_*` 宏接入。不允许重复任何预处理 / 后处理代码。

## 配置

参数不再以宏的形式硬编码，统一放在 YAML 中：

- [config/config_cpp.yaml](config/config_cpp.yaml) —— C++ 程序目录模式（`model.backend: onnx|mnn`、模型路径、运行时、IO）。
- [config/config_python.yaml](config/config_python.yaml) —— Python 工具（ONNX provider 列表等）。
- [config/config_sample.yaml](config/config_sample.yaml) —— 单样本模式（KITTI 样例，`point_dim: 4`，含 `visualization` 节）。

配置分节：`point_cloud`、`pillar`、`bev`、`postprocess`、`rpn_output_names`、
`model`、`runtime`、`io`、`visualization`（相机投影所需的 `image_path` /
`calib_path` / `label_path` / `show_gt`）。C++ 与 Python 两份配置需保持一致，
以保证结果对齐。

## 编译（C++）

依赖：C++17 编译器、**yaml-cpp**、**ONNX Runtime** 和/或 **MNN**，
可选 **OpenCV**（用于可视化）。

```bash
export ORT_ROOT=/path/to/onnxruntime          # ONNX Runtime 安装根目录
cmake -B build \
  -DUSE_ONNXRUNTIME=ON -DUSE_MNN=OFF -DUSE_OPENCV=ON \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \          # 使 find_package(yaml-cpp) 能定位
  -DONNXRUNTIME_ROOT=$ORT_ROOT
cmake --build build -j4
```

CMake 选项：`USE_ONNXRUNTIME`（默认 ON）、`USE_MNN`（默认 OFF）、
`USE_OPENCV`（默认 ON）。至少需启用一个后端。启用 MNN：
`-DUSE_MNN=ON -DMNN_ROOT=/path/to/MNN`。

`yaml-cpp` 可通过 `conda install -c conda-forge yaml-cpp` 或
`apt install libyaml-cpp-dev` 安装。

## 运行

C++（需让动态链接器找到运行时库）：

```bash
LD_LIBRARY_PATH=$ORT_ROOT/lib:$CONDA_PREFIX/lib \
  ./build/centerpoint --config config/config_cpp.yaml --backend onnx
# 可覆盖项：--data-dir DIR  --save-dir DIR  --no-viz
```

Python（matplotlib BEV 可视化）：

```bash
python tools/infer_onnx.py --config config/config_python.yaml
# 可覆盖项：--data_dir --save_dir --score_thresh --nms_thresh --providers --no_viz
```

每帧输出 `<save_dir>/<stem>.txt`（每行一个框：
`x y z l h w 0 0 theta score cls`）和 `<stem>_bev.png`。

## 单样本模式（点云 + 相机 + 标定）

参考 PointPillars_ONNX_MNN_CPP 的可视化流程，读取单帧 KITTI 格式数据并同时输出 BEV
与相机图像上的 3D 框投影。由 `--input` 触发：

```bash
bash run_sample.sh
# 等价命令：
./build/centerpoint --config config/config_sample.yaml --backend onnx \
  --input data/sample.bin --image data/sample.png \
  --calib data/sample_calib.txt --label data/sample_label.txt
# 可选：--no-gt 关闭真值叠加
```

输出到 `output/sample/`：`sample.txt`、`sample_bev.png`（预测框 + 青色 GT）、
`sample_img3d.png`（相机 3D 框投影）。

实现要点：

- **相机投影**（[src/visualization.cpp](src/visualization.cpp) 的 `SaveImageProjection`）：
  预测框在雷达系，先取 8 个角点，经 `R0_rect · Tr_velo_to_cam` 变换到相机系、再用
  `P2` 投影到像素，`z>0.1` 判定可见，画 12 条棱的线框立方体；GT 标签本就在相机系，
  直接经 `P2` 投影。
- **BEV 中的 GT**（`SaveBev`）：KITTI 标签在相机系，用 `(R0·Tr)^-1` 转回雷达系再投到 BEV。
- **几何约定**：预测框 z 为框中心高度，角点 z 取 `z ± h/2`；朝向用 `theta` 绕 z 轴旋转。
- KITTI 标定/标签解析（`readCalib` / `readLabels`）封装在 .cpp 内，头文件保持无 OpenCV 依赖。
- **数据 / 模型域差异**：`data/` 为 KITTI（每点 4 维，`point_dim: 4`），而 `models/` 为
  Waymo 模型，跨域检测精度有限；该模式主要用于打通“点云 + 相机”读取、推理与可视化全流程。

## 模型转换

- [tools/setup_det3d.sh](tools/setup_det3d.sh) —— 配置 det3d 环境（安装 PyTorch + 最小依赖，编译 dcn/iou3d_nms 扩展）。依赖清单见 [tools/requirements_det3d.txt](tools/requirements_det3d.txt)。仅模型导出需要，纯推理不依赖。
- [tools/export_onnx.py](tools/export_onnx.py) —— PyTorch → ONNX（PFE + RPN，基于 det3d）。
- [tools/convert_model.py](tools/convert_model.py) —— 下载权重、导出 ONNX、校验，并将 ONNX → MNN（`MNNConvert` CLI 或 MNN Python SDK）。读取 `--runtime_config`，使转换出的模型落在推理流程预期的路径上。

## 实现要点 / 注意事项

- **必须使用旋转 NMS。** 框是带朝向的；轴对齐 IoU 会低估重叠，导致同一目标的旋转
  重复框无法被抑制。NMS 在 C++（[src/centerpoint_detector.cpp](src/centerpoint_detector.cpp)
  中的 `rotatedBevIoU`）和 Python（[tools/infer_onnx.py](tools/infer_onnx.py) 中的
  `_iou_bev_rotated`）里都使用旋转 BEV IoU（Sutherland-Hodgman 凸多边形裁剪 +
  鞋带公式求面积）。两端实现需保持等价。
- **dim 头通道顺序为 (width, length, height)。** RPN 的 `dim` 输出三个通道依次是
  宽、长、高，解码时必须按此赋值（`b.w=dim[0], b.l=dim[1], b.h=dim[2]`）。早期误按
  (l, h, w) 赋值，导致框尺寸错乱（高约 4.6m、底面约 2×1.6），可视化里立方体异常竖高。
  C++ 与 Python 解码两端都需保持该顺序。
- **rot 头通道顺序为 (cos, sin)。** 朝向解码为 `theta = atan2(rot[1], rot[0])`
  （即通道 0=cos、通道 1=sin），与常见的 (sin, cos) 相反。若按 `atan2(rot[0],rot[1])`
  解码，BEV 里框会相对真值偏转约 90°。可用一帧带 KITTI 真值的样本核对：把预测朝向与
  GT 朝向逐车比较，正确时两者之差应在数度以内。C++ 与 Python 两端保持一致。
- **int64 类别头。** RPN 的 `cls` 输出（`266`）在 ONNX 中为 int64；ONNX 引擎在输出
  缓存中将其（及 int32）转为 float，使检测器与数据类型无关。MNN 已直接返回 float。
- **ONNX 图优化。** 某些 ONNX Runtime 版本会对 PFE 图施加过激的 Gemm 融合，在
  `(P, N, C)` 输入上形状推断失败。Python 会话设置 `ORT_DISABLE_ALL` 以规避。
- **输出数量上限。** 最终框数受 `postprocess.output_nms_max_size` 限制。若大量低分框
  撞到上限，这是 `score_threshold` 调参问题，而非 NMS 失效——调高 `score_threshold`
  即可得到更干净的结果。

## 改动验证

C++ 与 Python 共用流程，结果应数值一致。改动后请在 `lidars/` 的样例帧上运行两个后端，
确认 TXT 输出一致，且没有任何残留框对的旋转 IoU 超过 NMS 阈值。

## 本次重构的改动

- 移除全部 TensorRT 推理 / 转换代码、CUDA kernel，以及 PCL/Boost 依赖。
- 在共享的 `IRuntimeEngine` 接口下新增 ONNX Runtime 与 MNN 后端，由
  `CenterPointDetector` 持有公共预处理 / 后处理。
- 将所有参数迁移到 `config/config_cpp.yaml` / `config/config_python.yaml`。
- 新增 BEV 可视化（C++ 用 OpenCV，Python 用 matplotlib）。
- 修复 NMS：用旋转（朝向）BEV IoU 取代轴对齐 IoU。
- 新增单样本模式与相机 3D 框投影可视化（KITTI 标定），配套
  `config/config_sample.yaml` 与 `run_sample.sh`。
