"""
CenterPoint PointPillars model config for KITTI.

Key differences from Waymo config:
  - Front-facing range: x [0, 69.12], y [-39.68, 39.68], z [-3, 1]
  - Voxel size: 0.16 x 0.16 x 4.0  →  BEV 432 x 496
  - 4-dim input features (x, y, z, intensity; no time_lag)
  - Classes: Car, Pedestrian, Cyclist
  - max_pillars: 12000, max_points_per_pillar: 100
"""

import itertools
import logging
from det3d.utils.config_tool import get_downsample_factor

EXPORT_ONNX = True

# Key params
max_pillars = 12000
max_points_in_voxel = 100
x_min, x_max = 0.0, 69.12
y_min, y_max = -39.68, 39.68
z_min, z_max = -3.0, 1.0
x_step = 0.16
y_step = 0.16

# BEV: W=432, H=496
bev_w = int(round((x_max - x_min) / x_step))   # 432
bev_h = int(round((y_max - y_min) / y_step))   # 496

feature_num = 9       # 4-dim input + 3 cluster offsets + 2 pillar-center offsets
pfe_output_dim = 64

tasks = [
    dict(num_class=3, class_names=["Car", "Pedestrian", "Cyclist"]),
]
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

target_assigner = dict(tasks=tasks)

model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 64],
        num_input_features=4,       # x, y, z, intensity
        with_distance=False,
        voxel_size=(x_step, y_step, z_max - z_min),
        pc_range=(x_min, y_min, z_min, x_max, y_max, z_max),
        export_onnx=EXPORT_ONNX,
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[1, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[1, 2, 4],
        us_num_filters=[128, 128, 128],
        num_input_features=pfe_output_dim,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=128 * 3,
        tasks=tasks,
        dataset="kitti",
        weight=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={
            "reg": (2, 2),
            "height": (1, 2),
            "dim": (3, 2),
            "rot": (2, 2),
        },
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[x_min, y_min, z_min - 5, x_max, y_max, z_max + 5],
    nms=dict(
        nms_pre_max_size=500,
        nms_post_max_size=83,
        nms_iou_threshold=0.5,
    ),
    score_threshold=0.3,
    pc_range=[x_min, y_min],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[x_step, y_step],
)
