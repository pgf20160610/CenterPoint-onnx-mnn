#pragma once
/**
 * CenterPoint runtime configuration.
 *
 * All tunable parameters live in a YAML file (config/config_cpp.yaml) and are
 * loaded into this struct at startup. No values are hard-coded as macros.
 */

#include <array>
#include <string>
#include <vector>

struct Config {
    // ---- point cloud range & grid ----
    float x_min = -74.88f, x_max = 74.88f;
    float y_min = -74.88f, y_max = 74.88f;
    float z_min = -2.0f,   z_max = 4.0f;
    // voxel_size: [vx, vy, vz] — XY equals the BEV grid step; VZ spans the full Z range (pillar height)
    std::array<float, 3> voxel_size = {0.32f, 0.32f, 6.0f};
    int   point_dim = 5;

    // ---- pillar encoder ----
    int max_pillars = 32000;
    int max_points_per_pillar = 20;
    int feature_num = 10;
    int pfe_output_dim = 64;

    // ---- BEV grid ----
    int bev_w = 468;
    int bev_h = 468;

    // ---- postprocess ----
    float out_size_factor = 1.0f;
    float score_threshold = 0.1f;
    float nms_threshold = 0.7f;
    int   input_nms_max_size = 4096;
    int   output_nms_max_size = 500;
    int   output_h = 468;
    int   output_w = 468;

    // ---- RPN output tensor names (role -> tensor name) ----
    std::string name_reg = "246";
    std::string name_height = "250";
    std::string name_dim = "264";
    std::string name_rot = "258";
    std::string name_score = "265";
    std::string name_cls = "266";

    // ---- model ----
    std::string backend = "onnx";   // "onnx" | "mnn"
    std::string pfe_path;
    std::string rpn_path;

    // ---- runtime ----
    bool use_cuda = true;
    int  num_threads = 4;
    bool save_viz = true;

    // ---- io ----
    std::string data_dir;
    std::string save_dir = "output";

    // ---- visualization (single-sample camera projection) ----
    std::string vis_image_path;   // KITTI camera image (.png)
    std::string vis_calib_path;   // KITTI calibration (P2, R0_rect, Tr_velo_to_cam)
    std::string vis_label_path;   // KITTI ground-truth labels (optional)
    bool        vis_show_gt = true;  // overlay GT boxes when a label file is given

    // ---- BEV style (matches PointPillars_ONNX_MNN_CPP reference) ----
    std::vector<std::string> vis_class_names{"Vehicle", "Pedestrian", "Cyclist"};
    std::string vis_bev_view = "forward-up";       // forward-up|forward-down|left-up|right-up
    int   vis_bev_width  = 1786;
    int   vis_bev_height = 1122;
    bool  vis_reference_style = true;              // white bg + legend + sensor axes
    std::array<float, 2> vis_x_range{0.0f, 70.4f}; // BEV render range (m), independent of model grid
    std::array<float, 2> vis_y_range{-40.0f, 40.0f};

    //! Load configuration from a YAML file. Missing keys keep their defaults.
    static Config fromYaml(const std::string& path);
};
