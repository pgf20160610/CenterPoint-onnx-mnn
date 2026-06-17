#include "centerpoint_config.h"

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>

// Helper: read node[key] into dst if present, else leave default.
template <typename T>
static void get(const YAML::Node& node, const char* key, T& dst) {
    if (node && node[key]) {
        dst = node[key].as<T>();
    }
}

Config Config::fromYaml(const std::string& path) {
    Config c;

    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load config '" + path + "': " + e.what());
    }

    if (auto pc = root["point_cloud"]) {
        get(pc, "x_min", c.x_min);   get(pc, "x_max", c.x_max);
        get(pc, "y_min", c.y_min);   get(pc, "y_max", c.y_max);
        get(pc, "z_min", c.z_min);   get(pc, "z_max", c.z_max);
        // Accept either voxel_size: [vx, vy, vz] or the legacy x_step / y_step keys.
        if (pc["voxel_size"]) {
            auto vs = pc["voxel_size"].as<std::vector<float>>();
            if (vs.size() >= 1) c.voxel_size[0] = vs[0];
            if (vs.size() >= 2) c.voxel_size[1] = vs[1];
            if (vs.size() >= 3) c.voxel_size[2] = vs[2];
        } else {
            float xs = c.voxel_size[0], ys = c.voxel_size[1];
            get(pc, "x_step", xs); get(pc, "y_step", ys);
            c.voxel_size[0] = xs;  c.voxel_size[1] = ys;
        }
        get(pc, "point_dim", c.point_dim);
    }

    if (auto p = root["pillar"]) {
        get(p, "max_pillars", c.max_pillars);
        get(p, "max_points_per_pillar", c.max_points_per_pillar);
        get(p, "feature_num", c.feature_num);
        get(p, "pfe_output_dim", c.pfe_output_dim);
    }

    if (auto b = root["bev"]) {
        get(b, "width", c.bev_w);
        get(b, "height", c.bev_h);
    }

    if (auto pp = root["postprocess"]) {
        get(pp, "out_size_factor", c.out_size_factor);
        get(pp, "score_threshold", c.score_threshold);
        get(pp, "nms_threshold", c.nms_threshold);
        get(pp, "input_nms_max_size", c.input_nms_max_size);
        get(pp, "output_nms_max_size", c.output_nms_max_size);
        get(pp, "output_h", c.output_h);
        get(pp, "output_w", c.output_w);
    }

    if (auto n = root["rpn_output_names"]) {
        get(n, "reg", c.name_reg);
        get(n, "height", c.name_height);
        get(n, "dim", c.name_dim);
        get(n, "rot", c.name_rot);
        get(n, "score", c.name_score);
        get(n, "cls", c.name_cls);
    }

    if (auto m = root["model"]) {
        get(m, "backend", c.backend);
        get(m, "pfe_path", c.pfe_path);
        get(m, "rpn_path", c.rpn_path);
    }

    if (auto r = root["runtime"]) {
        get(r, "use_cuda", c.use_cuda);
        get(r, "num_threads", c.num_threads);
        get(r, "save_viz", c.save_viz);
    }

    if (auto io = root["io"]) {
        get(io, "data_dir", c.data_dir);
        get(io, "save_dir", c.save_dir);
    }

    if (auto v = root["visualization"]) {
        get(v, "image_path", c.vis_image_path);
        get(v, "calib_path", c.vis_calib_path);
        get(v, "label_path", c.vis_label_path);
        get(v, "show_gt",    c.vis_show_gt);
        get(v, "bev_view",   c.vis_bev_view);
        get(v, "bev_width",  c.vis_bev_width);
        get(v, "bev_height", c.vis_bev_height);
        get(v, "reference_style", c.vis_reference_style);
        if (v["class_names"]) c.vis_class_names = v["class_names"].as<std::vector<std::string>>();
        if (v["x_range"]) {
            auto r = v["x_range"].as<std::vector<float>>();
            if (r.size() == 2) c.vis_x_range = {r[0], r[1]};
        }
        if (v["y_range"]) {
            auto r = v["y_range"].as<std::vector<float>>();
            if (r.size() == 2) c.vis_y_range = {r[0], r[1]};
        }
    }

    return c;
}
