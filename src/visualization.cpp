#include "visualization.h"

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace {
constexpr float kPi = 3.14159265358979323846f;

// 12 edges of a cuboid (corners 0-3 bottom, 4-7 top).
constexpr std::array<std::pair<int, int>, 12> kEdges{{
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
}};

const cv::Scalar kGtColor{0, 176, 176};  // ground truth – yellow (BGR)

// Class colour keyed by name (matches the PointPillars reference scheme).
cv::Scalar classColor(int cls, const std::vector<std::string>& names) {
    static const std::map<std::string, cv::Scalar> named{
        {"Pedestrian", {0, 0, 255}}, {"Cyclist", {0, 192, 0}},
        {"Car", {255, 0, 0}}, {"Vehicle", {255, 0, 0}}};
    static const std::array<cv::Scalar, 5> fallback{{
        {255, 0, 0}, {0, 0, 255}, {0, 192, 0}, {255, 0, 255}, {255, 255, 0}}};
    if (cls >= 0 && cls < static_cast<int>(names.size())) {
        auto it = named.find(names[cls]);
        if (it != named.end()) return it->second;
    }
    return fallback[static_cast<size_t>(std::max(0, cls)) % fallback.size()];
}

std::string className(int cls, const std::vector<std::string>& names) {
    if (cls >= 0 && cls < static_cast<int>(names.size())) return names[cls];
    return "cls=" + std::to_string(cls);
}

// ---- KITTI calibration ----
struct Calib {
    cv::Matx34f p2{};
    cv::Matx44f tr = cv::Matx44f::eye();
    cv::Matx44f r0 = cv::Matx44f::eye();
    cv::Matx44f cam_to_lidar = cv::Matx44f::eye();
    bool valid = false;
};

bool readCalib(const std::string& path, Calib& calib) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    std::map<std::string, std::vector<float>> vals;
    std::string line;
    while (std::getline(ifs, line)) {
        const auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::istringstream iss(line.substr(pos + 1));
        float v;
        while (iss >> v) vals[line.substr(0, pos)].push_back(v);
    }
    if (vals["P2"].size() != 12 || vals["Tr_velo_to_cam"].size() != 12 ||
        vals["R0_rect"].size() != 9)
        return false;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) calib.p2(r, c) = vals["P2"][r * 4 + c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) calib.tr(r, c) = vals["Tr_velo_to_cam"][r * 4 + c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) calib.r0(r, c) = vals["R0_rect"][r * 3 + c];
    calib.cam_to_lidar = (calib.r0 * calib.tr).inv();
    calib.valid = true;
    return true;
}

// ---- KITTI ground-truth labels (camera frame) ----
struct GtLabel {
    std::string type;
    float h = 0, w = 0, l = 0, x = 0, y = 0, z = 0, ry = 0;
};

std::vector<GtLabel> readLabels(const std::string& path) {
    std::ifstream ifs(path);
    std::vector<GtLabel> labels;
    if (!ifs.is_open()) return labels;
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        GtLabel l;
        float fbuf;
        int ibuf;
        float bbox[4];
        if (!(iss >> l.type) || l.type == "DontCare") continue;
        if (iss >> fbuf >> ibuf >> fbuf >> bbox[0] >> bbox[1] >> bbox[2] >> bbox[3] >>
            l.h >> l.w >> l.l >> l.x >> l.y >> l.z >> l.ry)
            labels.push_back(l);
    }
    return labels;
}

struct Vec3 { float x, y, z; };

// 8 corners of a predicted box in the LiDAR frame (z = box centre height).
std::array<Vec3, 8> boxCornersLidar(const Box& b) {
    const float hl = b.l * 0.5f, hw = b.w * 0.5f, hh = b.h * 0.5f;
    const float c = std::cos(b.theta), s = std::sin(b.theta);
    const std::array<Vec3, 8> local{{
        {-hl, -hw, -hh}, {-hl, hw, -hh}, {hl, hw, -hh}, {hl, -hw, -hh},
        {-hl, -hw,  hh}, {-hl, hw,  hh}, {hl, hw,  hh}, {hl, -hw,  hh},
    }};
    std::array<Vec3, 8> out{};
    for (size_t i = 0; i < 8; ++i) {
        out[i].x = local[i].x * c - local[i].y * s + b.x;
        out[i].y = local[i].x * s + local[i].y * c + b.y;
        out[i].z = local[i].z + b.z;
    }
    return out;
}

// 8 corners of a GT label in the camera frame (y down, bottom at y).
std::array<Vec3, 8> labelCornersCamera(const GtLabel& l) {
    const float c = std::cos(l.ry), s = std::sin(l.ry);
    const std::array<Vec3, 8> local{{
        {l.l / 2, 0, l.w / 2}, {l.l / 2, 0, -l.w / 2},
        {-l.l / 2, 0, -l.w / 2}, {-l.l / 2, 0, l.w / 2},
        {l.l / 2, -l.h, l.w / 2}, {l.l / 2, -l.h, -l.w / 2},
        {-l.l / 2, -l.h, -l.w / 2}, {-l.l / 2, -l.h, l.w / 2},
    }};
    std::array<Vec3, 8> out{};
    for (size_t i = 0; i < 8; ++i) {
        out[i].x = local[i].x * c + local[i].z * s + l.x;
        out[i].y = local[i].y + l.y;
        out[i].z = -local[i].x * s + local[i].z * c + l.z;
    }
    return out;
}

Vec3 transform(const cv::Matx44f& m, const Vec3& p) {
    const cv::Vec4f o = m * cv::Vec4f(p.x, p.y, p.z, 1.0f);
    return {o[0], o[1], o[2]};
}

// ---- BEV projection (configurable view) ----
cv::Point bevProject(float x, float y, int width, int height, const Config& cfg) {
    const float x0 = cfg.vis_x_range[0], x1 = cfg.vis_x_range[1];
    const float y0 = cfg.vis_y_range[0], y1 = cfg.vis_y_range[1];
    float u = 0.f, v = 0.f;
    if (cfg.vis_bev_view == "forward-down") {
        u = width - 1 - (y - y0) / (y1 - y0) * (width - 1);
        v = (x - x0) / (x1 - x0) * (height - 1);
    } else if (cfg.vis_bev_view == "left-up") {
        u = (x - x0) / (x1 - x0) * (width - 1);
        v = height - 1 - (y - y0) / (y1 - y0) * (height - 1);
    } else if (cfg.vis_bev_view == "right-up") {
        u = width - 1 - (x - x0) / (x1 - x0) * (width - 1);
        v = height - 1 - (y - y0) / (y1 - y0) * (height - 1);
    } else {  // forward-up (default)
        u = width - 1 - (y - y0) / (y1 - y0) * (width - 1);
        v = height - 1 - (x - x0) / (x1 - x0) * (height - 1);
    }
    return {static_cast<int>(std::lround(u)), static_cast<int>(std::lround(v))};
}

void drawSensorAxes(cv::Mat& canvas, int w, int h, const std::string& view) {
    const cv::Point origin = view.rfind("forward", 0) == 0
        ? cv::Point(w / 2, h - 45) : cv::Point(60, h / 2);
    cv::circle(canvas, origin, 7, {80, 80, 80}, cv::FILLED, cv::LINE_AA);
    cv::arrowedLine(canvas, origin, {origin.x, std::max(15, origin.y - 90)},
                    {0, 0, 255}, 5, cv::LINE_AA, 0, 0.25);
    const int y_tip_x = view.rfind("forward", 0) == 0 ? std::max(15, origin.x - 90) : origin.x;
    cv::arrowedLine(canvas, origin, {y_tip_x, origin.y}, {0, 255, 0}, 5, cv::LINE_AA, 0, 0.25);
    cv::putText(canvas, "x/front", {origin.x + 8, std::max(20, origin.y - 95)},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 180}, 1, cv::LINE_AA);
    cv::putText(canvas, "y/left", {std::max(5, origin.x - 115), origin.y - 12},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 150, 0}, 1, cv::LINE_AA);
}

void drawLegend(cv::Mat& img, int legend_x, const std::vector<std::string>& names) {
    cv::rectangle(img, {legend_x, 0}, {img.cols - 1, img.rows - 1}, {0, 0, 0}, cv::FILLED);
    int y = std::max(120, img.rows / 3);
    for (const auto& n : names) {
        cv::putText(img, n + ":", {legend_x + 60, y},
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, classColor(0, {n}), 3, cv::LINE_AA);
        y += 70;
    }
    cv::putText(img, "Ground truth:", {legend_x + 60, y},
                cv::FONT_HERSHEY_SIMPLEX, 1.2, kGtColor, 3, cv::LINE_AA);
}

// ---- camera projection ----
void projectLidar(const std::array<Vec3, 8>& corners, const Calib& calib,
                  std::array<cv::Point2f, 8>& pts, std::array<bool, 8>& valid) {
    for (size_t i = 0; i < 8; ++i) {
        const cv::Vec4f cam = calib.r0 * calib.tr *
                              cv::Vec4f(corners[i].x, corners[i].y, corners[i].z, 1.0f);
        valid[i] = cam[2] > 0.1f;
        const cv::Vec3f proj = calib.p2 * cam;
        const float z = std::max(proj[2], 1e-6f);
        pts[i] = {proj[0] / z, proj[1] / z};
    }
}

void projectCamera(const std::array<Vec3, 8>& corners, const Calib& calib,
                   std::array<cv::Point2f, 8>& pts, std::array<bool, 8>& valid) {
    for (size_t i = 0; i < 8; ++i) {
        valid[i] = corners[i].z > 0.1f;
        const cv::Vec3f proj =
            calib.p2 * cv::Vec4f(corners[i].x, corners[i].y, corners[i].z, 1.0f);
        const float z = std::max(proj[2], 1e-6f);
        pts[i] = {proj[0] / z, proj[1] / z};
    }
}

void drawProjectedBox(cv::Mat& img, const std::array<cv::Point2f, 8>& pts,
                      const std::array<bool, 8>& valid, const cv::Scalar& color,
                      const std::string& label) {
    const cv::Rect frame(0, 0, img.cols, img.rows);
    bool drawn = false;
    for (auto [a, b] : kEdges) {
        if (!valid[a] || !valid[b]) continue;
        cv::Point p0(std::lround(pts[a].x), std::lround(pts[a].y));
        cv::Point p1(std::lround(pts[b].x), std::lround(pts[b].y));
        if (cv::clipLine(frame, p0, p1)) {
            cv::line(img, p0, p1, color, 2, cv::LINE_AA);
            drawn = true;
        }
    }
    if (drawn && !label.empty()) {
        for (size_t i = 0; i < 8; ++i) {
            if (valid[i] && pts[i].x >= 0 && pts[i].x < img.cols &&
                pts[i].y >= 0 && pts[i].y < img.rows) {
                cv::putText(img, label,
                            {static_cast<int>(std::lround(pts[i].x)),
                             std::max(15, static_cast<int>(std::lround(pts[i].y)) - 4)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
                break;
            }
        }
    }
}
}  // namespace

// ============================================================================
bool SaveBev(const Config& cfg,
             const std::vector<float>& points, int nPoints,
             const std::vector<Box>& boxes,
             const std::string& path,
             const std::string& calib_path,
             const std::string& label_path,
             bool show_gt) {
    const int   D    = cfg.point_dim;
    const bool  ref  = cfg.vis_reference_style;
    const int   legend = ref ? 520 : 0;
    const int   W    = cfg.vis_bev_width - legend;   // canvas width
    const int   H    = cfg.vis_bev_height;
    if (W <= 0 || H <= 0) return false;

    const cv::Scalar bg = ref ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);
    cv::Mat img(H, cfg.vis_bev_width, CV_8UC3, bg);
    cv::Mat canvas = img(cv::Rect(0, 0, W, H));

    auto inRange = [&](float x, float y) {
        return x >= cfg.vis_x_range[0] && x <= cfg.vis_x_range[1] &&
               y >= cfg.vis_y_range[0] && y <= cfg.vis_y_range[1];
    };

    // Point cloud.
    for (int i = 0; i < nPoints; ++i) {
        const float x = points[i * D + 0], y = points[i * D + 1];
        const float inten = (D > 3) ? points[i * D + 3] : 0.f;
        if (!inRange(x, y)) continue;
        const cv::Point uv = bevProject(x, y, W, H, cfg);
        if (uv.x < 0 || uv.x >= W || uv.y < 0 || uv.y >= H) continue;
        // White bg: dark points (low KITTI intensity must still read clearly).
        const int g = ref ? std::clamp(static_cast<int>(110.f - inten * 150.f), 0, 130)
                          : std::clamp(static_cast<int>(inten * 255.f), 60, 255);
        canvas.at<cv::Vec3b>(uv.y, uv.x) = cv::Vec3b(g, g, g);
    }

    // Predicted boxes (bottom 4 corners).
    for (const auto& b : boxes) {
        const auto c = boxCornersLidar(b);
        std::vector<cv::Point> poly;
        for (int i = 0; i < 4; ++i) poly.push_back(bevProject(c[i].x, c[i].y, W, H, cfg));
        const cv::Scalar col = classColor(b.cls, cfg.vis_class_names);
        cv::polylines(canvas, poly, true, col, 2, cv::LINE_AA);
        char buf[48];
        std::snprintf(buf, sizeof(buf), "Pred %s %.2f",
                      className(b.cls, cfg.vis_class_names).c_str(), b.score);
        cv::putText(canvas, buf, poly.front(), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
    }

    // GT boxes (camera-frame labels → LiDAR via calib).
    if (show_gt && !calib_path.empty() && !label_path.empty()) {
        Calib calib;
        if (readCalib(calib_path, calib)) {
            for (const auto& l : readLabels(label_path)) {
                const auto cam = labelCornersCamera(l);
                std::vector<cv::Point> poly;
                for (int i = 0; i < 4; ++i) {
                    const Vec3 lp = transform(calib.cam_to_lidar, cam[i]);
                    poly.push_back(bevProject(lp.x, lp.y, W, H, cfg));
                }
                cv::polylines(canvas, poly, true, kGtColor, 2, cv::LINE_AA);
                cv::putText(canvas, "GT " + l.type, poly.front(),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, kGtColor, 1, cv::LINE_AA);
            }
        }
    }

    if (ref) {
        drawSensorAxes(canvas, W, H, cfg.vis_bev_view);
        drawLegend(img, W, cfg.vis_class_names);
    } else {
        cv::circle(canvas, cv::Point(W / 2, H / 2), 4, cv::Scalar(0, 255, 255), -1);
    }

    if (!cv::imwrite(path, img)) {
        std::cerr << "[viz] failed to write " << path << "\n";
        return false;
    }
    std::cout << "  [viz] BEV  : " << path << "\n";
    return true;
}

bool SaveImageProjection(const Config& cfg,
                         const std::vector<Box>& boxes,
                         const std::string& image_path,
                         const std::string& calib_path,
                         const std::string& label_path,
                         bool show_gt,
                         const std::string& path) {
    Calib calib;
    if (!readCalib(calib_path, calib)) {
        std::cerr << "[viz] failed to read calib: " << calib_path << "\n";
        return false;
    }
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[viz] failed to read image: " << image_path << "\n";
        return false;
    }

    for (const auto& b : boxes) {
        std::array<cv::Point2f, 8> pts;
        std::array<bool, 8> valid;
        projectLidar(boxCornersLidar(b), calib, pts, valid);
        char buf[48];
        std::snprintf(buf, sizeof(buf), "Pred %s %.2f",
                      className(b.cls, cfg.vis_class_names).c_str(), b.score);
        drawProjectedBox(img, pts, valid, classColor(b.cls, cfg.vis_class_names), buf);
    }

    if (show_gt && !label_path.empty()) {
        for (const auto& l : readLabels(label_path)) {
            std::array<cv::Point2f, 8> pts;
            std::array<bool, 8> valid;
            projectCamera(labelCornersCamera(l), calib, pts, valid);
            drawProjectedBox(img, pts, valid, kGtColor, "GT " + l.type);
        }
    }

    if (!cv::imwrite(path, img)) {
        std::cerr << "[viz] failed to write " << path << "\n";
        return false;
    }
    std::cout << "  [viz] image: " << path << "\n";
    return true;
}

#else  // HAVE_OPENCV

bool SaveBev(const Config&, const std::vector<float>&, int,
             const std::vector<Box>&, const std::string&,
             const std::string&, const std::string&, bool) {
    return false;
}
bool SaveImageProjection(const Config&, const std::vector<Box>&, const std::string&,
                         const std::string&, const std::string&, bool, const std::string&) {
    return false;
}

#endif  // HAVE_OPENCV
