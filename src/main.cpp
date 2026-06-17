/**
 * CenterPoint C++ inference demo.
 *
 *   centerpoint [--config config/config_cpp.yaml] [--backend onnx|mnn]
 *               [--data-dir DIR] [--save-dir DIR] [--no-viz]
 *
 * Loads the configuration, builds a CenterPointDetector for the chosen backend
 * (ONNX Runtime or MNN — both go through the shared pre/post-processing in
 * CenterPointDetector) and runs it over every .bin file in the data directory.
 * Each frame produces a `<stem>.txt` of boxes and, when OpenCV is available and
 * visualisation is enabled, a `<stem>_bev.png`.
 */

#include "centerpoint_config.h"
#include "centerpoint_detector.h"
#include "visualization.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::string argValue(int argc, char** argv, const std::string& key, const std::string& def = {}) {
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == key) return argv[i + 1];
    return def;
}
bool hasArg(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] == key) return true;
    return false;
}

void printHelp() {
    std::cout <<
        "Usage: centerpoint [options]\n"
        "  --config PATH     YAML config (default: config/config_cpp.yaml)\n"
        "  --backend NAME    override backend: onnx | mnn\n"
        "  --data-dir DIR    directory mode: run every .bin in DIR\n"
        "  --save-dir DIR    override output directory\n"
        "  --no-viz          disable visualisation output\n"
        "\n"
        " Single-sample mode (KITTI: point cloud + camera image + calib):\n"
        "  --input  BIN      one .bin point cloud; renders BEV + camera projection\n"
        "  --image  PNG      camera image for 3D-box projection\n"
        "  --calib  TXT      KITTI calibration (P2, R0_rect, Tr_velo_to_cam)\n"
        "  --label  TXT      KITTI ground-truth labels (optional, overlaid)\n"
        "  --no-gt           do not overlay ground-truth boxes\n"
        "  --help            show this message\n";
}

// Collect *.bin files in `dir`, sorted by name.
std::vector<std::string> collectBins(const std::string& dir) {
    std::vector<std::string> files;
    if (!fs::exists(dir)) {
        std::cerr << "[main] data dir does not exist: " << dir << "\n";
        return files;
    }
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".bin")
            files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    return files;
}

void saveTxt(const std::vector<Box>& boxes, const std::string& path) {
    std::ofstream f(path);
    for (const auto& b : boxes)
        f << b.x << " " << b.y << " " << b.z << " "
          << b.l << " " << b.h << " " << b.w << " "
          << 0.f << " " << 0.f << " "
          << b.theta << " " << b.score << " " << b.cls << "\n";
}
}  // namespace

int main(int argc, char** argv) {
    if (hasArg(argc, argv, "--help")) { printHelp(); return 0; }

    const std::string config_path = argValue(argc, argv, "--config", "config/config_cpp.yaml");
    Config cfg;
    try {
        cfg = Config::fromYaml(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[main] " << e.what() << " — using defaults\n";
    }

    // CLI overrides.
    const std::string backend = argValue(argc, argv, "--backend");
    if (!backend.empty()) {
        cfg.backend = backend;
        // Auto-switch model paths when the extension doesn't match the new backend.
        auto swapExt = [](std::string& p, const std::string& from, const std::string& to) {
            if (p.size() >= from.size() && p.compare(p.size() - from.size(), from.size(), from) == 0)
                p = p.substr(0, p.size() - from.size()) + to;
        };
        if (backend == "mnn") {
            swapExt(cfg.pfe_path, ".onnx", ".mnn");
            swapExt(cfg.rpn_path, ".onnx", ".mnn");
        } else if (backend == "onnx" || backend == "onnxruntime") {
            swapExt(cfg.pfe_path, ".mnn", ".onnx");
            swapExt(cfg.rpn_path, ".mnn", ".onnx");
        }
    }
    const std::string dd = argValue(argc, argv, "--data-dir");
    if (!dd.empty())       cfg.data_dir = dd;
    const std::string sd = argValue(argc, argv, "--save-dir");
    if (!sd.empty()) {
        cfg.save_dir = sd;
    } else if (!backend.empty()) {
        // No explicit --save-dir: segregate outputs by backend so MNN and ONNX
        // results land in separate sub-directories for easy side-by-side comparison.
        cfg.save_dir += "/" + backend;
    }
    if (hasArg(argc, argv, "--no-viz")) cfg.save_viz = false;
    if (hasArg(argc, argv, "--no-gt"))  cfg.vis_show_gt = false;

    // Visualisation source overrides (single-sample mode).
    const std::string input = argValue(argc, argv, "--input");
    const std::string img   = argValue(argc, argv, "--image", cfg.vis_image_path);
    const std::string calib = argValue(argc, argv, "--calib", cfg.vis_calib_path);
    const std::string label = argValue(argc, argv, "--label", cfg.vis_label_path);

    fs::create_directories(cfg.save_dir);

    CenterPointDetector detector(std::move(cfg));
    if (!detector.Init()) {
        std::cerr << "[main] detector init failed\n";
        return 1;
    }
    const Config& c = detector.cfg();
    using Clock = std::chrono::steady_clock;

    // ---- single-sample mode: one bin + camera image + calib ----
    if (!input.empty()) {
        std::vector<float> points;
        int nPoints = 0;
        if (!detector.ReadBin(input, points, nPoints)) {
            std::cerr << "[main] failed to read " << input << "\n";
            return 2;
        }
        const auto t0 = Clock::now();
        const auto boxes = detector.Infer(points, nPoints);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        const std::string stem = fs::path(input).stem().string();
        std::cout << "[" << c.backend << "] " << stem
                  << "  points=" << nPoints << "  boxes=" << boxes.size()
                  << "  " << ms << " ms\n";

        saveTxt(boxes, c.save_dir + "/" + stem + ".txt");
        if (c.save_viz) {
            SaveBev(c, points, nPoints, boxes,
                    c.save_dir + "/" + stem + "_bev.png",
                    calib, label, c.vis_show_gt);
            if (!img.empty() && !calib.empty())
                SaveImageProjection(c, boxes, img, calib, label, c.vis_show_gt,
                                    c.save_dir + "/" + stem + "_img3d.png");
            else
                std::cout << "[main] skip image projection (need --image and --calib)\n";
        }
        return 0;
    }

    // ---- directory mode: every .bin in data_dir ----
    const auto files = collectBins(c.data_dir);
    if (files.empty()) {
        std::cerr << "[main] no .bin files found in " << c.data_dir << "\n";
        return 2;
    }
    std::cout << "[main] " << files.size() << " frame(s) in " << c.data_dir << "\n";

    double total_ms = 0.0;
    for (const auto& bin : files) {
        std::vector<float> points;
        int nPoints = 0;
        if (!detector.ReadBin(bin, points, nPoints)) continue;

        const auto t0 = Clock::now();
        const auto boxes = detector.Infer(points, nPoints);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        total_ms += ms;

        const std::string stem = fs::path(bin).stem().string();
        std::cout << "[" << c.backend << "] " << stem
                  << "  points=" << nPoints
                  << "  boxes=" << boxes.size()
                  << "  " << ms << " ms\n";

        saveTxt(boxes, c.save_dir + "/" + stem + ".txt");
        if (c.save_viz)
            SaveBev(c, points, nPoints, boxes, c.save_dir + "/" + stem + "_bev.png");
    }

    std::cout << "\n[main] done — " << files.size() << " frame(s), avg "
              << (total_ms / files.size()) << " ms/frame (infer only)\n";
    return 0;
}
