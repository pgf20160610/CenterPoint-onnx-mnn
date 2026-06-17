#pragma once
/**
 * Visualisation helpers (BEV + camera 3D-box projection).
 *
 * Mirrors the PointPillars_ONNX_MNN_CPP pipeline: detections are rendered both
 * as a bird's-eye-view image and projected onto the KITTI camera image using
 * the calibration (P2 · R0_rect · Tr_velo_to_cam). Ground-truth labels can be
 * overlaid for comparison.
 *
 * All functions are no-ops returning false when built without OpenCV
 * (HAVE_OPENCV undefined), so callers degrade gracefully to TXT-only output.
 * KITTI calib/label parsing is kept inside the .cpp so this header stays free
 * of OpenCV types.
 */

#include "centerpoint_config.h"
#include "centerpoint_detector.h"

#include <string>
#include <vector>

//! Render a BEV image of `points` overlaid with predicted `boxes` (and, when
//! `calib_path`/`label_path` are given and `show_gt` is true, KITTI GT boxes).
bool SaveBev(const Config& cfg,
             const std::vector<float>& points, int nPoints,
             const std::vector<Box>& boxes,
             const std::string& path,
             const std::string& calib_path = "",
             const std::string& label_path = "",
             bool show_gt = false);

//! Project predicted `boxes` (LiDAR frame) — and optionally KITTI GT boxes —
//! onto the camera image at `image_path` using `calib_path`, writing to `path`.
bool SaveImageProjection(const Config& cfg,
                         const std::vector<Box>& boxes,
                         const std::string& image_path,
                         const std::string& calib_path,
                         const std::string& label_path,
                         bool show_gt,
                         const std::string& path);
