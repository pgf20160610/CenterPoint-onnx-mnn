

/********************************************************** 
 * creater      : PGF
 * since        : 2026-06-16 18:49:43
 * lastTime     : 2026-06-16 19:01:21
 * LastAuthor   : PGF
 * message      : The function of this file is to define the CenterPointDetector class for handling the CenterPoint detection pipeline.
 * 文件相对于项目的路径   : /CenterPoint/include/centerpoint_detector.h
 * Copyright (c) 2026 by pgf email: nchu_pgf@163.com, All Rights Reserved.
 **********************************************************/

#pragma once
/**
 * CenterPointDetector – the backend-agnostic CenterPoint pipeline.
 *
 *   read bin → voxelise → PFE engine → scatter → RPN engine → decode + NMS
 *
 * All pre/post-processing lives here and is shared by every backend. The only
 * backend-specific code is the IRuntimeEngine implementations (ONNX / MNN),
 * created via CreateEngine() according to cfg_.backend. This realises the
 * requirement that the ONNX and MNN engines share preprocessing and
 * postprocessing through virtual-function inheritance.
 */

#include "centerpoint_config.h"
#include "runtime_engine.h"

#include <memory>
#include <string>
#include <vector>

// Decoded 3D bounding box (LiDAR frame).
struct Box {
    float x, y, z;
    float l, h, w;
    float theta;
    float score;
    int   cls;
};

class CenterPointDetector {
public:
    explicit CenterPointDetector(Config cfg);

    //! Create and load the PFE + RPN engines for cfg_.backend.
    bool Init();
    

    //! Read a point-cloud .bin (point_dim floats per point) into `points`.
    bool ReadBin(const std::string& path, std::vector<float>& points, int& nPoints) const;

    //! Full pipeline for one frame. Returns decoded + NMS-filtered boxes.
    std::vector<Box> Infer(const std::vector<float>& points, int nPoints);

    const Config& cfg() const { return cfg_; }

private:
    // Collected RPN head buffers (float, NCHW with batch=1).
    struct RpnOutputs {
        std::vector<float> reg, height, dim, rot, score, cls;
    };

    std::unique_ptr<IRuntimeEngine> CreateEngine() const;

    // ---- pipeline stages (shared) ----
    int  voxelise(const float* points, int nPoints);
    void scatter(int nPillars);
    std::vector<Box> decodeAndNMS(const RpnOutputs& raw) const;

    Config cfg_;
    std::unique_ptr<IRuntimeEngine> pfe_engine_;
    std::unique_ptr<IRuntimeEngine> rpn_engine_;

    // Working buffers (host), sized in the constructor.
    std::vector<float> pillars_;   // (max_pillars * pts * feat)
    std::vector<int>   indices_;   // (max_pillars) flat BEV index per pillar
    std::vector<float> pfe_out_;   // (max_pillars * pfe_output_dim)
    std::vector<float> bev_feat_;  // (pfe_output_dim * bev_h * bev_w)
};
