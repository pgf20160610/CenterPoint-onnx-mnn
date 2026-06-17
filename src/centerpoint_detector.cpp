#include "centerpoint_detector.h"

#ifdef HAVE_ONNXRUNTIME
#include "onnx_engine.h"
#endif
#ifdef HAVE_MNN
#include "mnn_engine.h"
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

CenterPointDetector::CenterPointDetector(Config cfg) : cfg_(std::move(cfg)) {
    pillars_.assign(static_cast<size_t>(cfg_.max_pillars) * cfg_.max_points_per_pillar * cfg_.feature_num, 0.f);
    indices_.assign(cfg_.max_pillars, -1);
    pfe_out_.assign(static_cast<size_t>(cfg_.max_pillars) * cfg_.pfe_output_dim, 0.f);
    bev_feat_.assign(static_cast<size_t>(cfg_.pfe_output_dim) * cfg_.bev_h * cfg_.bev_w, 0.f);
}

std::unique_ptr<IRuntimeEngine> CenterPointDetector::CreateEngine() const {
#ifdef HAVE_ONNXRUNTIME
    if (cfg_.backend == "onnx" || cfg_.backend == "onnxruntime")
        return std::make_unique<OnnxEngine>();
#endif
#ifdef HAVE_MNN
    if (cfg_.backend == "mnn")
        return std::make_unique<MnnEngine>();
#endif
    std::cerr << "[detector] backend '" << cfg_.backend
              << "' not supported or not compiled in\n";
    return nullptr;
}

bool CenterPointDetector::Init() {
    pfe_engine_ = CreateEngine();
    rpn_engine_ = CreateEngine();
    if (!pfe_engine_ || !rpn_engine_) return false;

    if (!pfe_engine_->Init(cfg_.pfe_path, cfg_.use_cuda, cfg_.num_threads)) {
        std::cerr << "[detector] failed to load PFE: " << cfg_.pfe_path << "\n";
        return false;
    }
    if (!rpn_engine_->Init(cfg_.rpn_path, cfg_.use_cuda, cfg_.num_threads)) {
        std::cerr << "[detector] failed to load RPN: " << cfg_.rpn_path << "\n";
        return false;
    }
    std::cout << "[detector] backend=" << pfe_engine_->BackendName()
              << " PFE=" << cfg_.pfe_path << " RPN=" << cfg_.rpn_path << "\n";
    return true;
}

bool CenterPointDetector::ReadBin(const std::string& path,
                                  std::vector<float>& points, int& nPoints) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "[detector] cannot open " << path << "\n"; return false; }
    f.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    nPoints = static_cast<int>(bytes / sizeof(float) / cfg_.point_dim);
    points.resize(static_cast<size_t>(nPoints) * cfg_.point_dim);
    f.read(reinterpret_cast<char*>(points.data()), static_cast<std::streamsize>(bytes));
    return true;
}

// ----------------------------------------------------------------------------
// voxelise – build 10-dim pillar features on the CPU.
// Feature layout per point: [x, y, z, intensity, time_lag,
//                            x-clusterMean, y-clusterMean, z-clusterMean,
//                            x-pillarCentre, y-pillarCentre]
// ----------------------------------------------------------------------------
int CenterPointDetector::voxelise(const float* points, int nPoints) {
    const int   D   = cfg_.point_dim;
    const int   F   = cfg_.feature_num;
    const int   P   = cfg_.max_points_per_pillar;
    const int   W   = cfg_.bev_w, H = cfg_.bev_h;
    const float xmn = cfg_.x_min, ymn = cfg_.y_min;
    const float xs  = cfg_.voxel_size[0], ys = cfg_.voxel_size[1];

    std::fill(pillars_.begin(), pillars_.end(), 0.f);
    std::fill(indices_.begin(), indices_.end(), -1);

    std::vector<int>   bev2slot(static_cast<size_t>(W) * H, -1);
    std::vector<int>   pt_count(cfg_.max_pillars, 0);
    std::vector<float> sum_x(cfg_.max_pillars, 0.f),
                       sum_y(cfg_.max_pillars, 0.f),
                       sum_z(cfg_.max_pillars, 0.f);
    int nPillars = 0;

    // Pass 1: assign points to pillars, accumulate per-pillar sums.
    for (int i = 0; i < nPoints; ++i) {
        const float x = points[i * D + 0];
        const float y = points[i * D + 1];
        const float z = points[i * D + 2];
        if (x < cfg_.x_min || x >= cfg_.x_max ||
            y < cfg_.y_min || y >= cfg_.y_max ||
            z < cfg_.z_min || z >= cfg_.z_max) continue;

        const int xi = std::min(static_cast<int>((x - xmn) / xs), W - 1);
        const int yi = std::min(static_cast<int>((y - ymn) / ys), H - 1);
        const int flat = yi * W + xi;

        int slot = bev2slot[flat];
        if (slot == -1) {
            if (nPillars >= cfg_.max_pillars) continue;
            slot = nPillars++;
            bev2slot[flat] = slot;
            indices_[slot] = flat;
        }
        const int cnt = pt_count[slot];
        if (cnt >= P) continue;

        const size_t base = (static_cast<size_t>(slot) * P + cnt) * F;
        pillars_[base + 0] = x;
        pillars_[base + 1] = y;
        pillars_[base + 2] = z;
        pillars_[base + 3] = points[i * D + 3];                 // intensity
        pillars_[base + 4] = (D > 4) ? points[i * D + 4] : 0.f; // time_lag
        pillars_[base + 8] = x - (xi * xs + xmn + xs * 0.5f);   // x offset to pillar centre
        pillars_[base + 9] = y - (yi * ys + ymn + ys * 0.5f);   // y offset to pillar centre

        sum_x[slot] += x; sum_y[slot] += y; sum_z[slot] += z;
        pt_count[slot] = cnt + 1;
    }

    // Pass 2: cluster-mean offsets (features 5,6,7).
    for (int slot = 0; slot < nPillars; ++slot) {
        const int cnt = pt_count[slot];
        if (cnt == 0) continue;
        const float mx = sum_x[slot] / cnt;
        const float my = sum_y[slot] / cnt;
        const float mz = sum_z[slot] / cnt;
        for (int p = 0; p < cnt; ++p) {
            const size_t base = (static_cast<size_t>(slot) * P + p) * F;
            pillars_[base + 5] = pillars_[base + 0] - mx;
            pillars_[base + 6] = pillars_[base + 1] - my;
            pillars_[base + 7] = pillars_[base + 2] - mz;
        }
    }
    return nPillars;
}

// ----------------------------------------------------------------------------
// scatter – pillar embeddings (max_pillars, dim) → BEV grid (dim, H, W).
// ----------------------------------------------------------------------------
void CenterPointDetector::scatter(int nPillars) {
    const int W = cfg_.bev_w, H = cfg_.bev_h, C = cfg_.pfe_output_dim;
    std::fill(bev_feat_.begin(), bev_feat_.end(), 0.f);
    for (int i = 0; i < nPillars; ++i) {
        const int idx = indices_[i];
        if (idx < 0 || idx >= H * W) continue;
        const int y = idx / W, x = idx % W;
        for (int f = 0; f < C; ++f)
            bev_feat_[(static_cast<size_t>(f) * H + y) * W + x]
                = pfe_out_[static_cast<size_t>(i) * C + f];
    }
}

// ----------------------------------------------------------------------------
// Rotated BEV IoU helpers.
// ----------------------------------------------------------------------------
namespace {
struct Pt2 { float x, y; };

// Four corners of a box's BEV footprint (length l along heading, width w).
void boxCorners(const Box& b, Pt2 c[4]) {
    const float co = std::cos(b.theta), si = std::sin(b.theta);
    const float dx = b.l * 0.5f, dy = b.w * 0.5f;
    const float lx[4] = {-dx, -dx, dx, dx};
    const float ly[4] = {-dy,  dy, dy, -dy};
    for (int i = 0; i < 4; ++i) {
        c[i].x = b.x + lx[i] * co - ly[i] * si;
        c[i].y = b.y + lx[i] * si + ly[i] * co;
    }
}

float polyArea(const std::vector<Pt2>& p) {
    float a = 0.f;
    const int n = static_cast<int>(p.size());
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        a += p[i].x * p[j].y - p[j].x * p[i].y;
    }
    return std::fabs(a) * 0.5f;
}

// Clip convex polygon `subj` against convex quad `clip` (Sutherland-Hodgman).
std::vector<Pt2> clipPoly(std::vector<Pt2> subj, const Pt2 clip[4]) {
    // Orientation sign of the clip polygon so the inside test works either way.
    float sa = 0.f;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) % 4;
        sa += clip[i].x * clip[j].y - clip[j].x * clip[i].y;
    }
    const float sign = sa >= 0.f ? 1.f : -1.f;

    for (int e = 0; e < 4 && !subj.empty(); ++e) {
        const Pt2 A = clip[e], B = clip[(e + 1) % 4];
        const float ex = B.x - A.x, ey = B.y - A.y;
        std::vector<Pt2> out;
        const int n = static_cast<int>(subj.size());
        for (int i = 0; i < n; ++i) {
            const Pt2 P = subj[i], Q = subj[(i + 1) % n];
            const float dP = sign * (ex * (P.y - A.y) - ey * (P.x - A.x));
            const float dQ = sign * (ex * (Q.y - A.y) - ey * (Q.x - A.x));
            if (dP >= 0.f) out.push_back(P);
            if ((dP >= 0.f) != (dQ >= 0.f)) {
                const float t = dP / (dP - dQ);
                out.push_back({P.x + t * (Q.x - P.x), P.y + t * (Q.y - P.y)});
            }
        }
        subj = std::move(out);
    }
    return subj;
}

float rotatedBevIoU(const Box& a, const Box& b) {
    Pt2 ca[4], cb[4];
    boxCorners(a, ca);
    boxCorners(b, cb);
    const std::vector<Pt2> inter = clipPoly(std::vector<Pt2>(ca, ca + 4), cb);
    const float ai = inter.size() < 3 ? 0.f : polyArea(inter);
    if (ai <= 0.f) return 0.f;
    const float uni = a.l * a.w + b.l * b.w - ai;
    return uni > 1e-6f ? ai / uni : 0.f;
}
}  // namespace

// ----------------------------------------------------------------------------
// decodeAndNMS – decode RPN heads to boxes, then rotated BEV IoU NMS.
// ----------------------------------------------------------------------------
std::vector<Box> CenterPointDetector::decodeAndNMS(const RpnOutputs& raw) const {
    const int   H  = cfg_.output_h, W = cfg_.output_w;
    const int   HW = H * W;
    const float osf = cfg_.out_size_factor;

    std::vector<Box> boxes;
    boxes.reserve(2048);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            const float s = raw.score[idx];
            if (s < cfg_.score_threshold) continue;

            Box b;
            b.x     = (x + raw.reg[idx])      * osf * cfg_.voxel_size[0] + cfg_.x_min;
            b.y     = (y + raw.reg[HW + idx]) * osf * cfg_.voxel_size[1] + cfg_.y_min;
            b.z     = raw.height[idx];
            // dim head channel order is (width, length, height).
            b.w     = raw.dim[idx];
            b.l     = raw.dim[HW + idx];
            b.h     = raw.dim[2 * HW + idx];
            // rot channels are (cos, sin) for this model -> heading = atan2(ch1, ch0).
            b.theta = std::atan2(raw.rot[HW + idx], raw.rot[idx]);
            b.score = s;
            b.cls   = static_cast<int>(std::lround(raw.cls[idx]));
            boxes.push_back(b);
        }
    }

    // Rotated (oriented) BEV IoU NMS. Using axis-aligned IoU here lets rotated
    // duplicates of the same object survive, so we intersect the true oriented
    // footprints (Sutherland-Hodgman clipping + shoelace area).
    std::sort(boxes.begin(), boxes.end(),
              [](const Box& a, const Box& b) { return a.score > b.score; });
    if (static_cast<int>(boxes.size()) > cfg_.input_nms_max_size)
        boxes.resize(cfg_.input_nms_max_size);

    auto iou = [](const Box& a, const Box& b) { return rotatedBevIoU(a, b); };

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<Box> keep;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(boxes[i]);
        if (static_cast<int>(keep.size()) >= cfg_.output_nms_max_size) break;
        for (size_t j = i + 1; j < boxes.size(); ++j)
            if (!suppressed[j] && iou(boxes[i], boxes[j]) > cfg_.nms_threshold)
                suppressed[j] = true;
    }
    return keep;
}

// ----------------------------------------------------------------------------
// Infer – orchestrate the full pipeline for one frame.
// ----------------------------------------------------------------------------
std::vector<Box> CenterPointDetector::Infer(const std::vector<float>& points, int nPoints) {
    if (!pfe_engine_ || !rpn_engine_) return {};

    const int nPillars = voxelise(points.data(), nPoints);

    // --- PFE ---
    const std::vector<int64_t> pfe_shape{
        cfg_.max_pillars, cfg_.max_points_per_pillar, cfg_.feature_num};
    const auto pfe_in = pfe_engine_->InputNames();
    if (pfe_in.empty() ||
        !pfe_engine_->SetInput(pfe_in[0], pfe_shape, pillars_.data(), pillars_.size()) ||
        !pfe_engine_->Run())
        return {};
    std::vector<int64_t> pfe_out_shape;
    const auto pfe_out_names = pfe_engine_->OutputNames();
    if (pfe_out_names.empty() ||
        !pfe_engine_->GetOutput(pfe_out_names[0], pfe_out_, pfe_out_shape))
        return {};

    // --- scatter ---
    scatter(nPillars);

    // --- RPN ---
    const std::vector<int64_t> rpn_shape{1, cfg_.pfe_output_dim, cfg_.bev_h, cfg_.bev_w};
    const auto rpn_in = rpn_engine_->InputNames();
    if (rpn_in.empty() ||
        !rpn_engine_->SetInput(rpn_in[0], rpn_shape, bev_feat_.data(), bev_feat_.size()) ||
        !rpn_engine_->Run())
        return {};

    RpnOutputs raw;
    std::vector<int64_t> sh;
    const bool ok =
        rpn_engine_->GetOutput(cfg_.name_reg,    raw.reg,    sh) &&
        rpn_engine_->GetOutput(cfg_.name_height, raw.height, sh) &&
        rpn_engine_->GetOutput(cfg_.name_dim,    raw.dim,    sh) &&
        rpn_engine_->GetOutput(cfg_.name_rot,    raw.rot,    sh) &&
        rpn_engine_->GetOutput(cfg_.name_score,  raw.score,  sh) &&
        rpn_engine_->GetOutput(cfg_.name_cls,    raw.cls,    sh);
    if (!ok) {
        std::cerr << "[detector] failed to fetch all RPN outputs (check rpn_output_names)\n";
        return {};
    }

    return decodeAndNMS(raw);
}
