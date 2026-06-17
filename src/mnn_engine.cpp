#include "mnn_engine.h"

#ifdef HAVE_MNN
#include <algorithm>
#include <iostream>

MnnEngine::~MnnEngine() {
    if (net_ && session_) net_->releaseSession(session_);
}

bool MnnEngine::Init(const std::string& model_path, bool use_cuda, int num_threads) {
    net_.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (!net_) {
        std::cerr << "[mnn] failed to load " << model_path << "\n";
        return false;
    }
    MNN::ScheduleConfig cfg;
    cfg.numThread = num_threads;
    cfg.type = use_cuda ? MNN_FORWARD_CUDA : MNN_FORWARD_CPU;
    session_ = net_->createSession(cfg);
    if (!session_) {
        std::cerr << "[mnn] failed to create session for " << model_path << "\n";
        return false;
    }

    input_names_.clear();
    output_names_.clear();
    for (const auto& kv : net_->getSessionInputAll(session_))  input_names_.push_back(kv.first);
    for (const auto& kv : net_->getSessionOutputAll(session_)) output_names_.push_back(kv.first);
    return true;
}

bool MnnEngine::SetInput(const std::string& name, const std::vector<int64_t>& shape,
                         const float* data, size_t count) {
    if (!net_->getSessionInput(session_, name.c_str())) return false;
    size_t expected = 1;
    for (int64_t d : shape) expected *= static_cast<size_t>(d);
    if (expected != count) return false;
    input_cache_[name] = RuntimeTensor{name, shape, std::vector<float>(data, data + count)};
    return true;
}

bool MnnEngine::Run() {
    if (!net_ || !session_) return false;

    // Resize first so dynamic dimensions match the staged input shapes.
    for (const auto& kv : input_cache_) {
        auto* in = net_->getSessionInput(session_, kv.first.c_str());
        if (!in) return false;
        std::vector<int> shape;
        shape.reserve(kv.second.shape.size());
        for (int64_t d : kv.second.shape) shape.push_back(static_cast<int>(d));
        net_->resizeTensor(in, shape);
    }
    net_->resizeSession(session_);

    for (const auto& kv : input_cache_) {
        auto* in = net_->getSessionInput(session_, kv.first.c_str());
        if (!in) return false;
        MNN::Tensor host(in, MNN::Tensor::CAFFE);
        if (static_cast<size_t>(host.elementSize()) != kv.second.data.size()) {
            std::cerr << "[mnn] input size mismatch for " << kv.first << "\n";
            return false;
        }
        std::copy(kv.second.data.begin(), kv.second.data.end(), host.host<float>());
        in->copyFromHostTensor(&host);
    }
    return net_->runSession(session_) == MNN::NO_ERROR;
}

bool MnnEngine::GetOutput(const std::string& name, std::vector<float>& data,
                          std::vector<int64_t>& shape) {
    auto* out = net_->getSessionOutput(session_, name.c_str());
    if (!out) return false;
    MNN::Tensor host(out, MNN::Tensor::CAFFE);
    out->copyToHostTensor(&host);
    shape.clear();
    for (int i = 0; i < host.dimensions(); ++i) shape.push_back(host.length(i));
    data.assign(host.host<float>(), host.host<float>() + host.elementSize());
    return true;
}

#endif  // HAVE_MNN
