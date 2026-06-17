#include "onnx_engine.h"

#ifdef HAVE_ONNXRUNTIME
#include <algorithm>
#include <iostream>

namespace {
size_t numel(const std::vector<int64_t>& s) {
    size_t n = 1;
    for (auto d : s) n *= static_cast<size_t>(d);
    return n;
}
}  // namespace

OnnxEngine::OnnxEngine()
    : env_(ORT_LOGGING_LEVEL_WARNING, "centerpoint_onnx"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

bool OnnxEngine::Init(const std::string& model_path, bool use_cuda, int num_threads) {
    opts_.SetIntraOpNumThreads(num_threads);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            opts_.AppendExecutionProvider_CUDA(cuda_opts);
            std::cout << "[onnx] CUDA execution provider enabled\n";
        } catch (const Ort::Exception& e) {
            std::cout << "[onnx] CUDA EP unavailable (" << e.what() << "), using CPU\n";
        }
    }

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts_);
        cacheIO();
    } catch (const Ort::Exception& e) {
        std::cerr << "[onnx] failed to load " << model_path << ": " << e.what() << "\n";
        return false;
    }
    return true;
}

void OnnxEngine::cacheIO() {
    input_names_.clear();   output_names_.clear();
    input_names_c_.clear(); output_names_c_.clear();
    for (size_t i = 0; i < session_->GetInputCount(); ++i)
        input_names_.emplace_back(session_->GetInputNameAllocated(i, alloc_).get());
    for (size_t i = 0; i < session_->GetOutputCount(); ++i)
        output_names_.emplace_back(session_->GetOutputNameAllocated(i, alloc_).get());
    for (auto& n : input_names_)  input_names_c_.push_back(n.c_str());
    for (auto& n : output_names_) output_names_c_.push_back(n.c_str());
}

bool OnnxEngine::SetInput(const std::string& name, const std::vector<int64_t>& shape,
                          const float* data, size_t count) {
    if (std::find(input_names_.begin(), input_names_.end(), name) == input_names_.end())
        return false;
    if (numel(shape) != count) return false;
    input_cache_[name] = RuntimeTensor{name, shape, std::vector<float>(data, data + count)};
    return true;
}

bool OnnxEngine::Run() {
    std::vector<Ort::Value> inputs;
    inputs.reserve(input_names_.size());
    for (const auto& name : input_names_) {
        auto it = input_cache_.find(name);
        if (it == input_cache_.end()) {
            std::cerr << "[onnx] missing input: " << name << "\n";
            return false;
        }
        auto& t = it->second;
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            mem_info_, t.data.data(), t.data.size(), t.shape.data(), t.shape.size()));
    }

    try {
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     input_names_c_.data(), inputs.data(), inputs.size(),
                                     output_names_c_.data(), output_names_c_.size());
        output_cache_.clear();
        for (size_t i = 0; i < outputs.size(); ++i) {
            RuntimeTensor out;
            out.name = output_names_[i];
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            out.shape = info.GetShape();
            const size_t n = info.GetElementCount();
            out.data.resize(n);

            // Convert any integer head (e.g. the int64 class-id output "266") to float
            // so downstream code is dtype-agnostic.
            switch (info.GetElementType()) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                    const float* p = outputs[i].GetTensorData<float>();
                    std::copy(p, p + n, out.data.begin());
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                    const int64_t* p = outputs[i].GetTensorData<int64_t>();
                    for (size_t k = 0; k < n; ++k) out.data[k] = static_cast<float>(p[k]);
                    break;
                }
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                    const int32_t* p = outputs[i].GetTensorData<int32_t>();
                    for (size_t k = 0; k < n; ++k) out.data[k] = static_cast<float>(p[k]);
                    break;
                }
                default:
                    std::cerr << "[onnx] unsupported output dtype for " << out.name << "\n";
                    return false;
            }
            output_cache_[out.name] = std::move(out);
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "[onnx] run failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

bool OnnxEngine::GetOutput(const std::string& name, std::vector<float>& data,
                           std::vector<int64_t>& shape) {
    auto it = output_cache_.find(name);
    if (it == output_cache_.end()) return false;
    data  = it->second.data;
    shape = it->second.shape;
    return true;
}

#endif  // HAVE_ONNXRUNTIME
