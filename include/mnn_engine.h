#pragma once
/**
 * MNN implementation of IRuntimeEngine.
 *
 * Compiled only when HAVE_MNN is defined. Inputs are cached and the session is
 * resized lazily on Run() so dynamic shapes are honoured. MNN tensors are
 * always read back as float (CAFFE / NCHW layout), matching the ONNX backend.
 */

#include "runtime_engine.h"

#ifdef HAVE_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>
#include <unordered_map>

class MnnEngine final : public IRuntimeEngine {
public:
    ~MnnEngine() override;
    bool Init(const std::string& model_path, bool use_cuda, int num_threads) override;
    std::vector<std::string> InputNames()  const override { return input_names_; }
    std::vector<std::string> OutputNames() const override { return output_names_; }
    bool SetInput(const std::string& name, const std::vector<int64_t>& shape,
                  const float* data, size_t count) override;
    bool Run() override;
    bool GetOutput(const std::string& name, std::vector<float>& data,
                   std::vector<int64_t>& shape) override;
    std::string BackendName() const override { return "mnn"; }

private:
    std::shared_ptr<MNN::Interpreter> net_;
    MNN::Session* session_ = nullptr;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::unordered_map<std::string, RuntimeTensor> input_cache_;
};

#endif  // HAVE_MNN
