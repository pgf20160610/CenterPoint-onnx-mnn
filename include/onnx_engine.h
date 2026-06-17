#pragma once
/**
 * ONNX Runtime implementation of IRuntimeEngine.
 *
 * Compiled only when HAVE_ONNXRUNTIME is defined. Tries the CUDA execution
 * provider when requested, falling back to CPU. Outputs of any element type
 * (float / int64 / int32) are converted to float in the output cache, so the
 * detector handles the int64 class head uniformly.
 */

#include "runtime_engine.h"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <unordered_map>

class OnnxEngine final : public IRuntimeEngine {
public:
    OnnxEngine();
    bool Init(const std::string& model_path, bool use_cuda, int num_threads) override;
    std::vector<std::string> InputNames()  const override { return input_names_; }
    std::vector<std::string> OutputNames() const override { return output_names_; }
    bool SetInput(const std::string& name, const std::vector<int64_t>& shape,
                  const float* data, size_t count) override;
    bool Run() override;
    bool GetOutput(const std::string& name, std::vector<float>& data,
                   std::vector<int64_t>& shape) override;
    std::string BackendName() const override { return "onnx"; }

private:
    void cacheIO();

    Ort::Env                       env_;
    Ort::SessionOptions            opts_;
    Ort::MemoryInfo                mem_info_;
    Ort::AllocatorWithDefaultOptions alloc_;
    std::unique_ptr<Ort::Session>  session_;

    std::vector<std::string>  input_names_;
    std::vector<std::string>  output_names_;
    std::vector<const char*>  input_names_c_;
    std::vector<const char*>  output_names_c_;

    std::unordered_map<std::string, RuntimeTensor> input_cache_;
    std::unordered_map<std::string, RuntimeTensor> output_cache_;
};

#endif  // HAVE_ONNXRUNTIME
