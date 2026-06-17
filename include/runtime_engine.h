#pragma once
/**
 * IRuntimeEngine – backend-agnostic neural-network runtime.
 *
 * A pure abstract base: the ONNX Runtime and MNN backends derive from it and
 * implement the virtual methods. Everything above the engine (voxelisation,
 * scatter, box decode, NMS, visualisation in CenterPointDetector) is written
 * once against this interface, so the pre/post-processing is shared verbatim
 * across backends — only the tensor-execution layer differs.
 *
 * The engine deals only in flat float tensors keyed by name. Backends that
 * produce non-float outputs (e.g. the int64 class-id head in ONNX) convert to
 * float internally so callers always see float data.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A named tensor with shape + row-major float payload.
struct RuntimeTensor {
    std::string          name;
    std::vector<int64_t> shape;
    std::vector<float>   data;
};

class IRuntimeEngine {
public:
    virtual ~IRuntimeEngine() = default;

    //! Load a model file. `use_cuda` is a hint (ORT CUDA EP); `num_threads`
    //! controls intra-op parallelism. Returns false on failure.
    virtual bool Init(const std::string& model_path, bool use_cuda, int num_threads) = 0;

    //! Names of the model inputs / outputs, in model order.
    virtual std::vector<std::string> InputNames()  const = 0;
    virtual std::vector<std::string> OutputNames() const = 0;

    //! Stage an input tensor for the next Run(). `count` must equal the product
    //! of `shape`. Returns false if the name is unknown or the size mismatches.
    virtual bool SetInput(const std::string& name,
                          const std::vector<int64_t>& shape,
                          const float* data, size_t count) = 0;

    //! Execute the network on the staged inputs.
    virtual bool Run() = 0;

    //! Fetch an output by name as float data + shape. Returns false if absent.
    virtual bool GetOutput(const std::string& name,
                           std::vector<float>& data,
                           std::vector<int64_t>& shape) = 0;

    //! Human-readable backend tag ("onnx" | "mnn").
    virtual std::string BackendName() const = 0;
};
