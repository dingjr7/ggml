#pragma once
#include "../common.cuh"
#include <cmath>
#include <cstring>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cusolverDn.h>
#include <map>
#include <vector>

namespace ggml_l4dgs_cuda {
struct Plan;
struct State {
    cudaStream_t stream;
    cublasLtHandle_t blas{};
    cusolverDnHandle_t solver{};
    std::map<std::vector<int64_t>, std::shared_ptr<Plan>> plans;
    explicit State(cudaStream_t s) : stream(s) {}
    ~State() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        plans.clear();
        if (solver)
            cusolverDnDestroy(solver);
        if (blas)
            cublasLtDestroy(blas);
    }
    cublasLtHandle_t lt() {
        if (!blas)
            CUBLAS_CHECK(cublasLtCreate(&blas));
        return blas;
    }
    cusolverDnHandle_t svd() {
        if (!solver) {
            GGML_ASSERT(cusolverDnCreate(&solver) == CUSOLVER_STATUS_SUCCESS);
        }
        GGML_ASSERT(cusolverDnSetStream(solver, stream) == CUSOLVER_STATUS_SUCCESS);
        return solver;
    }
};
inline int grid(int64_t n) {
    GGML_ASSERT(n >= 0 && (n + 255) / 256 <= INT_MAX);
    return int((n + 255) / 256);
}
__device__ inline float load(const void* p, ggml_type type, int64_t i) {
    return type == GGML_TYPE_F32 ? static_cast<const float*>(p)[i]
                                 : __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}
__device__ inline void save(void* p, ggml_type type, int64_t i, float x) {
    if (type == GGML_TYPE_F32)
        static_cast<float*>(p)[i] = x;
    else
        static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16(x);
}
void norm(ggml_backend_cuda_context&, ggml_tensor*);
void prefill(ggml_backend_cuda_context&, State&, ggml_tensor*);
} // namespace ggml_l4dgs_cuda
struct ggml_cuda_l4dgs_state {
    std::map<cudaStream_t, std::unique_ptr<ggml_l4dgs_cuda::State>> streams;
    ggml_l4dgs_cuda::State& get(cudaStream_t s) {
        auto& state = streams[s];
        if (!state)
            state = std::make_unique<ggml_l4dgs_cuda::State>(s);
        return *state;
    }
};
