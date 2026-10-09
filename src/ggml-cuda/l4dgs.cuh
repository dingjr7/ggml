#pragma once
#include "common.cuh"
struct ggml_cuda_l4dgs_state;
bool ggml_cuda_l4dgs_supports_op(const ggml_tensor*);
void ggml_cuda_l4dgs_compute(ggml_backend_cuda_context&, ggml_tensor*);
inline bool ggml_cuda_l4dgs_op(ggml_op op) {
    return op >= GGML_OP_L4DGS_LINEAR && op <= GGML_OP_L4DGS_PREFILL_PAGED;
}
inline bool ggml_cuda_l4dgs_host_input(const ggml_tensor* op, int i) {
    return (op->op == GGML_OP_L4DGS_PREFILL_RAGGED || op->op == GGML_OP_L4DGS_PREFILL_PAGED) &&
           i >= 3;
}
