#include "support.cuh"
namespace ggml_l4dgs_cuda {
// Welford statistics with four contiguous values per lane and four warps per row.
// Keep this reduction order stable: BF16 casts after normalization amplify changes
// at rounding boundaries. Match the reference's float arithmetic and FMA policy.
struct Moments {
    float mean = 0, m2 = 0, count = 0;
};
__device__ Moments combine(Moments b, Moments a) {
    float delta = b.mean - a.mean, count = a.count + b.count;
    if (count == 0)
        return {};
    float reciprocal = 1.f / count, na = a.count * reciprocal, nb = b.count * reciprocal;
    return {na * a.mean + nb * b.mean, a.m2 + b.m2 + delta * delta * a.count * nb, count};
}
__global__ void norm_kernel(const void* x, ggml_type xt, const void* gamma, ggml_type gt,
                            const void* beta, ggml_type bt, void* y, int channels, float eps) {
    __shared__ Moments shared[4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, row = blockIdx.x;
    Moments m;
    for (int first = threadIdx.x * 4; first < channels; first += 128 * 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (first + j < channels) {
                float value = load(x, xt, int64_t(row) * channels + first + j),
                      delta = value - m.mean;
                float count = m.count + 1.f, mean = m.mean + delta * (1.f / count);
                m = {mean, m.m2 + delta * (value - mean), count};
            }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        Moments other{__shfl_down_sync(0xffffffff, m.mean, offset),
                      __shfl_down_sync(0xffffffff, m.m2, offset),
                      __shfl_down_sync(0xffffffff, m.count, offset)};
        m = combine(m, other);
    }
    for (int offset = 2; offset > 0; offset /= 2) {
        if (lane == 0 && warp >= offset && warp < 2 * offset)
            shared[warp - offset] = m;
        __syncthreads();
        if (lane == 0 && warp < offset)
            m = combine(m, shared[warp]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        shared[0] = m;
    __syncthreads();
    float mean = shared[0].mean, inv = rsqrtf(shared[0].m2 / float(channels) + eps);
    for (int c = threadIdx.x; c < channels; c += 128) {
        int64_t i = int64_t(row) * channels + c;
        float w = gamma ? load(gamma, gt, c) : 1.f, b = beta ? load(beta, bt, c) : 0.f;
        save(y, xt, i, w * (inv * (load(x, xt, i) - mean)) + b);
    }
}

void norm(ggml_backend_cuda_context& ctx, ggml_tensor* y) {
    auto* x = y->src[0];
    auto* w = y->src[1];
    auto* b = y->src[2];
    float eps;
    memcpy(&eps, y->op_params, sizeof(eps));
    norm_kernel<<<ggml_nelements(x) / x->ne[0], 128, 0, ctx.stream()>>>(
        x->data, x->type, w ? w->data : nullptr, w ? w->type : GGML_TYPE_F32, b ? b->data : nullptr,
        b ? b->type : GGML_TYPE_F32, y->data, x->ne[0], eps);
}
} // namespace ggml_l4dgs_cuda
