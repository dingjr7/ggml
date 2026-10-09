#include "../l4dgs.cuh"
#include "support.cuh"
namespace ggml_l4dgs_cuda {
#include "kernels.cuh"

struct Converted {
    ggml_cuda_pool_alloc<char> scratch;
    const void* data;
    Converted(ggml_backend_cuda_context& ctx, const ggml_tensor* x, ggml_type t)
        : scratch(ctx.pool()), data(x ? x->data : nullptr) {
        if (x && x->type != t && ggml_nelements(x)) {
            data = scratch.alloc(ggml_nelements(x) * ggml_type_size(t));
            cast_kernel<<<grid(ggml_nelements(x)), 256, 0, ctx.stream()>>>(
                x->data, x->type, const_cast<void*>(data), t, ggml_nelements(x));
        }
    }
};
static void linear(ggml_backend_cuda_context& ctx, State& state, ggml_tensor* y) {
    auto* x = y->src[0];
    auto* w = y->src[1];
    Converted a(ctx, w, y->type), b(ctx, x, y->type), bias(ctx, y->src[2], y->type);
    int64_t in = x->ne[0], out = w->ne[1], rows = ggml_nelements(x) / in;
    cudaDataType_t dt = y->type == GGML_TYPE_F32 ? CUDA_R_32F : CUDA_R_16BF;
    struct Descriptors {
        cublasLtMatmulDesc_t op{};
        cublasLtMatrixLayout_t a{}, b{}, c{};
        cublasLtMatmulPreference_t pref{};
        ~Descriptors() {
            if (op)
                cublasLtMatmulDescDestroy(op);
            if (a)
                cublasLtMatrixLayoutDestroy(a);
            if (b)
                cublasLtMatrixLayoutDestroy(b);
            if (c)
                cublasLtMatrixLayoutDestroy(c);
            if (pref)
                cublasLtMatmulPreferenceDestroy(pref);
        }
    } d;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_T;
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    if (bias.data) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                                    sizeof(epilogue)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias.data, sizeof(void*)));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d.a, dt, in, out, in));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d.b, dt, in, rows, in));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d.c, dt, out, rows, out));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&d.pref));
    size_t bytes = 1024 * 1024;
    ggml_cuda_pool_alloc<char> work(ctx.pool(), bytes);
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        d.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
    auto alignment = [](const void* ptr) {
        uint32_t n = 256;
        while (reinterpret_cast<uintptr_t>(ptr) % n)
            n /= 2;
        return n;
    };
    for (auto entry :
         {std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a.data},
          std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b.data},
          std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, static_cast<const void*>(y->data)},
          std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                    static_cast<const void*>(y->data)}}) {
        auto n = alignment(entry.second);
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(d.pref, entry.first, &n, sizeof(n)));
    }
    cublasLtMatmulHeuristicResult_t algo{};
    int count = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(state.lt(), d.op, d.a, d.b, d.c, d.c, d.pref, 1,
                                                &algo, &count));
    GGML_ASSERT(count > 0);
    float alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasLtMatmul(state.lt(), d.op, &alpha, a.data, d.a, b.data, d.b, &beta, y->data,
                                d.c, y->data, d.c, &algo.algo, work.get(), bytes, ctx.stream()));
}
static void pose(ggml_backend_cuda_context& ctx, State& state, ggml_tensor* y) {
    auto s = ctx.stream();
    int n = y->ne[2];
    ggml_cuda_pool_alloc<float> a(ctx.pool(), 9 * n), u(ctx.pool(), 9 * n), v(ctx.pool(), 9 * n),
        singular(ctx.pool(), 3 * n);
    ggml_cuda_pool_alloc<int> info(ctx.pool(), n);
    prepare_svd<<<grid(n * 3), 256, 0, s>>>(static_cast<const float*>(y->src[0]->data), a.get(), n);
    struct Svd {
        gesvdjInfo_t value{};
        ~Svd() {
            if (value)
                cusolverDnDestroyGesvdjInfo(value);
        }
    } settings;
#define L4DGS_SOLVER(call) GGML_ASSERT((call) == CUSOLVER_STATUS_SUCCESS)
    L4DGS_SOLVER(cusolverDnCreateGesvdjInfo(&settings.value));
    L4DGS_SOLVER(cusolverDnXgesvdjSetTolerance(settings.value, 1e-7));
    L4DGS_SOLVER(cusolverDnXgesvdjSetMaxSweeps(settings.value, 100));
    int size = 0;
    L4DGS_SOLVER(cusolverDnSgesvdjBatched_bufferSize(state.svd(), CUSOLVER_EIG_MODE_VECTOR, 3, 3,
                                                     a.get(), 3, singular.get(), u.get(), 3,
                                                     v.get(), 3, &size, settings.value, n));
    ggml_cuda_pool_alloc<float> work(ctx.pool(), size);
    L4DGS_SOLVER(cusolverDnSgesvdjBatched(state.svd(), CUSOLVER_EIG_MODE_VECTOR, 3, 3, a.get(), 3,
                                          singular.get(), u.get(), 3, v.get(), 3, work.get(), size,
                                          info.get(), settings.value, n));
    pose_kernel<<<grid(n), 256, 0, s>>>(u.get(), v.get(),
                                        static_cast<const float*>(y->src[1]->data), info.get(),
                                        static_cast<float*>(y->data), n);
#undef L4DGS_SOLVER
}
__global__ void gaussian_packed(const float* x, float* y, int64_t rows) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= rows)
        return;
    for (int c = 0; c < 3; ++c)
        y[i * 8 + c] = expf(x[i * 8 + c]);
    float norm = 0;
    for (int c = 0; c < 4; ++c) {
        float a = x[i * 8 + 3 + c];
        norm += a * a;
    }
    norm = fmaxf(sqrtf(norm), 1e-12f);
    for (int c = 0; c < 4; ++c)
        y[i * 8 + 3 + c] = x[i * 8 + 3 + c] / norm;
    y[i * 8 + 7] = 1.f / (1.f + expf(-x[i * 8 + 7]));
}
} // namespace ggml_l4dgs_cuda

bool ggml_cuda_l4dgs_supports_op(const ggml_tensor* op) {
    if (!ggml_cuda_l4dgs_op(op->op) || !ggml_is_contiguous(op))
        return false;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const auto* x = op->src[i];
        if (!x)
            continue;
        if (!ggml_is_contiguous(x))
            return false;
        if (ggml_cuda_l4dgs_host_input(op, i)) {
            if (x->type != GGML_TYPE_I32 || (x->buffer && !ggml_backend_buffer_is_host(x->buffer)))
                return false;
        }
    }
    return true; // Constructors validate individual type/shape contracts.
}
void ggml_cuda_l4dgs_compute(ggml_backend_cuda_context& ctx, ggml_tensor* y) {
    using namespace ggml_l4dgs_cuda;
    GGML_ASSERT(ggml_cuda_l4dgs_supports_op(y));
    if (!ctx.l4dgs_state)
        ctx.l4dgs_state = std::make_shared<ggml_cuda_l4dgs_state>();
    auto& state = ctx.l4dgs_state->get(ctx.stream());
    auto* x = y->src[0];
    auto* b = y->src[1];
    auto* z = y->src[2];
    auto s = ctx.stream();
    auto n = ggml_nelements(y);
    auto* p = y->op_params;
    float f;
    memcpy(&f, p, sizeof(f));
    auto fdata = [](ggml_tensor* t) { return static_cast<float*>(t->data); };
    switch (y->op) {
    case GGML_OP_L4DGS_LINEAR:
        linear(ctx, state, y);
        break;
    case GGML_OP_L4DGS_NORM:
        norm(ctx, y);
        break;
    case GGML_OP_L4DGS_RESIDUAL:
        residual_kernel<<<grid(n), 256, 0, s>>>(x->data, x->type, b->data, b->type,
                                                z ? z->data : nullptr, z ? z->type : GGML_TYPE_F32,
                                                y->data, y->type, n, x->ne[0]);
        break;
    case GGML_OP_L4DGS_ROPE2D:
        rope_kernel<<<grid(n), 256, 0, s>>>(
            x->data, y->data, x->type, static_cast<int64_t*>(b->data), n, x->ne[1], x->ne[0], f);
        break;
    case GGML_OP_L4DGS_POSITIONS2D:
        positions_kernel<<<grid(n), 256, 0, s>>>(static_cast<int64_t*>(y->data), n, p[0], p[1],
                                                 p[3]);
        break;
    case GGML_OP_L4DGS_GATHER_ROWS:
    case GGML_OP_L4DGS_SCATTER_ROWS: {
        int64_t w;
        memcpy(&w, p, sizeof(w));
        w *= ggml_type_size(x->type);
        bool scatter = y->op == GGML_OP_L4DGS_SCATTER_ROWS;
        if (scatter)
            CUDA_CHECK(
                cudaMemcpyAsync(y->data, z->data, ggml_nbytes(y), cudaMemcpyDeviceToDevice, s));
        int64_t bytes = scatter ? ggml_nbytes(x) : ggml_nbytes(y);
        if (bytes)
            rows_kernel<<<grid(bytes), 256, 0, s>>>(
                static_cast<const char*>(x->data), static_cast<char*>(y->data),
                static_cast<int32_t*>(b->data), bytes, w, scatter);
        break;
    }
    case GGML_OP_L4DGS_NORMALIZE_RGB:
        rgb_kernel<<<grid(n), 256, 0, s>>>(x->data, x->type, fdata(y), n);
        break;
    case GGML_OP_L4DGS_INTERPOLATE: {
        Converted in(ctx, x, GGML_TYPE_F32);
        int h = p[0], w = p[1], size = p[2], c = x->ne[0];
        if (h == size && w == size) {
            CUDA_CHECK(
                cudaMemcpyAsync(y->data, in.data, ggml_nbytes(y), cudaMemcpyDeviceToDevice, s));
            break;
        }
        ggml_cuda_pool_alloc<float> temp(ctx.pool(), int64_t(c) * w * size);
        interpolate_kernel<<<grid(int64_t(c) * w * size), 256, 0, s>>>(
            static_cast<const float*>(in.data), temp.get(), c, size, size, size, w, true, true,
            false);
        interpolate_kernel<<<grid(int64_t(c) * h * w), 256, 0, s>>>(temp.get(), fdata(y), c, size,
                                                                    w, h, w, false, false, true);
        CUDA_CHECK(
            cudaMemcpyAsync(y->data, in.data, c * sizeof(float), cudaMemcpyDeviceToDevice, s));
        break;
    }
    case GGML_OP_L4DGS_ADD_POSITIONS: {
        Converted pos(ctx, b, GGML_TYPE_F32);
        add_position_kernel<<<grid(n), 256, 0, s>>>(
            x->data, x->type, static_cast<const float*>(pos.data), static_cast<int64_t*>(z->data),
            fdata(y), x->ne[0], b->ne[1] - 1, n);
        break;
    }
    case GGML_OP_L4DGS_PIXEL_SHUFFLE:
        shuffle_kernel<<<grid(n), 256, 0, s>>>(x->data, y->data, x->type, p[0], p[1], p[2],
                                               y->ne[0], n);
        break;
    case GGML_OP_L4DGS_SPATIAL_MEAN:
        mean_kernel<<<grid(n), 256, 0, s>>>(x->data, x->type, y->data, x->ne[0],
                                            ggml_nelements(x) / n, n);
        break;
    case GGML_OP_L4DGS_GAUSSIAN:
        gaussian_packed<<<grid(n / 8), 256, 0, s>>>(fdata(x), fdata(y), n / 8);
        break;
    case GGML_OP_L4DGS_POSE:
        pose(ctx, state, y);
        break;
    case GGML_OP_L4DGS_DEPTH_TO_XYZ:
        depth_kernel<<<grid(n / 3), 256, 0, s>>>(fdata(x), fdata(y), n / 3);
        break;
    case GGML_OP_L4DGS_TRANSFORM_POINTS:
        transform_kernel<<<grid(n), 256, 0, s>>>(fdata(x), fdata(b), fdata(y), n,
                                                 (n / 3) / (ggml_nelements(b) / 16));
        break;
    case GGML_OP_L4DGS_PREFILL_RAGGED:
    case GGML_OP_L4DGS_PREFILL_PAGED:
        prefill(ctx, state, y);
        break;
    default:
        GGML_ABORT("Unknown Live4DGS op");
    }
    CUDA_CHECK(cudaGetLastError());
}
