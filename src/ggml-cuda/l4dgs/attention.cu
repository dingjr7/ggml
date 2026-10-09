#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>

#include "support.cuh"

namespace ggml_l4dgs_cuda {
using namespace flashinfer;
using BFloat = __nv_bfloat16;
using Paged = BatchPrefillPagedParams<BFloat, BFloat, BFloat, int32_t>;
using Ragged = BatchPrefillRaggedParams<BFloat, BFloat, BFloat, int32_t>;
using Variant = DefaultAttention<false, false, false, false>;
struct Buffer {
    void* data = nullptr;
    size_t bytes = 0;
    cudaStream_t stream{};
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (data)
            CUDA_CHECK(cudaFreeAsync(data, stream));
    }
    void alloc(size_t n, cudaStream_t s) {
        GGML_ASSERT(!data);
        stream = s;
        bytes = std::max(n, size_t(1));
        CUDA_CHECK(cudaMallocAsync(&data, bytes, s));
    }
    void upload(const std::vector<int32_t>& v, cudaStream_t s) {
        alloc(v.size() * sizeof(int32_t), s);
        if (!v.empty())
            CUDA_CHECK(cudaMemcpyAsync(data, v.data(), v.size() * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, s));
    }
    int32_t* ids() { return static_cast<int32_t*>(data); }
};
struct Plan {
    PrefillPlanInfo info;
    Buffer integers, floats, qptr, kvptr, pages, last;
    std::vector<int32_t> hq, hk, hp, hl;
    void* pinned = nullptr;
    cudaStream_t stream{};
    ~Plan() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (pinned)
            CUDA_CHECK(cudaFreeHost(pinned));
    }
};
static std::vector<int32_t> host_indices(const ggml_tensor* t) {
    GGML_ASSERT(t && t->type == GGML_TYPE_I32 && t->buffer &&
                ggml_backend_buffer_is_host(t->buffer));
    auto* p = static_cast<const int32_t*>(t->data);
    return {p, p + ggml_nelements(t)};
}
static std::shared_ptr<Plan> plan(ggml_tensor* y, State& state) {
    auto hq = host_indices(y->src[3]), hk = host_indices(y->src[4]);
    bool paged = y->op == GGML_OP_L4DGS_PREFILL_PAGED;
    std::vector<int32_t> hp = paged ? host_indices(y->src[5]) : std::vector<int32_t>{};
    GGML_ASSERT(hq.size() == hk.size() && hq.size() > 1 && hq.size() <= INT_MAX &&
                hq.front() == 0 && hk.front() == 0);
    GGML_ASSERT(hq.back() == y->ne[2] &&
                hk.back() == (paged ? int64_t(hp.size()) : y->src[1]->ne[2]));
    for (size_t i = 1; i < hq.size(); ++i) {
        GGML_ASSERT(hq[i] > hq[i - 1] && hk[i] > hk[i - 1]);
    }
    for (auto id : hp)
        GGML_ASSERT(id >= 0 && id < y->src[1]->ne[2]);
    std::vector<int64_t> key = {
        paged, y->ne[1], y->ne[2], y->src[1]->ne[2], int64_t(hq.size()), int64_t(hp.size())};
    key.insert(key.end(), hq.begin(), hq.end());
    key.insert(key.end(), hk.begin(), hk.end());
    key.insert(key.end(), hp.begin(), hp.end());
    auto found = state.plans.find(key);
    if (found != state.plans.end())
        return found->second;
    auto p = std::make_shared<Plan>();
    p->stream = state.stream;
    p->hq = std::move(hq);
    p->hk = std::move(hk);
    p->hp = std::move(hp);
    p->hl.assign(p->hq.size() - 1, 1);
    size_t fb = 0, ib = 0;
    int requests = p->hq.size() - 1, heads = y->ne[1];
    auto s = state.stream;
    CUDA_CHECK(PrefillPlanWorkspaceSize<int32_t>(fb, ib, p->hq.data(), p->hk.data(), y->ne[2],
                                                 requests, heads, heads, 64, 64, 1, false, 2, -1,
                                                 -1, false, 0, s));
    p->floats.alloc(fb, s);
    p->integers.alloc(ib, s);
    CUDA_CHECK(cudaMallocHost(&p->pinned, p->integers.bytes));
    CUDA_CHECK(PrefillPlan<int32_t>(p->floats.data, p->floats.bytes, p->integers.data, p->pinned,
                                    p->integers.bytes, p->info, p->hq.data(), p->hk.data(),
                                    y->ne[2], requests, heads, heads, 64, 64, 1, false, 2, -1, -1,
                                    false, 0, s));
    p->qptr.upload(p->hq, s);
    p->kvptr.upload(p->hk, s);
    if (paged) {
        p->pages.upload(p->hp, s);
        p->last.upload(p->hl, s);
    }
    state.plans.emplace(std::move(key), p);
    return p;
}
template <class Params> static void schedule(Params& a, Plan& p, ggml_tensor* y) {
    a.q = static_cast<BFloat*>(y->src[0]->data);
    a.o = static_cast<BFloat*>(y->data);
    a.q_indptr = p.qptr.ids();
    a.num_qo_heads = y->ne[1];
    a.group_size = uint_fastdiv(1);
    a.q_stride_n = y->ne[0] * y->ne[1];
    a.q_stride_h = y->ne[0];
    a.window_left = -1;
    a.sm_scale = 1.f / 8.f;
    auto ptr = [&](size_t offset) {
        return reinterpret_cast<int32_t*>(static_cast<char*>(p.integers.data) + offset);
    };
    a.request_indices = ptr(p.info.request_indices_offset);
    a.qo_tile_indices = ptr(p.info.qo_tile_indices_offset);
    a.kv_tile_indices = ptr(p.info.kv_tile_indices_offset);
    a.o_indptr = ptr(p.info.o_indptr_offset);
    a.kv_chunk_size_ptr = ptr(p.info.kv_chunk_size_ptr_offset);
    if (p.info.split_kv)
        a.merge_indptr = ptr(p.info.merge_indptr_offset);
    a.padded_batch_size = p.info.padded_batch_size;
    a.max_total_num_rows = p.info.total_num_rows;
}
template <int Tile, class Params>
static cudaError_t run(Params params, BFloat* tmp, float* lse, cudaStream_t s) {
    if constexpr (std::is_same_v<Params, Paged>)
        return BatchPrefillWithPagedKVCacheDispatched<Tile, 64, 64, PosEncodingMode::kNone, false,
                                                      MaskMode::kNone, Variant, Params>(
            params, tmp, lse, false, s);
    else
        return BatchPrefillWithRaggedKVCacheDispatched<Tile, 64, 64, PosEncodingMode::kNone, false,
                                                       MaskMode::kNone, Variant, Params>(
            params, tmp, lse, false, s);
}
template <class Params> static void dispatch(Params& a, Plan& p, cudaStream_t s) {
    auto* base = static_cast<char*>(p.floats.data);
    auto* tmp = p.info.split_kv ? reinterpret_cast<BFloat*>(base + p.info.v_offset) : nullptr;
    auto* lse = p.info.split_kv ? reinterpret_cast<float*>(base + p.info.s_offset) : nullptr;
    switch (p.info.cta_tile_q) {
    case 16:
        CUDA_CHECK((run<16>(a, tmp, lse, s)));
        break;
    case 32:
        CUDA_CHECK((run<32>(a, tmp, lse, s)));
        break;
    case 64:
        CUDA_CHECK((run<64>(a, tmp, lse, s)));
        break;
    case 128:
        CUDA_CHECK((run<128>(a, tmp, lse, s)));
        break;
    default:
        GGML_ABORT("Unsupported FlashInfer query tile");
    }
}
void prefill(ggml_backend_cuda_context& ctx, State& state, ggml_tensor* y) {
    auto p = plan(y, state);
    if (y->op == GGML_OP_L4DGS_PREFILL_PAGED) {
        Paged a;
        schedule(a, *p, y);
        a.paged_kv = paged_kv_t<BFloat, int32_t>(y->ne[1], 1, 64, p->hq.size() - 1, QKVLayout::kNHD,
                                                 static_cast<BFloat*>(y->src[1]->data),
                                                 static_cast<BFloat*>(y->src[2]->data),
                                                 p->pages.ids(), p->kvptr.ids(), p->last.ids());
        dispatch(a, *p, ctx.stream());
    } else {
        Ragged a;
        schedule(a, *p, y);
        a.k = static_cast<BFloat*>(y->src[1]->data);
        a.v = static_cast<BFloat*>(y->src[2]->data);
        a.kv_indptr = p->kvptr.ids();
        a.num_kv_heads = y->ne[1];
        a.k_stride_n = a.v_stride_n = y->ne[0] * y->ne[1];
        a.k_stride_h = a.v_stride_h = y->ne[0];
        dispatch(a, *p, ctx.stream());
    }
}
} // namespace ggml_l4dgs_cuda
