#include "ggml-l4dgs.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <initializer_list>

static void floating(const ggml_tensor* x) {
    GGML_ASSERT(x && ggml_is_contiguous(x));
    GGML_ASSERT(x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_BF16);
}
static ggml_tensor* node(ggml_context* ctx, ggml_op op, ggml_type type, const int64_t* ne,
                         std::initializer_list<ggml_tensor*> inputs) {
    auto* y = ggml_new_tensor(ctx, type, 4, ne);
    y->op = op;
    std::copy(inputs.begin(), inputs.end(), y->src);
    return y;
}
static void ints(ggml_tensor* y, std::initializer_list<int32_t> values) {
    std::copy(values.begin(), values.end(), y->op_params);
}
static void scalar(ggml_tensor* y, float value) {
    std::memcpy(y->op_params, &value, sizeof(value));
}
static void width(ggml_tensor* y, int64_t value) {
    std::memcpy(y->op_params, &value, sizeof(value));
}
static void channel_weight(const ggml_tensor* x, const ggml_tensor* w) {
    if (w) {
        floating(w);
        GGML_ASSERT(ggml_nelements(w) == x->ne[0]);
    }
}
ggml_tensor* ggml_l4dgs_linear(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                               ggml_type t) {
    floating(x);
    floating(w);
    GGML_ASSERT((t == GGML_TYPE_F32 || t == GGML_TYPE_BF16) && x->ne[0] > 0);
    GGML_ASSERT(w->ne[0] == x->ne[0] && w->ne[1] > 0 && ggml_is_matrix(w));
    if (b) {
        floating(b);
        GGML_ASSERT(ggml_nelements(b) == w->ne[1]);
    }
    int64_t ne[4];
    std::copy(x->ne, x->ne + 4, ne);
    ne[0] = w->ne[1];
    return node(c, GGML_OP_L4DGS_LINEAR, t, ne, {x, w, b});
}
ggml_tensor* ggml_l4dgs_norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                             float eps) {
    floating(x);
    channel_weight(x, w);
    channel_weight(x, b);
    GGML_ASSERT(x->ne[0] > 0 && eps > 0);
    auto* y = node(c, GGML_OP_L4DGS_NORM, x->type, x->ne, {x, w, b});
    scalar(y, eps);
    return y;
}
ggml_tensor* ggml_l4dgs_residual(ggml_context* c, ggml_tensor* x, ggml_tensor* b, ggml_tensor* s) {
    floating(x);
    floating(b);
    channel_weight(x, s);
    GGML_ASSERT(ggml_are_same_shape(x, b));
    auto t = x->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F32 || (s && s->type == GGML_TYPE_F32)
                 ? GGML_TYPE_F32
                 : GGML_TYPE_BF16;
    return node(c, GGML_OP_L4DGS_RESIDUAL, t, x->ne, {x, b, s});
}
ggml_tensor* ggml_l4dgs_rope2d(ggml_context* c, ggml_tensor* x, ggml_tensor* p, float base) {
    floating(x);
    GGML_ASSERT(x->ne[0] > 0 && x->ne[0] % 4 == 0 && x->ne[3] == 1 && base > 0);
    GGML_ASSERT(p && p->type == GGML_TYPE_I64 && ggml_is_contiguous(p) && p->ne[0] == 2 &&
                ggml_nelements(p) == 2 * x->ne[2]);
    auto* y = node(c, GGML_OP_L4DGS_ROPE2D, x->type, x->ne, {x, p});
    scalar(y, base);
    return y;
}
static void row_args(ggml_tensor* x, ggml_tensor* ids, int64_t w) {
    GGML_ASSERT(x && ggml_is_contiguous(x) && !ggml_is_quantized(x->type) && w > 0 &&
                ggml_nelements(x) % w == 0);
    GGML_ASSERT(ids && ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
}
ggml_tensor* ggml_l4dgs_gather_rows(ggml_context* c, ggml_tensor* x, ggml_tensor* ids, int64_t w) {
    row_args(x, ids, w);
    int64_t ne[4] = {w, ggml_nelements(ids), 1, 1};
    auto* y = node(c, GGML_OP_L4DGS_GATHER_ROWS, x->type, ne, {x, ids});
    width(y, w);
    return y;
}
ggml_tensor* ggml_l4dgs_scatter_rows(ggml_context* c, ggml_tensor* x, ggml_tensor* ids,
                                     ggml_tensor* b, int64_t w) {
    row_args(x, ids, w);
    row_args(b, ids, w);
    GGML_ASSERT(x->type == b->type && ggml_nelements(x) / w == ggml_nelements(ids));
    auto* y = node(c, GGML_OP_L4DGS_SCATTER_ROWS, b->type, b->ne, {x, ids, b});
    width(y, w);
    return y;
}
ggml_tensor* ggml_l4dgs_normalize_rgb(ggml_context* c, ggml_tensor* x) {
    floating(x);
    GGML_ASSERT(x->ne[0] == 588);
    return node(c, GGML_OP_L4DGS_NORMALIZE_RGB, GGML_TYPE_F32, x->ne, {x});
}
ggml_tensor* ggml_l4dgs_positions2d(ggml_context* c, int h, int w, int b, int special) {
    GGML_ASSERT(h > 0 && w > 0 && b > 0 && special >= 0 && int64_t(h) * w + special <= INT_MAX);
    int64_t ne[4] = {2, int64_t(b) * (int64_t(h) * w + special), 1, 1};
    auto* y = node(c, GGML_OP_L4DGS_POSITIONS2D, GGML_TYPE_I64, ne, {});
    ints(y, {h, w, b, special});
    return y;
}
ggml_tensor* ggml_l4dgs_interpolate(ggml_context* c, ggml_tensor* x, int h, int w) {
    floating(x);
    GGML_ASSERT(ggml_is_matrix(x) && h > 0 && w > 0 && x->ne[1] > 1);
    int size = int(std::sqrt(double(x->ne[1] - 1)));
    GGML_ASSERT(int64_t(size) * size == x->ne[1] - 1);
    int64_t ne[4] = {x->ne[0], 1 + int64_t(h) * w, 1, 1};
    auto* y = node(c, GGML_OP_L4DGS_INTERPOLATE, GGML_TYPE_F32, ne, {x});
    ints(y, {h, w, size});
    return y;
}
ggml_tensor* ggml_l4dgs_add_positions(ggml_context* c, ggml_tensor* x, ggml_tensor* p,
                                      ggml_tensor* ids) {
    floating(x);
    floating(p);
    GGML_ASSERT(ggml_is_matrix(x) && ggml_is_matrix(p) && x->ne[0] == p->ne[0] && p->ne[1] > 1);
    GGML_ASSERT(ids && ids->type == GGML_TYPE_I64 && ggml_is_contiguous(ids) &&
                ggml_nelements(ids) == x->ne[1]);
    return node(c, GGML_OP_L4DGS_ADD_POSITIONS, GGML_TYPE_F32, x->ne, {x, p, ids});
}
ggml_tensor* ggml_l4dgs_pixel_shuffle(ggml_context* c, ggml_tensor* x, int h, int w, int p, int b) {
    floating(x);
    GGML_ASSERT(h > 0 && w > 0 && p > 0 && b > 0 && h % p == 0 && w % p == 0 &&
                x->ne[0] % (int64_t(p) * p) == 0);
    int64_t ne[4] = {x->ne[0] / p / p, w, h, b};
    GGML_ASSERT(ggml_nelements(x) == ne[0] * w * h * b);
    auto* y = node(c, GGML_OP_L4DGS_PIXEL_SHUFFLE, x->type, ne, {x});
    ints(y, {h, w, p, b});
    return y;
}
ggml_tensor* ggml_l4dgs_spatial_mean(ggml_context* c, ggml_tensor* x, int b) {
    floating(x);
    GGML_ASSERT(b > 0 && x->ne[0] > 0 && ggml_nelements(x) % (x->ne[0] * b) == 0 &&
                ggml_nelements(x) > 0);
    int64_t ne[4] = {x->ne[0], b, 1, 1};
    return node(c, GGML_OP_L4DGS_SPATIAL_MEAN, x->type, ne, {x});
}
ggml_tensor* ggml_l4dgs_gaussian(ggml_context* c, ggml_tensor* x) {
    floating(x);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == 8);
    return node(c, GGML_OP_L4DGS_GAUSSIAN, x->type, x->ne, {x});
}
ggml_tensor* ggml_l4dgs_pose(ggml_context* c, ggml_tensor* r, ggml_tensor* t) {
    floating(r);
    floating(t);
    GGML_ASSERT(r->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 && r->ne[0] == 9 &&
                t->ne[0] == 3 && ggml_nelements(r) / 9 == ggml_nelements(t) / 3 &&
                ggml_nelements(r) > 0);
    int64_t ne[4] = {4, 4, ggml_nelements(r) / 9, 1};
    return node(c, GGML_OP_L4DGS_POSE, GGML_TYPE_F32, ne, {r, t});
}
ggml_tensor* ggml_l4dgs_depth_to_xyz(ggml_context* c, ggml_tensor* x) {
    floating(x);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == 3);
    return node(c, GGML_OP_L4DGS_DEPTH_TO_XYZ, x->type, x->ne, {x});
}
ggml_tensor* ggml_l4dgs_transform_points(ggml_context* c, ggml_tensor* x, ggml_tensor* p) {
    floating(x);
    floating(p);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && p->type == GGML_TYPE_F32 && x->ne[0] == 3 &&
                p->ne[0] == 4 && p->ne[1] == 4 && ggml_nelements(p) > 0 &&
                (ggml_nelements(x) / 3) % (ggml_nelements(p) / 16) == 0);
    return node(c, GGML_OP_L4DGS_TRANSFORM_POINTS, x->type, x->ne, {x, p});
}
static ggml_tensor* prefill(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                            ggml_tensor* qp, ggml_tensor* kp, ggml_tensor* pages) {
    floating(q);
    floating(k);
    floating(v);
    GGML_ASSERT(q->type == GGML_TYPE_BF16 && k->type == q->type && v->type == q->type &&
                ggml_are_same_shape(k, v));
    GGML_ASSERT(q->ne[0] == 64 && k->ne[0] == 64 && q->ne[1] > 0 && q->ne[1] == k->ne[1] &&
                q->ne[3] == 1 && k->ne[3] == 1 && q->ne[2] > 0 && q->ne[2] <= INT_MAX &&
                k->ne[2] > 0 && k->ne[2] <= INT_MAX);
    for (auto* m : {qp, kp}) {
        GGML_ASSERT(m && m->type == GGML_TYPE_I32 && ggml_is_vector(m) && ggml_is_contiguous(m) &&
                    ggml_nelements(m) >= 2);
    }
    GGML_ASSERT(ggml_nelements(qp) == ggml_nelements(kp));
    if (pages) {
        GGML_ASSERT(pages->type == GGML_TYPE_I32 && ggml_is_vector(pages) &&
                    ggml_is_contiguous(pages));
    }
    return node(c, pages ? GGML_OP_L4DGS_PREFILL_PAGED : GGML_OP_L4DGS_PREFILL_RAGGED, q->type,
                q->ne, {q, k, v, qp, kp, pages});
}
ggml_tensor* ggml_l4dgs_prefill_ragged(ggml_context* c, ggml_tensor* q, ggml_tensor* k,
                                       ggml_tensor* v, ggml_tensor* qp, ggml_tensor* kp) {
    return prefill(c, q, k, v, qp, kp, nullptr);
}
ggml_tensor* ggml_l4dgs_prefill_paged(ggml_context* c, ggml_tensor* q, ggml_tensor* k,
                                      ggml_tensor* v, ggml_tensor* qp, ggml_tensor* kp,
                                      ggml_tensor* pages) {
    GGML_ASSERT(pages);
    return prefill(c, q, k, v, qp, kp, pages);
}
