#pragma once
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward-only CUDA ops preserving Live4DGS arithmetic. All floating inputs are
// contiguous F32/BF16 unless documented otherwise. No external stream is needed.
GGML_API struct ggml_tensor* ggml_splatcodec_linear(struct ggml_context*, struct ggml_tensor* x,
                                                    struct ggml_tensor* weight,
                                                    struct ggml_tensor* bias,
                                                    enum ggml_type output_type);
GGML_API struct ggml_tensor* ggml_splatcodec_norm(struct ggml_context*, struct ggml_tensor* x,
                                                  struct ggml_tensor* weight,
                                                  struct ggml_tensor* bias, float eps);
GGML_API struct ggml_tensor* ggml_splatcodec_residual(struct ggml_context*, struct ggml_tensor* x,
                                                      struct ggml_tensor* branch,
                                                      struct ggml_tensor* scale);
GGML_API struct ggml_tensor* ggml_splatcodec_rope2d(struct ggml_context*, struct ggml_tensor* x,
                                                    struct ggml_tensor* positions, float base);
// Rows are flattened tokens of row_width elements; indices are I32. Scatter
// returns a new tensor initialized from base, preserving unaddressed rows.
GGML_API struct ggml_tensor* ggml_splatcodec_gather_rows(struct ggml_context*,
                                                         struct ggml_tensor* x,
                                                         struct ggml_tensor* indices,
                                                         int64_t row_width);
GGML_API struct ggml_tensor* ggml_splatcodec_scatter_rows(struct ggml_context*,
                                                          struct ggml_tensor* x,
                                                          struct ggml_tensor* indices,
                                                          struct ggml_tensor* base,
                                                          int64_t row_width);
GGML_API struct ggml_tensor* ggml_splatcodec_normalize_rgb(struct ggml_context*,
                                                           struct ggml_tensor*);
GGML_API struct ggml_tensor* ggml_splatcodec_positions2d(struct ggml_context*, int h, int w,
                                                         int batches, int special);
GGML_API struct ggml_tensor* ggml_splatcodec_interpolate(struct ggml_context*, struct ggml_tensor*,
                                                         int h, int w);
GGML_API struct ggml_tensor* ggml_splatcodec_add_positions(struct ggml_context*,
                                                           struct ggml_tensor* x,
                                                           struct ggml_tensor* embedding,
                                                           struct ggml_tensor* indices);
GGML_API struct ggml_tensor* ggml_splatcodec_pixel_shuffle(struct ggml_context*,
                                                           struct ggml_tensor*, int h, int w,
                                                           int patch, int batches);
GGML_API struct ggml_tensor* ggml_splatcodec_spatial_mean(struct ggml_context*, struct ggml_tensor*,
                                                          int batches);
// F32 [8, ...] -> [8, ...]: exp(scale), normalized quaternion, sigmoid(opacity).
GGML_API struct ggml_tensor* ggml_splatcodec_gaussian(struct ggml_context*, struct ggml_tensor*);
GGML_API struct ggml_tensor* ggml_splatcodec_pose(struct ggml_context*,
                                                  struct ggml_tensor* rotation,
                                                  struct ggml_tensor* translation);
GGML_API struct ggml_tensor* ggml_splatcodec_depth_to_xyz(struct ggml_context*,
                                                          struct ggml_tensor*);
GGML_API struct ggml_tensor* ggml_splatcodec_transform_points(struct ggml_context*,
                                                              struct ggml_tensor*,
                                                              struct ggml_tensor* poses);
// Q/K/V: BF16 [64, heads, rows]. Metadata: immutable, contiguous I32 host-buffer
// tensors, retained until graph completion. Indptr starts at 0 and is monotone.
// Ragged KV uses independent segment offsets. Paged KV uses one token per page;
// page_indices contains physical KV row indices. Empty requests must be omitted.
// Metadata content is part of the backend plan-cache key, not tensor addresses.
GGML_API struct ggml_tensor*
ggml_flashinfer_prefill_ragged(struct ggml_context*, struct ggml_tensor* q, struct ggml_tensor* k,
                               struct ggml_tensor* v, struct ggml_tensor* qo_indptr,
                               struct ggml_tensor* kv_indptr);
GGML_API struct ggml_tensor*
ggml_flashinfer_prefill_paged(struct ggml_context*, struct ggml_tensor* q, struct ggml_tensor* k,
                              struct ggml_tensor* v, struct ggml_tensor* qo_indptr,
                              struct ggml_tensor* kv_indptr, struct ggml_tensor* page_indices);

#ifdef __cplusplus
}
#endif
