// Derived from splatcodec Live4DGS kernels; see THIRD_PARTY_NOTICES.md.
__global__ void cast_kernel(const void* in, ggml_type it, void* out, ggml_type ot, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        save(out, ot, i, load(in, it, i));
}

__global__ void residual_kernel(const void* x, ggml_type xt, const void* b, ggml_type bt,
                                const void* scale, ggml_type st, void* out, ggml_type ot, int64_t n,
                                int channels) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) {
        float branch = load(b, bt, i);
        if (scale)
            branch *= load(scale, st, i % channels);
        // LayerScale is a separate operation in the reference, including BF16 rounding.
        if (scale && bt == GGML_TYPE_BF16 && st == GGML_TYPE_BF16)
            branch = __bfloat162float(__float2bfloat16(branch));
        save(out, ot, i, load(x, xt, i) + branch);
    }
}

__global__ void rope_kernel(const void* x, void* y, ggml_type t, const int64_t* positions,
                            int64_t n, int heads, int d, float base) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n)
        return;
    int c = i % d, axis = c / (d / 2), j = c % (d / 2), half = d / 4;
    int64_t row = i / (heads * d);
    // The source einsum is autocast: frequencies are BF16 before the outer product.
    float frequency = 1.f / powf(base, float(2 * (j % half)) / float(d / 2));
    if (t == GGML_TYPE_BF16)
        frequency = __bfloat162float(__float2bfloat16(frequency));
    float angle = float(positions[row * 2 + axis]) * frequency;
    if (t == GGML_TYPE_BF16)
        angle = __bfloat162float(__float2bfloat16(angle));
    float co = cosf(angle), si = sinf(angle);
    if (t == GGML_TYPE_BF16) {
        co = __bfloat162float(__float2bfloat16(co));
        si = __bfloat162float(__float2bfloat16(si));
    }
    int64_t other = i + (j < half ? half : -half);
    float v = load(x, t, i) * co + (j < half ? -1.f : 1.f) * load(x, t, other) * si;
    save(y, t, i, v);
}

__global__ void positions_kernel(int64_t* out, int64_t n, int h, int w, int special) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n)
        return;
    int token = (i / 2) % (h * w + special), axis = i % 2;
    out[i] = token < special
                 ? 0
                 : ((axis ? (token - special) % w : (token - special) / w) + (special ? 1 : 0));
}

__global__ void rows_kernel(const char* src, char* dst, const int32_t* ids, int64_t n,
                            int64_t width, bool scatter) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n)
        return;
    int64_t mapped = int64_t(ids[i / width]) * width + i % width;
    if (scatter)
        dst[mapped] = src[i];
    else
        dst[i] = src[mapped];
}

__global__ void rgb_kernel(const void* x, ggml_type t, float* y, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n)
        return;
    constexpr float mean[3] = {.485f, .456f, .406f}, std[3] = {.229f, .224f, .225f};
    int c = (i / 196) % 3;
    y[i] = (load(x, t, i) - mean[c]) / std[c];
}

__device__ float cubic(float x) {
    x = fabsf(x);
    if (x < 1.f)
        return ((1.5f * x - 2.5f) * x) * x + 1.f;
    if (x < 2.f)
        return ((-.5f * x + 2.5f) * x - 4.f) * x + 2.f;
    return 0.f;
}

__global__ void interpolate_kernel(const float* x, float* y, int c, int ih, int iw, int oh, int ow,
                                   bool horizontal, bool input_cls, bool output_cls) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x, n = int64_t(c) * oh * ow;
    if (i >= n)
        return;
    int channel = i % c, ox = (i / c) % ow, oy = i / (c * ow);
    int in_size = horizontal ? iw : ih, out_size = horizontal ? ow : oh,
        index = horizontal ? ox : oy;
    float scale = float(in_size) / out_size, aa = fmaxf(1.f, scale), center = scale * (index + .5f),
          support = 2.f * aa;
    int start = max(0, int(floorf(center - support + .5f))),
        end = min(in_size, int(floorf(center + support + .5f)));
    float sum = 0.f, weight = 0.f;
    for (int j = start; j < end; ++j) {
        float w = cubic((j - center + .5f) / aa);
        int sx = horizontal ? j : ox, sy = horizontal ? oy : j;
        sum += x[(int64_t(sy) * iw + sx + (input_cls ? 1 : 0)) * c + channel] * w;
        weight += w;
    }
    y[i + (output_cls ? c : 0)] = sum / weight;
}

__global__ void add_position_kernel(const void* x, ggml_type xt, const float* pos,
                                    const int64_t* ids, float* y, int c, int patches, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < n)
        y[i] = load(x, xt, i) + pos[(ids[i / c] % patches + 1) * c + i % c];
}

__global__ void shuffle_kernel(const void* x, void* y, ggml_type type, int h, int w, int patch,
                               int c, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n)
        return;
    int channel = i % c, px = (i / c) % w, py = (i / (c * w)) % h, b = i / (int64_t(c) * w * h);
    int64_t row = (b * (h / patch) + py / patch) * (w / patch) + px / patch;
    int feature = channel * patch * patch + (py % patch) * patch + px % patch;
    save(y, type, i, load(x, type, row * c * patch * patch + feature));
}

__global__ void mean_kernel(const void* x, ggml_type t, void* y, int c, int rows, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n)
        return;
    int channel = i % c, b = i / c;
    float sum = 0;
    for (int j = 0; j < rows; ++j)
        sum += load(x, t, (int64_t(b) * rows + j) * c + channel);
    save(y, t, i, sum / rows);
}

__global__ void prepare_svd(const float* x, float* a, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * 3)
        return;
    float norm = 0;
    for (int j = 0; j < 3; ++j)
        norm += x[i * 3 + j] * x[i * 3 + j];
    norm = fmaxf(sqrtf(norm), 1e-12f);
    for (int j = 0; j < 3; ++j)
        a[i * 3 + j] = x[i * 3 + j] / norm;
    // Row-major normalized M is column-major M^T, exactly the reference SVD input.
}

__global__ void pose_kernel(const float* u, const float* v, const float* t, const int* info,
                            float* pose, int n) {
    int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= n)
        return;
    float r[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float z = 0;
            for (int k = 0; k < 3; ++k)
                z += v[b * 9 + i + k * 3] * u[b * 9 + j + k * 3];
            r[i * 3 + j] = z;
        }
    float det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                r[2] * (r[3] * r[7] - r[4] * r[6]);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float z = 0;
            if (i < 3 && j < 3)
                for (int k = 0; k < 3; ++k)
                    z += v[b * 9 + i + k * 3] * (k == 2 ? det : 1.f) * u[b * 9 + j + k * 3];
            else if (i < 3)
                z = t[b * 3 + i];
            else if (j == 3)
                z = 1;
            pose[b * 16 + i * 4 + j] = info[b] == 0 ? z : nanf("");
        }
}

__global__ void depth_kernel(const float* x, float* y, int64_t rows) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= rows)
        return;
    float depth = expf(x[i * 3 + 2]);
    y[i * 3] = x[i * 3] * depth;
    y[i * 3 + 1] = x[i * 3 + 1] * depth;
    y[i * 3 + 2] = depth;
}

__global__ void transform_kernel(const float* x, const float* poses, float* y, int64_t n,
                                 int64_t per_pose) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n)
        return;
    int axis = i % 3;
    int64_t point = i / 3;
    const float* p = poses + (point / per_pose) * 16 + axis * 4;
    y[i] = p[0] * x[point * 3] + p[1] * x[point * 3 + 1] + p[2] * x[point * 3 + 2] + p[3];
}
