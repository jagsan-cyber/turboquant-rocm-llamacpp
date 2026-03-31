/**
 * turboquant_hip.hip  —  ROCm/HIP GPU kernels (WHT-Focused Optimized Version)
 */

#include "turboquant.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <cmath>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#if defined(__gfx1201__) || defined(__gfx1200__)
#  define WARP_SIZE 32
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1030__)
#  define WARP_SIZE 32
#else
#  define WARP_SIZE 64
#endif

#define MAX_DIM 256

__constant__ float d_cb2_levels[4]  = {-0.7071f,-0.2357f, 0.2357f, 0.7071f};
__constant__ float d_cb2_thresh[5]  = {-1.0f,-0.5f,0.0f,0.5f,1.0f};
__constant__ float d_cb3_levels[8]  = {-0.9009f,-0.6235f,-0.3612f,-0.1205f, 0.1205f, 0.3612f, 0.6235f, 0.9009f};
__constant__ float d_cb3_thresh[9]  = {-1.0f,-0.7654f,-0.5000f,-0.2393f,0.0f, 0.2393f,0.5000f,0.7654f,1.0f};
__constant__ float d_cb4_levels[16] = {
    -0.9613f,-0.8315f,-0.6895f,-0.5345f, -0.3681f,-0.1951f,-0.0980f, 0.0000f,
     0.0980f, 0.1951f, 0.3681f, 0.5345f,  0.6895f, 0.8315f, 0.9613f, 0.9903f};
__constant__ float d_cb4_thresh[17] = {
    -1.0f,   -0.9009f,-0.7654f,-0.6124f, -0.4540f,-0.2820f,-0.1464f,-0.0490f,
     0.0490f, 0.1464f, 0.2820f, 0.4540f,  0.6124f, 0.7654f, 0.9009f, 0.9808f,1.0f};

__device__ __forceinline__ int tq_scalar_quantize(float v, int bits) {
    v = fmaxf(-1.0f, fminf(1.0f, v));
    if (bits == 2) {
        if (v < d_cb2_thresh[1]) return 0;
        if (v < d_cb2_thresh[2]) return 1;
        if (v < d_cb2_thresh[3]) return 2;
        return 3;
    }
    if (bits == 3) {
        int lo = 0, hi = 7;
        while (lo < hi) { int m = (lo+hi)/2; if (v < d_cb3_thresh[m+1]) hi=m; else lo=m+1; }
        return lo;
    }
    int lo = 0, hi = 15;
    while (lo < hi) { int m = (lo+hi)/2; if (v < d_cb4_thresh[m+1]) hi=m; else lo=m+1; }
    return lo;
}

__device__ __forceinline__ float tq_scalar_reconstruct(int idx, int bits) {
    if (bits == 2) return d_cb2_levels[idx];
    if (bits == 3) return d_cb3_levels[idx];
    return d_cb4_levels[idx];
}

__global__ void tq_rotate_whd_kernel(
    float *__restrict__ x, const float *__restrict__ sign1, const float *__restrict__ sign2,
    int head_dim, int n_heads, float inv_sqrt_d)
{
    __shared__ float smem[MAX_DIM];
    int tid  = threadIdx.x;
    int head = blockIdx.y;
    int tok  = blockIdx.x;
    if (tid >= head_dim) return;

    int base = (tok * n_heads + head) * head_dim;
    float val = x[base + tid] * sign1[tid];

    for (int s = 1; s < WARP_SIZE && s < head_dim; s <<= 1) {
        float other = __shfl_xor(val, s);
        val = ((tid & s) == 0) ? val + other : other - val;
    }

    if (head_dim > WARP_SIZE) {
        smem[tid] = val; __syncthreads();
        for (int s = WARP_SIZE; s < head_dim; s <<= 1) {
            int pair = tid ^ s;
            float a = smem[tid], b = smem[pair]; __syncthreads();
            if ((tid & s) == 0) { smem[tid] = a+b; smem[pair] = a-b; }
            __syncthreads();
        }
        val = smem[tid];
    }
    x[base + tid] = val * sign2[tid] * inv_sqrt_d;
}

__global__ void tq_quantize_scatter_kernel(
    const float *__restrict__ x_rot, uint8_t *__restrict__ q_out, const int *__restrict__ indices,
    int head_dim, int n_heads, int bits)
{
    int tid  = threadIdx.x;
    int head = blockIdx.y;
    int tok  = blockIdx.x; 
    if (tid >= head_dim) return;

    int vec_src = tok * n_heads + head;
    int vec_dst = (indices ? indices[tok] : tok) * n_heads + head;

    float v = x_rot[vec_src * head_dim + tid];
    int q_idx = tq_scalar_quantize(v, bits);

    int byte_pos = (tid * bits) / 8;
    int bit_off  = (tid * bits) % 8;
    int bpv      = (head_dim * bits + 7) / 8;
    uint8_t *base = q_out + vec_dst * bpv;

    uint32_t mask = (1u << bits) - 1u;
    uint32_t val  = (uint32_t)q_idx & mask;

    uint32_t *aligned_ptr = (uint32_t *)((uintptr_t)(base + byte_pos) & ~3);
    uint32_t shift = (((uintptr_t)(base + byte_pos) & 3) * 8) + bit_off;

    atomicOr(aligned_ptr, val << shift);
    if (shift + bits > 32) {
        atomicOr(aligned_ptr + 1, val >> (32 - shift));
    }
}

__global__ void tq_attn_logits_kernel(
    const float *__restrict__ query, const uint8_t *__restrict__ keys_q,
    const float *__restrict__ sign1, const float *__restrict__ sign2,
    float *__restrict__ logits,
    int head_dim, int n_heads_q, int n_heads_kv, int n_tokens_kv, int n_queries, int key_bits,
    float inv_sqrt_d, float scale)
{
    __shared__ float q_rot[MAX_DIM];
    __shared__ float smem[MAX_DIM];

    int q_head = blockIdx.x;
    int q_tok  = blockIdx.z; 
    int tid    = threadIdx.x;
    if (tid >= head_dim) return;

    int gqa_ratio = n_heads_q / n_heads_kv;
    int kv_head   = q_head / gqa_ratio;

    int query_idx = (q_tok * n_heads_q + q_head) * head_dim + tid;
    float q_v = query[query_idx];
    __syncthreads();

    float val = q_v * sign1[tid];
    for (int s = 1; s < WARP_SIZE && s < head_dim; s <<= 1) {
        float o = __shfl_xor(val, s);
        val = ((tid & s) == 0) ? val + o : o - val;
    }
    if (head_dim > WARP_SIZE) {
        smem[tid] = val; __syncthreads();
        for (int s = WARP_SIZE; s < head_dim; s <<= 1) {
            int p = tid ^ s;
            float a = smem[tid], b = smem[p]; __syncthreads();
            if ((tid & s) == 0) { smem[tid] = a+b; smem[p] = a-b; } __syncthreads();
        }
        val = smem[tid];
    }
    q_rot[tid] = val * sign2[tid] * inv_sqrt_d;
    __syncthreads();

    const int k_bpv = (head_dim * key_bits + 7) / 8;
    int q_tok_abs = (n_tokens_kv - n_queries) + q_tok;

    for (int tok_kv = blockIdx.y; tok_kv < n_tokens_kv; tok_kv += gridDim.y) {
        if (tok_kv > q_tok_abs) {
            if (tid == 0) {
                int logit_idx = (q_tok * n_heads_q + q_head) * n_tokens_kv + tok_kv;
                logits[logit_idx] = -50000.0f; 
            }
            __syncthreads();
            continue;
        }

        int head_tok_base = tok_kv * n_heads_kv + kv_head;
        const uint8_t *kq = keys_q + head_tok_base * k_bpv;
        int bp = (tid * key_bits) / 8, bo = (tid * key_bits) % 8;
        uint32_t raw = (uint32_t)kq[bp];
        if (bo + key_bits > 8) raw |= (uint32_t)kq[bp+1] << 8;
        raw = (raw >> bo) & ((1u << key_bits) - 1u);

        float k_r = tq_scalar_reconstruct((int)raw, key_bits);
        float ip = q_rot[tid] * k_r;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            ip += __shfl_xor(ip, offset);

        __shared__ float warp_sums[MAX_DIM / 32];
        int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
        if (lane == 0) warp_sums[warp] = ip;
        __syncthreads();

        if (tid < (head_dim / WARP_SIZE)) {
            float w = warp_sums[tid];
            for (int o = (head_dim / WARP_SIZE) / 2; o > 0; o >>= 1) w += __shfl_xor(w, o);
            if (tid == 0) {
                int logit_idx = (q_tok * n_heads_q + q_head) * n_tokens_kv + tok_kv;
                logits[logit_idx] = w * scale;
            }
        }
        __syncthreads();
    }
}

__global__ void tq_attn_output_kernel(
    const float *__restrict__ logits, const uint8_t *__restrict__ vals_q,
    float *__restrict__ out, int head_dim, int n_heads_q, int n_heads_kv, int n_tokens_kv, int n_queries, int val_bits)
{
    int q_head = blockIdx.x;
    int q_tok  = blockIdx.z;
    int dim    = threadIdx.x; 

    if (q_head >= n_heads_q || dim >= head_dim) return;

    int gqa_ratio = n_heads_q / n_heads_kv;
    int kv_head   = q_head / gqa_ratio;

    const float *lgt = logits + (q_tok * n_heads_q + q_head) * n_tokens_kv;
    int q_tok_abs = (n_tokens_kv - n_queries) + q_tok;

    float mx = -50000.0f;
    for (int t = 0; t <= q_tok_abs && t < n_tokens_kv; t++) mx = fmaxf(mx, lgt[t]);
    
    float sum = 0.0f;
    for (int t = 0; t <= q_tok_abs && t < n_tokens_kv; t++) sum += expf(lgt[t] - mx);
    float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;

    int bpv = (head_dim * val_bits + 7) / 8;
    int bp  = (dim * val_bits) / 8;
    int bo  = (dim * val_bits) % 8;
    uint32_t mask = (1u << val_bits) - 1u;

    float acc = 0.0f;
    for (int tok_kv = 0; tok_kv <= q_tok_abs && tok_kv < n_tokens_kv; tok_kv++) {
        int head_tok = tok_kv * n_heads_kv + kv_head;
        const uint8_t *vq = vals_q + head_tok * bpv;
        uint32_t raw = (uint32_t)vq[bp] >> bo;
        if (bo + val_bits > 8) raw |= (uint32_t)vq[bp+1] << (8 - bo);
        raw &= mask;
        acc += (expf(lgt[tok_kv] - mx) * inv_sum) * tq_scalar_reconstruct((int)raw, val_bits);
    }
    out[(q_tok * n_heads_q + q_head) * head_dim + dim] = acc;
}

__global__ void tq_fp16_to_float_kernel(const half *__restrict__ src, float *__restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

extern "C" {

int tq_launch_rotate(float *d_x, const float *d_sign1, const float *d_sign2, int n_tokens, int n_heads, int head_dim, tq_hip_stream_t stream) {
    hipLaunchKernelGGL(tq_rotate_whd_kernel, dim3(n_tokens, n_heads), dim3(head_dim), 0, (hipStream_t)stream, d_x, d_sign1, d_sign2, head_dim, n_heads, 1.0f/sqrtf(head_dim));
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_quantize_scatter(const float *d_x_rot, uint8_t *d_q_out, const int *d_indices, int n_tokens, int n_heads, int head_dim, int bits, tq_hip_stream_t stream) {
    hipLaunchKernelGGL(tq_quantize_scatter_kernel, dim3(n_tokens, n_heads), dim3(head_dim), 0, (hipStream_t)stream, d_x_rot, d_q_out, d_indices, head_dim, n_heads, bits);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_attn_logits(const float *d_query, const uint8_t *d_keys_q, const float *d_sign1, const float *d_sign2, float *d_logits, int n_heads_q, int n_heads_kv, int head_dim, int n_tokens_kv, int n_queries, int key_bits, tq_hip_stream_t stream) {
    int blocks_y = (n_tokens_kv + 127) / 128;
    if (blocks_y > 256) blocks_y = 256;
    dim3 grid(n_heads_q, blocks_y, n_queries);
    dim3 block(head_dim);
    size_t smem = sizeof(float) * head_dim * 4;
    float inv = 1.0f / sqrtf((float)head_dim);
    hipLaunchKernelGGL(tq_attn_logits_kernel, grid, block, smem, (hipStream_t)stream, d_query, d_keys_q, d_sign1, d_sign2, d_logits, head_dim, n_heads_q, n_heads_kv, n_tokens_kv, n_queries, key_bits, inv, inv);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_attn_output(const float *d_logits, const uint8_t *d_vals_q, float *d_output, int n_heads_q, int n_heads_kv, int head_dim, int n_tokens_kv, int n_queries, int val_bits, tq_hip_stream_t stream) {
    dim3 grid(n_heads_q, 1, n_queries);
    dim3 block(head_dim);
    hipLaunchKernelGGL(tq_attn_output_kernel, grid, block, 0, (hipStream_t)stream, d_logits, d_vals_q, d_output, head_dim, n_heads_q, n_heads_kv, n_tokens_kv, n_queries, val_bits);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_fp16_to_float(const void *d_src, float *d_dst, int n, tq_hip_stream_t stream) {
    int block = 256;
    hipLaunchKernelGGL(tq_fp16_to_float_kernel, (n + block - 1) / block, block, 0, (hipStream_t)stream, (const half *)d_src, d_dst, n);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // extern "C"
