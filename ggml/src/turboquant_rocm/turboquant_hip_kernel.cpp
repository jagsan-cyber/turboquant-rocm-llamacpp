/**
 * turboquant_hip.hip
 * ROCm / HIP implementation of TurboQuant kernels
 *
 * Build:
 *   hipcc -O3 -arch=gfx90a turboquant_hip.hip turboquant.cpp -o turboquant.o
 *
 * Kernels:
 *   1. tq_rotate_whd_kernel    EWalsh-Hadamard + sign-flip random rotation
 *   2. tq_quantize_kernel      ELloyd-Max scalar quantization (per coordinate)
 *   3. tq_qjl_kernel           E1-bit QJL residual sign projection
 *   4. tq_attn_logits_kernel   Efused dequant + inner product (keys ÁEquery)
 *   5. tq_attn_output_kernel   Esoftmax + weighted sum (values)
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
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers / Macros
// ─────────────────────────────────────────────────────────────────────────────

#define HIP_CHECK(cmd) \
    do { \
        hipError_t e = (cmd); \
        if (e != hipSuccess) { \
            fprintf(stderr, "[TurboQuant] HIP error %s at %s:%d\n", \
                    hipGetErrorString(e), __FILE__, __LINE__); \
            return -1; \
        } \
    } while(0)

// RDNA4 (gfx1201 / RX 9070): wavefront size = 32
// RDNA2/3 (gfx1030/1100):    wavefront size = 32
// GCN / CDNA (gfx90a etc.):  wavefront size = 64
// ↁEAt runtime we branch on __AMDGCN_WAVEFRONT_SIZE__ or use hipDeviceProp_t
#ifdef __gfx1201__
#  define WARP_SIZE  32   // RDNA4: RX 9070 / R9700
#elif defined(__gfx1200__)
#  define WARP_SIZE  32   // RDNA4: RX 9060 XT
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1030__)
#  define WARP_SIZE  32   // RDNA2/3
#else
#  define WARP_SIZE  64   // CDNA / GCN fallback
#endif

#define MAX_DIM      256

// ─────────────────────────────────────────────────────────────────────────────
// Device-side codebook (loaded from host constants, read-only)
// ─────────────────────────────────────────────────────────────────────────────

// Lloyd-Max codebooks for Beta(0.5, 0.5) distribution
// Generated offline via Lloyd's algorithm; coordinates after rotation
// are approximately arcsine-distributed on [-1, 1].

__constant__ float d_cb2_levels[4] = {
    -0.7071f, -0.2357f,  0.2357f,  0.7071f
};
__constant__ float d_cb2_thresh[5] = {
    -1.0f, -0.5f, 0.0f, 0.5f, 1.0f
};

__constant__ float d_cb3_levels[8] = {
    -0.9009f, -0.6235f, -0.3612f, -0.1205f,
     0.1205f,  0.3612f,  0.6235f,  0.9009f
};
__constant__ float d_cb3_thresh[9] = {
    -1.0f, -0.7654f, -0.5000f, -0.2393f, 0.0f,
     0.2393f, 0.5000f, 0.7654f, 1.0f
};

__constant__ float d_cb4_levels[16] = {
    -0.9613f, -0.8315f, -0.6895f, -0.5345f,
    -0.3681f, -0.1951f, -0.0980f,  0.0000f,
     0.0980f,  0.1951f,  0.3681f,  0.5345f,
     0.6895f,  0.8315f,  0.9613f,  0.9903f
};
__constant__ float d_cb4_thresh[17] = {
    -1.0f,    -0.9009f, -0.7654f, -0.6124f,
    -0.4540f, -0.2820f, -0.1464f, -0.0490f,
     0.0490f,  0.1464f,  0.2820f,  0.4540f,
     0.6124f,  0.7654f,  0.9009f,  0.9808f, 1.0f
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Walsh-Hadamard Random Rotation
// ─────────────────────────────────────────────────────────────────────────────
/**
 * In-place randomized Walsh-Hadamard transform:
 *   x = D2 * WHT(D1 * x) / sqrt(d)
 *
 * D1, D2: random ±1 diagonal matrices (d_sign1, d_sign2)
 * WHT: Walsh-Hadamard transform (butterfly network, O(d log d))
 *
 * One block handles one (token, head) pair.
 * blockDim.x = head_dim (must be power of 2, ≤ 256)
 */
__global__ void tq_rotate_whd_kernel(
    float       *__restrict__ x,       // [n_tokens, n_heads, head_dim]
    const float *__restrict__ sign1,   // [head_dim]
    const float *__restrict__ sign2,   // [head_dim]
    int head_dim,
    int n_heads,
    float inv_sqrt_d)
{
    __shared__ float smem[MAX_DIM];

    int tid  = threadIdx.x;
    int head = blockIdx.y;
    int tok  = blockIdx.x;

    if (tid >= head_dim) return;

    int base = (tok * n_heads + head) * head_dim;

    // Step 1: Apply D1 (sign flip)
    float val = x[base + tid] * sign1[tid];

    // Step 2: Walsh-Hadamard butterfly
    // Use fast wavefront shuffles for small strides
    for (int stride = 1; stride < WARP_SIZE && stride < head_dim; stride <<= 1) {
        float other = __shfl_xor(val, stride);
        if ((tid & stride) == 0) val = val + other;
        else                     val = other - val;
    }

    // Use shared memory for strides >= WARP_SIZE
    if (head_dim > WARP_SIZE) {
        smem[tid] = val;
        __syncthreads();

        for (int stride = WARP_SIZE; stride < head_dim; stride <<= 1) {
            int pair = tid ^ stride;
            float a = smem[tid];
            float b = smem[pair];
            __syncthreads();
            if ((tid & stride) == 0) {
                smem[tid]  = a + b;
                smem[pair] = a - b;
            }
            __syncthreads();
        }
        val = smem[tid];
    }

    // Step 3: Apply D2 and normalize
    x[base + tid] = val * sign2[tid] * inv_sqrt_d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Lloyd-Max Scalar Quantization
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Quantize rotated float coordinates to b-bit indices using the
 * Lloyd-Max codebook for the Beta(0.5, 0.5) / arcsine distribution.
 *
 * Also stores the reconstruction (for residual computation in QJL).
 *
 * Output packing: bits are packed MSB-first into uint8 arrays.
 */
__device__ __forceinline__
int tq_scalar_quantize(float v, int bits) {
    // Clamp to [-1, 1] (rotation preserves L2 norm ~ unit sphere)
    v = fmaxf(-1.0f, fminf(1.0f, v));

    if (bits == 2) {
        if (v < d_cb2_thresh[1]) return 0;
        if (v < d_cb2_thresh[2]) return 1;
        if (v < d_cb2_thresh[3]) return 2;
        return 3;
    } else if (bits == 3) {
        // Binary search through 8 levels
        int lo = 0, hi = 7;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (v < d_cb3_thresh[mid + 1]) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    } else {  // bits == 4
        int lo = 0, hi = 15;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (v < d_cb4_thresh[mid + 1]) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }
}

__device__ __forceinline__
float tq_scalar_reconstruct(int idx, int bits) {
    if (bits == 2) return d_cb2_levels[idx];
    if (bits == 3) return d_cb3_levels[idx];
    return d_cb4_levels[idx];
}

/**
 * Quantize kernel: reads float, writes packed bits + reconstruction.
 *
 * Each thread handles one coordinate.
 * Pack b bits per coordinate into uint8 stream (b * head_dim / 8 bytes per vector).
 */
__global__ void tq_quantize_kernel(
    const float   *__restrict__ x_rot,    // [n_tokens, n_heads, head_dim] rotated
    uint8_t       *__restrict__ q_out,    // packed quantized output
    float         *__restrict__ recon,   // reconstruction for QJL residual
    int head_dim,
    int n_heads,
    int bits)
{
    int tid  = threadIdx.x;   // coordinate index within head
    int head = blockIdx.y;
    int tok  = blockIdx.x;

    if (tid >= head_dim) return;

    int vec_idx = tok * n_heads + head;
    float v = x_rot[vec_idx * head_dim + tid];

    int q_idx = tq_scalar_quantize(v, bits);
    float r   = tq_scalar_reconstruct(q_idx, bits);

    // Store reconstruction for QJL residual calculation
    if (recon) recon[vec_idx * head_dim + tid] = r;

    // Pack bits: head_dim * bits packed into bytes
    // Thread tid writes `bits` bits at bit position (tid * bits)
    int   byte_pos = (tid * bits) / 8;
    int   bit_off  = (tid * bits) % 8;
    int   bytes_per_vec = (head_dim * bits + 7) / 8;
    uint8_t *base = q_out + vec_idx * bytes_per_vec;

    // Write bits (atomic for threads sharing a byte)
    uint32_t mask = ((1u << bits) - 1u);
    uint32_t val  = (uint32_t)q_idx & mask;

    // Use atomicOr for thread-safe packing
    if (bits <= 8) {
        atomicOr((unsigned int *)(base + byte_pos),
                 (unsigned int)(val << bit_off));
        if (bit_off + bits > 8) {
            // Spans two bytes
            int overflow = bit_off + bits - 8;
            atomicOr((unsigned int *)(base + byte_pos + 1),
                     (unsigned int)(val >> (bits - overflow)));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Quantized Johnson-Lindenstrauss (QJL) Residual
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Compute 1-bit QJL projection of residual vector.
 *
 * residual = x_original - R^T(reconstruction)
 * qjl_sign[j] = sign( sum_i S[j,i] * residual[i] )
 *
 * S is a random Gaussian matrix (d_qjl_proj, shape [head_dim, head_dim]).
 * Output: 1 bit per coordinate packed into uint8.
 *
 * One block per (token, head), using reduction in shared memory.
 * blockDim.x = head_dim (projection dimension)
 * Each thread computes one projection dot product via __shfl_down_reduce.
 */
__global__ void tq_qjl_kernel(
    const float   *__restrict__ x_orig,    // original (pre-rotation) vectors
    const float   *__restrict__ recon_rot, // reconstruction in rotated space
    const float   *__restrict__ sign1,     // D1 for inverse rotation
    const float   *__restrict__ sign2,     // D2 for inverse rotation
    const float   *__restrict__ qjl_proj,  // random Gaussian [head_dim x head_dim]
    uint8_t       *__restrict__ qjl_out,   // packed sign bits output
    int head_dim,
    int n_heads,
    float inv_sqrt_d)
{
    __shared__ float smem[MAX_DIM];

    int tid  = threadIdx.x;
    int head = blockIdx.y;
    int tok  = blockIdx.x;
    int vec  = tok * n_heads + head;

    if (tid >= head_dim) return;

    // Step 1: Inverse-rotate reconstruction ↁEoriginal space
    // R^T = D1 * WHT(D2 * ·) * inv_sqrt_d  (WHT is self-inverse up to scale)
    smem[tid] = recon_rot[vec * head_dim + tid] * sign2[tid];
    __syncthreads();

    for (int stride = 1; stride < head_dim; stride <<= 1) {
        int pair = tid ^ stride;
        float a = smem[tid], b = smem[pair];
        __syncthreads();
        if ((tid & stride) == 0) {
            smem[tid]  = a + b;
            smem[pair] = a - b;
        }
        __syncthreads();
    }
    float recon_orig = smem[tid] * sign1[tid] * inv_sqrt_d;

    // Step 2: Residual
    float residual = x_orig[vec * head_dim + tid] - recon_orig;

    // Step 3: One QJL projection per thread (thread j computes projection j)
    // proj_j = sum_i S[j,i] * residual[i]
    // Each thread: store residual in smem, then each thread reads row j of S
    smem[tid] = residual;
    __syncthreads();

    // Thread `tid` computes projection index `tid`
    float proj = 0.0f;
    const float *S_row = qjl_proj + tid * head_dim;
    for (int i = 0; i < head_dim; i++) {
        proj += S_row[i] * smem[i];
    }

    // Step 4: Sign bit
    int   qjl_sign = (proj >= 0.0f) ? 1 : 0;
    int   bytes_per_vec = (head_dim + 7) / 8;
    uint8_t *base = qjl_out + vec * bytes_per_vec;
    atomicOr((unsigned int *)(base + tid / 8),
             (unsigned int)(qjl_sign << (tid % 8)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: Fused Dequantize + Attention Logits
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Compute attention logits: logit[tok] = dot(query, key_tok)
 *
 * For each stored token, reconstruct approximate key in rotated space,
 * then apply QJL correction for unbiased inner product estimate.
 *
 * logit = <R*q, recon(k_rot)> + QJL_correction(q, k_residual)
 *
 * Grid: [n_heads] blocks, each of blockDim.x = 32 warps ÁEthreads
 * Each block processes all tokens sequentially (or parallelized over tokens).
 */
__global__ void tq_attn_logits_kernel(
    const half    *__restrict__ query,     // [n_heads, head_dim] fp16
    const uint8_t *__restrict__ keys_q,   // packed quantized keys
    const uint8_t *__restrict__ keys_qjl, // QJL sign bits for keys
    const float   *__restrict__ sign1,    // D1 rotation signs
    const float   *__restrict__ sign2,    // D2 rotation signs
    const float   *__restrict__ qjl_proj, // QJL projection matrix [head_dim x head_dim]
    float         *__restrict__ logits,   // output [n_heads, n_tokens]
    int head_dim,
    int n_heads,
    int n_tokens,
    int key_bits,
    float inv_sqrt_d,
    float scale)                           // 1/sqrt(head_dim) attention scale
{
    __shared__ float q_rot[MAX_DIM];   // rotated query (shared across token loop)
    __shared__ float q_orig[MAX_DIM];  // original query (for QJL)
    __shared__ float smem[MAX_DIM];

    int head = blockIdx.x;
    int tid  = threadIdx.x;

    if (tid >= head_dim) return;

    // ── Rotate query ──────────────────────────────────────────────────────────
    // Load query from fp16
    q_orig[tid] = __half2float(query[head * head_dim + tid]);
    __syncthreads();

    // Apply D1 + WHT + D2
    smem[tid] = q_orig[tid] * sign1[tid];
    __syncthreads();

    for (int stride = 1; stride < head_dim; stride <<= 1) {
        int pair = tid ^ stride;
        float a = smem[tid], b = smem[pair];
        __syncthreads();
        if ((tid & stride) == 0) { smem[tid] = a + b; smem[pair] = a - b; }
        __syncthreads();
    }
    q_rot[tid] = smem[tid] * sign2[tid] * inv_sqrt_d;
    __syncthreads();

    // ── Pre-compute QJL query projections: Sq = S * q_orig ───────────────────
    // Thread tid computes (S * q_orig)[tid]
    float sq = 0.0f;
    const float *S_row = qjl_proj + tid * head_dim;
    for (int i = 0; i < head_dim; i++) {
        sq += S_row[i] * q_orig[i];
    }
    // sq is the tid-th element of S*q; store for QJL dot product
    smem[tid] = sq;
    __syncthreads();
    // Note: We'll use smem[] as S*q throughout token loop below.

    int bytes_per_key = (head_dim * key_bits + 7) / 8;
    int qjl_bytes     = (head_dim + 7) / 8;
    float qjl_scale = (float)(3.1415926535 / 2.0) / (float)head_dim;

    // ── Token loop ───────────────────────────────────────────────────────────
    // Parallelize across tokens by using blockIdx.y
    int tokens_per_block = 1; // Simplified for now, but blockIdx.y allows scaling
    for (int tok = blockIdx.y; tok < n_tokens; tok += gridDim.y) {
        int head_tok_base = (tok * n_heads + head);

        // Dequantize key coordinate `tid` from packed bits
        const uint8_t *kq_base = keys_q + head_tok_base * bytes_per_key;
        int   bit_pos   = tid * key_bits;
        int   byte_pos  = bit_pos / 8;
        int   bit_off   = bit_pos % 8;
        
        // Fast bit-unpacking
        uint32_t val_bits_raw = (uint32_t)kq_base[byte_pos];
        if (bit_off + key_bits > 8) val_bits_raw |= ((uint32_t)kq_base[byte_pos + 1]) << 8;
        uint32_t raw = (val_bits_raw >> bit_off) & ((1u << key_bits) - 1u);

        float k_recon_rot = tq_scalar_reconstruct((int)raw, key_bits);

        // Stage-1 inner product component: q_rot[tid] * k_recon_rot[tid]
        float ip1 = q_rot[tid] * k_recon_rot;

        // QJL stage-2: <Sq, sign(S*k_residual)>
        const uint8_t *qjl_base = keys_qjl + head_tok_base * qjl_bytes;
        int   sign_bit  = (qjl_base[tid / 8] >> (tid % 8)) & 1;
        float k_qjl_sgn = sign_bit ? 1.0f : -1.0f;

        // QJL correction: (pi/2) * <Sq, sign(Sk)> * sq_norm_inv
        float ip2 = smem[tid] * k_qjl_sgn * qjl_scale;

        float ip = ip1 + ip2;

        // Fast block-level reduction
        // 1. Warp-level reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            ip += __shfl_xor(ip, offset);

        // 2. Shared memory across warps
        __shared__ float block_reduce[MAX_DIM / 32];
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) block_reduce[warp_id] = ip;
        __syncthreads();

        if (tid < (head_dim / WARP_SIZE)) {
            float final_val = block_reduce[tid];
            for (int offset = (head_dim / WARP_SIZE) / 2; offset > 0; offset >>= 1)
                final_val += __shfl_xor(final_val, offset);
            
            if (tid == 0) {
                logits[head * n_tokens + tok] = final_val * scale;
            }
        }
        __syncthreads();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: Softmax + Weighted Value Sum (Attention Output)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Compute attention output:
 *   weights = softmax(logits)
 *   out[d]  = sum_tok weights[tok] * dequant(val[tok, d])
 *
 * Grid: [n_heads, head_dim] blocks, each a single thread
 * (Simple reference implementation  Eoptimize with tiling for production)
 */
__global__ void tq_attn_output_kernel(
    const float   *__restrict__ logits,    // [n_heads, n_tokens]
    const uint8_t *__restrict__ vals_q,    // packed quantized values
    const uint8_t *__restrict__ vals_qjl,  // QJL sign bits for values (unused in output)
    half          *__restrict__ out,       // [n_heads, head_dim] fp16 output
    int head_dim,
    int n_heads,
    int n_tokens,
    int val_bits)
{
    int head = blockIdx.x;
    int dim  = blockIdx.y;    // coordinate index

    if (head >= n_heads || dim >= head_dim) return;

    int bytes_per_val = (head_dim * val_bits + 7) / 8;

    // ── Softmax (one thread computes full softmax for its head) ───────────────
    // Note: each block only handles one `dim`, so we need the weights.
    // For efficiency, compute softmax in shared memory once per head.
    // Here we compute it per thread (redundant but simple; optimize with shared mem).
    const float *lgt = logits + head * n_tokens;
    float max_l = lgt[0];
    for (int t = 1; t < n_tokens; t++) max_l = fmaxf(max_l, lgt[t]);

    float sum_exp = 0.0f;
    for (int t = 0; t < n_tokens; t++) sum_exp += expf(lgt[t] - max_l);

    // ── Weighted sum of dequantized values ────────────────────────────────────
    int   bit_pos  = dim * val_bits;
    int   byte_pos = bit_pos / 8;
    int   bit_off  = bit_pos % 8;
    uint32_t mask  = (1u << val_bits) - 1u;

    float acc = 0.0f;
    for (int tok = 0; tok < n_tokens; tok++) {
        int   head_tok = tok * n_heads + head;
        const uint8_t *vq = vals_q + head_tok * bytes_per_val;

        uint32_t raw = (uint32_t)vq[byte_pos] >> bit_off;
        if (bit_off + val_bits > 8)
            raw |= (uint32_t)vq[byte_pos + 1] << (8 - bit_off);
        raw &= mask;

        float v_recon = tq_scalar_reconstruct((int)raw, val_bits);
        float weight  = expf(lgt[tok] - max_l) / sum_exp;
        acc += weight * v_recon;
    }

    out[head * head_dim + dim] = __float2half(acc);
}

// ─────────────────────────────────────────────────────────────────────────────
// fp16 ↁEfloat conversion kernel (for rotation input)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void tq_fp16_to_float_kernel(
    const half *__restrict__ src,
    float      *__restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher functions (called from turboquant.cpp)
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

int tq_launch_rotate(
    float *d_x, const float *d_sign1, const float *d_sign2,
    int n_tokens, int n_heads, int head_dim, tq_hip_stream_t stream)
{
    if ((head_dim & (head_dim - 1)) != 0 || head_dim > MAX_DIM) {
        fprintf(stderr, "[TurboQuant] head_dim must be power of 2 and ≤ %d\n", MAX_DIM);
        return -1;
    }
    dim3 grid(n_tokens, n_heads);
    dim3 block(head_dim);
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    hipLaunchKernelGGL(tq_rotate_whd_kernel, grid, block, 0, (hipStream_t)stream,
        d_x, d_sign1, d_sign2, head_dim, n_heads, inv_sqrt_d);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_quantize(
    const float *d_x_rot, uint8_t *d_q_out, float *d_recon,
    int n_tokens, int n_heads, int head_dim, int bits, tq_hip_stream_t stream)
{
    dim3 grid(n_tokens, n_heads);
    dim3 block(head_dim);
    hipLaunchKernelGGL(tq_quantize_kernel, grid, block, 0, (hipStream_t)stream,
        d_x_rot, d_q_out, d_recon, head_dim, n_heads, bits);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_qjl(
    const float *d_x_orig, const float *d_recon_rot,
    const float *d_sign1, const float *d_sign2, const float *d_qjl_proj,
    uint8_t *d_qjl_out,
    int n_tokens, int n_heads, int head_dim, tq_hip_stream_t stream)
{
    dim3 grid(n_tokens, n_heads);
    dim3 block(head_dim);
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    hipLaunchKernelGGL(tq_qjl_kernel, grid, block, 0, (hipStream_t)stream,
        d_x_orig, d_recon_rot, d_sign1, d_sign2, d_qjl_proj,
        d_qjl_out, head_dim, n_heads, inv_sqrt_d);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_attn_logits(
    const void *d_query, const uint8_t *d_keys_q, const uint8_t *d_keys_qjl,
    const float *d_sign1, const float *d_sign2, const float *d_qjl_proj,
    float *d_logits,
    int n_heads, int head_dim, int n_tokens, int key_bits, tq_hip_stream_t stream)
{
    // Use 2D grid: x for heads, y for tokens
    // We launch enough blocks in Y to saturate the GPU (e.g., 64-128 blocks)
    int blocks_y = (n_tokens + 127) / 128; 
    if (blocks_y > 256) blocks_y = 256; // Cap to avoid excessive overhead

    dim3 grid(n_heads, blocks_y);
    dim3 block(head_dim);
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    hipLaunchKernelGGL(tq_attn_logits_kernel, grid, block,
        head_dim * sizeof(float) * 4, (hipStream_t)stream,
        (const half*)d_query, d_keys_q, d_keys_qjl,
        d_sign1, d_sign2, d_qjl_proj,
        d_logits, head_dim, n_heads, n_tokens, key_bits,
        inv_sqrt_d, attn_scale);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_attn_output(
    const float *d_logits, const uint8_t *d_vals_q, const uint8_t *d_vals_qjl,
    void *d_output,
    int n_heads, int head_dim, int n_tokens, int val_bits, tq_hip_stream_t stream)
{
    dim3 grid(n_heads, head_dim);
    dim3 block(1);
    hipLaunchKernelGGL(tq_attn_output_kernel, grid, block, 0, (hipStream_t)stream,
        d_logits, d_vals_q, d_vals_qjl, (half*)d_output,
        head_dim, n_heads, n_tokens, val_bits);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

int tq_launch_fp16_to_float(
    const void *d_src, float *d_dst, int n, tq_hip_stream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    hipLaunchKernelGGL(tq_fp16_to_float_kernel, grid, block, 0, (hipStream_t)stream,
        (const half*)d_src, d_dst, n);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // extern "C"

