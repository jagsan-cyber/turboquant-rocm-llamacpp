#pragma once
#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <cstdint>
#include <cstddef>
#include <stdbool.h>

// Forward-declare hipStream_t without pulling in full HIP headers
typedef struct ihipStream_t *tq_hip_stream_t;

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(TQ_BUILD_DLL)
#    define TQ_API __declspec(dllexport)
#  else
#    define TQ_API __declspec(dllimport)
#  endif
#else
#  define TQ_API __attribute__((visibility("default")))
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

#define TQ_MAX_BITS      4
#define TQ_QJL_BITS      1
#define TQ_MAX_HEAD_DIM  256
#define TQ_SEED_ROTATION 0xDEADBEEFULL
#define TQ_SEED_QJL      0xCAFEBABEULL

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

typedef enum {
    TQ_BITS_2 = 2,
    TQ_BITS_3 = 3,
    TQ_BITS_4 = 4,
} tq_bits_t;

typedef struct {
    int       head_dim;
    int       n_heads_kv;
    int       max_seq_len;
    tq_bits_t key_bits;
    tq_bits_t val_bits;
    bool      use_qjl;
} tq_config_t;

/**
 * Quantized KV cache for one transformer layer.
 *
 * Memory layout: [token * n_heads + head] * bytes_per_vec
 *
 * FIX: Added d_tmp_* fields so tq_store_kv() does NOT hipMalloc every call.
 * These are allocated once in tq_kv_cache_alloc() and reused each decode step.
 */
typedef struct {
    // Quantized storage (persistent across tokens)
    uint8_t *d_keys_quant;
    uint8_t *d_keys_qjl;
    uint8_t *d_vals_quant;
    uint8_t *d_vals_qjl;

    // Rotation parameters (fixed, uploaded at alloc time)
    float   *d_sign_k1;
    float   *d_sign_k2;
    float   *d_qjl_proj_k;
    float   *d_sign_v1;
    float   *d_sign_v2;
    float   *d_qjl_proj_v;

    // --- FIX: Per-token scratch buffers (allocated once, reused every decode) ---
    float   *d_tmp_kf;       // fp16 → float conversion output for keys
    float   *d_tmp_vf;       // fp16 → float conversion output for values
    float   *d_tmp_kf_orig;  // copy before rotation (for QJL residual)
    float   *d_tmp_vf_orig;
    float   *d_tmp_recon_k;  // Stage-1 reconstruction (for QJL)
    float   *d_tmp_recon_v;

    tq_config_t cfg;
    int         n_tokens;    // number of tokens currently stored
} tq_kv_cache_t;

typedef struct {
    float levels[1 << TQ_MAX_BITS];
    float thresholds[(1 << TQ_MAX_BITS) + 1];
    int   n_levels;
} tq_codebook_t;

// ─────────────────────────────────────────────────────────────────────────────
// Core API
// ─────────────────────────────────────────────────────────────────────────────

/** Initialize global codebooks. Call once before any other tq_* function. */
TQ_API int  tq_init(void);
TQ_API int  tq_kv_cache_alloc(tq_kv_cache_t *cache, const tq_config_t *cfg);
TQ_API void tq_kv_cache_free(tq_kv_cache_t *cache);

TQ_API int  tq_store_k(tq_kv_cache_t *cache, const void *d_keys_f32, const int *d_indices, int n_tokens);
TQ_API int  tq_store_v(tq_kv_cache_t *cache, const void *d_vals_f32, const int *d_indices, int n_tokens);

/**
 * Compute attention logits against all stored keys.
 *
 * d_query:  float device pointer [n_heads_kv, head_dim]
 * d_logits: fp32 output         [n_heads_kv, n_tokens]
 */
TQ_API int  tq_attn_logits(const tq_kv_cache_t *cache,
                    const float         *d_query,
                    float               *d_logits,
                    int n_queries, int n_heads_q);

TQ_API int  tq_attn_output(const tq_kv_cache_t *cache,
                    const float         *d_logits,
                    float               *d_output,
                    int n_queries, int n_heads_q);

/** Return Lloyd-Max codebook for given bit-width. */
TQ_API const tq_codebook_t *tq_get_codebook(tq_bits_t bits);

/** Bytes consumed per token per layer (for memory budgeting). */
TQ_API size_t tq_bytes_per_token(const tq_config_t *cfg);

// ─────────────────────────────────────────────────────────────────────────────
// Low-level Kernel Launchers (exposed for testing / advanced use)
// ─────────────────────────────────────────────────────────────────────────────

TQ_API int tq_launch_rotate(float *d_x, const float *d_sign1, const float *d_sign2,
                     int n_tokens, int n_heads, int head_dim, tq_hip_stream_t stream);

TQ_API int tq_launch_quantize_scatter(const float *d_x_rot, uint8_t *d_q_out, float *d_recon,
                       const int *d_indices,
                       int n_tokens, int n_heads, int head_dim, int bits,
                       tq_hip_stream_t stream);

TQ_API int tq_launch_qjl_scatter(const float *d_x_orig, const float *d_recon_rot,
                  const float *d_sign1, const float *d_sign2,
                  const float *d_qjl_proj, uint8_t *d_qjl_out,
                  const int *d_indices,
                  int n_tokens, int n_heads, int head_dim,
                  tq_hip_stream_t stream);

TQ_API int tq_launch_attn_logits(const float *d_query,
                          const uint8_t *d_keys_q, const uint8_t *d_keys_qjl,
                          const float *d_sign1, const float *d_sign2,
                          const float *d_qjl_proj, float *d_logits,
                          int n_heads_q, int n_heads_kv, int head_dim, int n_tokens_kv, int n_queries, int key_bits,
                          tq_hip_stream_t stream);

TQ_API int tq_launch_attn_output(const float *d_logits,
                          const uint8_t *d_vals_q, const uint8_t *d_vals_qjl,
                          float *d_output,
                          int n_heads_q, int n_heads_kv, int head_dim, int n_tokens_kv, int n_queries, int val_bits,
                          tq_hip_stream_t stream);

TQ_API int tq_launch_fp16_to_float(const void *d_src, float *d_dst, int n,
                             tq_hip_stream_t stream);

// ─────────────────────────────────────────────────────────────────────────────
// llama.cpp Integration API  (implemented in llama_tq_hook.cpp)
// ─────────────────────────────────────────────────────────────────────────────

struct llama_tq_context;

TQ_API struct llama_tq_context *llama_tq_create(int n_layers, int n_heads_kv,
                                         int head_dim, int max_seq_len,
                                         int key_bits, int val_bits);
TQ_API void llama_tq_free  (struct llama_tq_context *ctx);
TQ_API void llama_tq_reset (struct llama_tq_context *ctx);

TQ_API void llama_tq_set_global(struct llama_tq_context *ctx);
TQ_API struct llama_tq_context *llama_tq_get_global(void);

TQ_API int  llama_tq_store_k(struct llama_tq_context *ctx, int layer, int start_token, const void *d_keys_f32, int n_tokens);
TQ_API int  llama_tq_store_v(struct llama_tq_context *ctx, int layer, int start_token, const void *d_vals_f32, int n_tokens);

TQ_API int  llama_tq_attn  (struct llama_tq_context *ctx, int layer,
                     const float *d_query, float *d_logits_out, float *d_output,
                     int n_heads_q, int n_queries);

TQ_API void llama_tq_print_stats(const struct llama_tq_context *ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TURBOQUANT_H
// ^^^^ FIX: All declarations are inside the header guard. ^^^^
