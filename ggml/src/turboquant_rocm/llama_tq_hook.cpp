/**
 * llama_tq_hook.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Correct integration of TurboQuant into llama.cpp's decode loop.
 *
 * HOW THIS WORKS
 * ──────────────
 * This file is NOT a patch to ggml-cuda.cu.  It is a higher-level layer that
 * hooks into llama.cpp's context creation and per-token decode path.
 *
 * The integration points in llama.cpp (src/llama.cpp) are:
 *
 *   1. llama_new_context_with_model()
 *      → after the kv_cache is set up, call llama_tq_create() and store
 *        the pointer in the llama_context.
 *
 *   2. llama_decode_impl()  (the inner decode loop, one token at a time)
 *      → after the normal attention graph writes keys/values into the kv_cache,
 *        intercept with tq_store_kv() on the raw tensors.
 *      → replace the attention kernel invocation with tq_attn_logits() +
 *        tq_attn_output().
 *
 * MINIMAL DIFF TO src/llama.cpp
 * ──────────────────────────────
 *
 *  // 1. Add to llama_context struct  (llama-context.h or top of llama.cpp)
 *  #ifdef LLAMA_TURBOQUANT
 *    struct llama_tq_context * tq_ctx = nullptr;
 *  #endif
 *
 *  // 2. In llama_new_context_with_model(), after kv cache init:
 *  #ifdef LLAMA_TURBOQUANT
 *    if (params.use_turboquant) {
 *        ctx->tq_ctx = llama_tq_create(
 *            hparams.n_layer,
 *            hparams.n_head_kv,
 *            hparams.n_embd_head_k,
 *            params.n_ctx,
 *            3,  // key bits
 *            2); // val bits
 *    }
 *  #endif
 *
 *  // 3. In llama_decode_impl(), after QKV projection tensors are computed:
 *  #ifdef LLAMA_TURBOQUANT
 *    if (lctx.tq_ctx) {
 *        // Store this token's KV pair
 *        llama_tq_store(lctx.tq_ctx, il, kv_head,
 *                       (const void*)Kcur->data,   // fp16 keys
 *                       (const void*)Vcur->data);  // fp16 values
 *        // Compute attention using TQ (replaces standard MHA kernel)
 *        float *d_logits = ...; // temp fp32 [n_heads, n_ctx]
 *        llama_tq_attn(lctx.tq_ctx, il,
 *                      (const void*)Qcur->data,
 *                      d_logits,
 *                      (void*)cur->data);
 *        // Skip the standard ggml attention ops for this layer
 *        continue;
 *    }
 *  #endif
 *
 * This is the ONLY way to correctly integrate TurboQuant.
 * Patching ggml-cuda.cu does not work because:
 *   - ggml-cuda.cu has no access to llama_context (wrong abstraction level)
 *   - Flash attention bypasses mul_mat entirely
 *   - ABI between llama.cpp versions changes frequently
 */

#ifdef LLAMA_TURBOQUANT

#include "turboquant.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Internal context
// ─────────────────────────────────────────────────────────────────────────────

struct llama_tq_context {
    std::vector<tq_kv_cache_t> layers;
    tq_config_t                cfg;
    float                     *d_logits = nullptr;  // [n_heads, max_seq_len]
};

// ─────────────────────────────────────────────────────────────────────────────
// Global context for hooking
// ─────────────────────────────────────────────────────────────────────────────

static struct llama_tq_context * g_tq_ctx = nullptr;

extern "C"
void llama_tq_set_global(struct llama_tq_context *ctx) {
    g_tq_ctx = ctx;
}

extern "C"
struct llama_tq_context *llama_tq_get_global(void) {
    return g_tq_ctx;
}

extern "C"
struct llama_tq_context *llama_tq_create(int n_layers, int n_heads_kv,
                                         int head_dim, int max_seq_len,
                                         int key_bits, int val_bits) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    if (tq_init() != 0) return nullptr;

    auto *ctx = new llama_tq_context();
    ctx->cfg  = tq_config_t{
        head_dim, n_heads_kv, max_seq_len,
        (tq_bits_t)key_bits,
        (tq_bits_t)val_bits,
        /*use_qjl=*/false  // QJLを永久に無効化し、WHTのみで最高の知能を維持
    };

    ctx->layers.resize(n_layers);
    for (int l = 0; l < n_layers; l++) {
        if (tq_kv_cache_alloc(&ctx->layers[l], &ctx->cfg) != 0) {
            llama_tq_free(ctx);
            return nullptr;
        }
    }

    size_t logits_sz = (size_t)n_heads_kv * max_seq_len * sizeof(float);
    if (hipMalloc(&ctx->d_logits, logits_sz) != hipSuccess) {
        llama_tq_free(ctx);
        return nullptr;
    }

    float savings = 32.0f / (key_bits + val_bits + 2.0f); // vs fp32
    fprintf(stderr,
        "[TQ] Context ready: %d layers × %d KV-heads × dim=%d  "
        "keys=%dbit vals=%dbit  ~%.1fx vs fp32\n",
        n_layers, n_heads_kv, head_dim, key_bits, val_bits, savings);
    return ctx;
}

extern "C"
void llama_tq_free(struct llama_tq_context *ctx) {
    if (!ctx) return;
    for (auto &l : ctx->layers) tq_kv_cache_free(&l);
    if (ctx->d_logits) hipFree(ctx->d_logits);
    delete ctx;
}

extern "C"
void llama_tq_reset(struct llama_tq_context *ctx) {
    if (!ctx) return;
    for (auto &l : ctx->layers) {
        l.n_tokens = 0;
        const tq_config_t &c = l.cfg;
        int k_bpv   = (c.head_dim * c.key_bits + 7) / 8;
        int v_bpv   = (c.head_dim * c.val_bits + 7) / 8;
        int qjl_bpv = (c.head_dim + 7) / 8;
        size_t H    = c.n_heads_kv;
        size_t T    = c.max_seq_len;
        hipMemset(l.d_keys_quant, 0, T * H * k_bpv);
        hipMemset(l.d_vals_quant, 0, T * H * v_bpv);
        hipMemset(l.d_keys_qjl,  0, T * H * qjl_bpv);
        hipMemset(l.d_vals_qjl,  0, T * H * qjl_bpv);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-token decode operations
// ─────────────────────────────────────────────────────────────────────────────

extern "C" int llama_tq_store_k(struct llama_tq_context *ctx, int layer, int start_token, const void *d_keys, int n_tokens, const int *d_indices) {
    if (!ctx || layer < 0 || layer >= (int)ctx->layers.size()) return -1;
    auto &l = ctx->layers[layer];
    // n_tokens の更新ロジックをより堅牢に (start_token ではなく、保存されるトークン数に基づき、llama_tq_attn 側で正確に同期されることを期待)
    return tq_store_k(&l, d_keys, d_indices, n_tokens);
}

extern "C" int llama_tq_store_v(struct llama_tq_context *ctx, int layer, int start_token, const void *d_vals, int n_tokens, const int *d_indices) {
    if (!ctx || layer < 0 || layer >= (int)ctx->layers.size()) return -1;
    auto &l = ctx->layers[layer];
    return tq_store_v(&l, d_vals, d_indices, n_tokens);
}

extern "C"
int llama_tq_attn(struct llama_tq_context *ctx, int layer,
                  const float *d_query, float *d_logits_out, float *d_output,
                  int n_heads_q, int n_queries, int n_tokens_kv) {
    if (!ctx || layer < 0 || layer >= (int)ctx->layers.size()) return -1;

    tq_kv_cache_t *cache = &ctx->layers[layer];
    cache->n_tokens = n_tokens_kv; // アテンション時に最新のトークン数を反映

    fprintf(stderr, "[TQ HOOK] L%d | heads_q=%d, queries=%d, total_kv=%d\n", layer, n_heads_q, n_queries, n_tokens_kv);
    fflush(stderr);

    float *logbuf = d_logits_out ? d_logits_out : ctx->d_logits;

    int r = tq_attn_logits(cache, d_query, logbuf, n_queries, n_heads_q);
    if (r != 0) return r;
    return tq_attn_output(cache, logbuf, d_output, n_queries, n_heads_q);
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

extern "C"
void llama_tq_print_stats(const struct llama_tq_context *ctx) {
    if (!ctx) return;
    const tq_config_t &c  = ctx->cfg;
    size_t n_layers        = ctx->layers.size();
    size_t tq_total        = n_layers * tq_bytes_per_token(&c) * c.max_seq_len;
    size_t fp16_total      = n_layers * (size_t)c.n_heads_kv * c.head_dim
                             * c.max_seq_len * 4;  // K+V, 2 bytes each
    fprintf(stderr, "[TQ] KV memory: TQ=%.1f MB  fp16=%.1f MB  ratio=%.2fx\n",
            tq_total / 1e6f, fp16_total / 1e6f,
            (double)fp16_total / (double)tq_total);
}

#endif // LLAMA_TURBOQUANT
