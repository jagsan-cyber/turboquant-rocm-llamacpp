import os

def patch_file(filepath, old_str, new_string):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(old_str, new_string)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# 1. Update turboquant.h
tq_h = 'ggml/src/turboquant_rocm/turboquant.h'
patch_file(tq_h, 
    'TQ_API int  llama_tq_store (struct llama_tq_context *ctx, int layer, int token,\n                     const void *d_keys, const void *d_vals);',
    'TQ_API int  llama_tq_store_k(struct llama_tq_context *ctx, int layer, int start_token, const void *d_keys, int n_tokens);\n' +
    'TQ_API int  llama_tq_store_v(struct llama_tq_context *ctx, int layer, int start_token, const void *d_vals, int n_tokens);')

patch_file(tq_h,
    'TQ_API int  tq_store_kv(tq_kv_cache_t *cache,\n                 const void    *d_keys,\n                 const void    *d_vals,\n                 int            token);',
    'TQ_API int  tq_store_k(tq_kv_cache_t *cache, const void *d_keys_fp16, int token);\n' +
    'TQ_API int  tq_store_v(tq_kv_cache_t *cache, const void *d_vals_fp16, int token);')

# 2. Update turboquant.cpp
tq_cpp = 'ggml/src/turboquant_rocm/turboquant.cpp'
old_tq_store = '''int tq_store_kv(tq_kv_cache_t *cache,
                const void    *d_keys_fp16,
                const void    *d_vals_fp16,
                int            token)
{
    const tq_config_t *cfg = &cache->cfg;
    int d = cfg->head_dim;
    int H = cfg->n_heads_kv;
    hipStream_t stream = nullptr;

    // Use pre-allocated scratch buffers
    float *d_kf      = cache->d_tmp_kf;
    float *d_vf      = cache->d_tmp_vf;
    float *d_kf_orig = cache->d_tmp_kf_orig;
    float *d_vf_orig = cache->d_tmp_vf_orig;
    float *d_recon_k = cache->d_tmp_recon_k;
    float *d_recon_v = cache->d_tmp_recon_v;

    // fp16 -> float
    tq_launch_fp16_to_float(d_keys_fp16, d_kf, H * d, (tq_hip_stream_t)stream);
    tq_launch_fp16_to_float(d_vals_fp16, d_vf, H * d, (tq_hip_stream_t)stream);

    // Save pre-rotation originals for QJL residual
    HIP_CHECK(hipMemcpyAsync(d_kf_orig, d_kf, H*d*sizeof(float), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_vf_orig, d_vf, H*d*sizeof(float), hipMemcpyDeviceToDevice, stream));

    // Stage-1: Rotate
    tq_launch_rotate(d_kf, cache->d_sign_k1, cache->d_sign_k2, 1, H, d, (tq_hip_stream_t)stream);
    tq_launch_rotate(d_vf, cache->d_sign_v1, cache->d_sign_v2, 1, H, d, (tq_hip_stream_t)stream);

    // Stage-1: Quantize
    int k_bpv    = (d * cfg->key_bits + 7) / 8;
    int v_bpv    = (d * cfg->val_bits + 7) / 8;
    uint8_t *kq  = cache->d_keys_quant + (size_t)token * H * k_bpv;
    uint8_t *vq  = cache->d_vals_quant + (size_t)token * H * v_bpv;

    HIP_CHECK(hipMemsetAsync(kq, 0, H * k_bpv, stream));
    HIP_CHECK(hipMemsetAsync(vq, 0, H * v_bpv, stream));

    tq_launch_quantize(d_kf, kq, d_recon_k, 1, H, d, cfg->key_bits, (tq_hip_stream_t)stream);
    tq_launch_quantize(d_vf, vq, d_recon_v, 1, H, d, cfg->val_bits, (tq_hip_stream_t)stream);

    // Stage-2: QJL residual
    if (cfg->use_qjl) {
        int qjl_bpv   = (d + 7) / 8;
        uint8_t *kqjl = cache->d_keys_qjl + (size_t)token * H * qjl_bpv;
        uint8_t *vqjl = cache->d_vals_qjl + (size_t)token * H * qjl_bpv;
        HIP_CHECK(hipMemsetAsync(kqjl, 0, H * qjl_bpv, stream));
        HIP_CHECK(hipMemsetAsync(vqjl, 0, H * qjl_bpv, stream));

        tq_launch_qjl(d_kf_orig, d_recon_k, cache->d_sign_k1, cache->d_sign_k2,
                      cache->d_qjl_proj_k, kqjl, 1, H, d, (tq_hip_stream_t)stream);
        tq_launch_qjl(d_vf_orig, d_recon_v, cache->d_sign_v1, cache->d_sign_v2,
                      cache->d_qjl_proj_v, vqjl, 1, H, d, (tq_hip_stream_t)stream);
    }
    
    // HIP_CHECK(hipStreamSynchronize(stream)); // removed for async performance
    
    if (token >= cache->n_tokens) {
        cache->n_tokens = token + 1;
    }
    return 0;
}'''
old_tq_store = old_tq_store.replace('->', '→')

new_tq_store = '''int tq_store_k(tq_kv_cache_t *cache, const void *d_keys_fp16, int token) {
    const tq_config_t *cfg = &cache->cfg; int d = cfg->head_dim; int H = cfg->n_heads_kv; hipStream_t stream = nullptr;
    float *d_kf = cache->d_tmp_kf; float *d_kf_orig = cache->d_tmp_kf_orig; float *d_recon_k = cache->d_tmp_recon_k;
    tq_launch_fp16_to_float(d_keys_fp16, d_kf, H * d, (tq_hip_stream_t)stream);
    HIP_CHECK(hipMemcpyAsync(d_kf_orig, d_kf, H*d*sizeof(float), hipMemcpyDeviceToDevice, stream));
    tq_launch_rotate(d_kf, cache->d_sign_k1, cache->d_sign_k2, 1, H, d, (tq_hip_stream_t)stream);
    int k_bpv = (d * cfg->key_bits + 7) / 8; uint8_t *kq = cache->d_keys_quant + (size_t)token * H * k_bpv;
    HIP_CHECK(hipMemsetAsync(kq, 0, H * k_bpv, stream));
    tq_launch_quantize(d_kf, kq, d_recon_k, 1, H, d, cfg->key_bits, (tq_hip_stream_t)stream);
    if (cfg->use_qjl) {
        int qjl_bpv = (d + 7) / 8; uint8_t *kqjl = cache->d_keys_qjl + (size_t)token * H * qjl_bpv;
        HIP_CHECK(hipMemsetAsync(kqjl, 0, H * qjl_bpv, stream));
        tq_launch_qjl(d_kf_orig, d_recon_k, cache->d_sign_k1, cache->d_sign_k2, cache->d_qjl_proj_k, kqjl, 1, H, d, (tq_hip_stream_t)stream);
    }
    return 0;
}
int tq_store_v(tq_kv_cache_t *cache, const void *d_vals_fp16, int token) {
    const tq_config_t *cfg = &cache->cfg; int d = cfg->head_dim; int H = cfg->n_heads_kv; hipStream_t stream = nullptr;
    float *d_vf = cache->d_tmp_vf; float *d_vf_orig = cache->d_tmp_vf_orig; float *d_recon_v = cache->d_tmp_recon_v;
    tq_launch_fp16_to_float(d_vals_fp16, d_vf, H * d, (tq_hip_stream_t)stream);
    HIP_CHECK(hipMemcpyAsync(d_vf_orig, d_vf, H*d*sizeof(float), hipMemcpyDeviceToDevice, stream));
    tq_launch_rotate(d_vf, cache->d_sign_v1, cache->d_sign_v2, 1, H, d, (tq_hip_stream_t)stream);
    int v_bpv = (d * cfg->val_bits + 7) / 8; uint8_t *vq = cache->d_vals_quant + (size_t)token * H * v_bpv;
    HIP_CHECK(hipMemsetAsync(vq, 0, H * v_bpv, stream));
    tq_launch_quantize(d_vf, vq, d_recon_v, 1, H, d, cfg->val_bits, (tq_hip_stream_t)stream);
    if (cfg->use_qjl) {
        int qjl_bpv = (d + 7) / 8; uint8_t *vqjl = cache->d_vals_qjl + (size_t)token * H * qjl_bpv;
        HIP_CHECK(hipMemsetAsync(vqjl, 0, H * qjl_bpv, stream));
        tq_launch_qjl(d_vf_orig, d_recon_v, cache->d_sign_v1, cache->d_sign_v2, cache->d_qjl_proj_v, vqjl, 1, H, d, (tq_hip_stream_t)stream);
    }
    return 0;
}'''
patch_file(tq_cpp, old_tq_store, new_tq_store)

# 3. Update llama_tq_hook.cpp
tq_hook = 'ggml/src/turboquant_rocm/llama_tq_hook.cpp'
old_hook_store = '''extern "C"
int llama_tq_store(struct llama_tq_context *ctx, int layer, int token,
                   const void *d_keys, const void *d_vals) {
    if (!ctx || layer < 0 || layer >= (int)ctx->layers.size()) return -1;
    return tq_store_kv(&ctx->layers[layer], d_keys, d_vals, token);
}'''
new_hook_store = '''extern "C" int llama_tq_store_k(struct llama_tq_context *ctx, int layer, int start_token, const void *d_keys, int n_tokens) {
    if (!ctx || layer < 0 || layer >= (int)ctx->layers.size()) return -1;
    auto &l = ctx->layers[layer];
    const _Float16* k_ptr = (const _Float16*)d_keys;
    int stride = l.cfg.n_heads_kv * l.cfg.head_dim;
    for (int i = 0; i < n_tokens; i++) {
        tq_store_k(&l, k_ptr + i * stride, start_token + i);
    }
    return 0;
}
extern "C" int llama_tq_store_v(struct llama_tq_context *ctx, int layer, int start_token, const void *d_vals, int n_tokens) {
    if (!ctx || layer < 0 || layer >= (int)ctx->layers.size()) return -1;
    auto &l = ctx->layers[layer];
    const _Float16* v_ptr = (const _Float16*)d_vals;
    int stride = l.cfg.n_heads_kv * l.cfg.head_dim;
    for (int i = 0; i < n_tokens; i++) {
        tq_store_v(&l, v_ptr + i * stride, start_token + i);
    }
    if (start_token + n_tokens > l.n_tokens) {
        l.n_tokens = start_token + n_tokens;
    }
    return 0;
}'''
patch_file(tq_hook, old_hook_store, new_hook_store)

# 4. Update ggml-cuda.cu
cuda_cu = 'ggml/src/ggml-cuda/ggml-cuda.cu'
old_cpy = '''        case GGML_OP_CPY:
#if defined(GGML_USE_HIP) && defined(LLAMA_TURBOQUANT)
            {
                struct llama_tq_context * tq = llama_tq_get_global();
                if (tq && dst->view_src && dst->view_src->name[0] != '\\0') {
                    if (strncmp(dst->view_src->name, "cache_k_l", 9) == 0) {
                        try {
                            int il = std::stoi(dst->view_src->name + 9);
                            llama_tq_store_k(tq, il, src0->data, src0->ne[1]);
                            break; // Skip standard copy to dummy buffer
                        } catch (...) {}
                    } else if (strncmp(dst->view_src->name, "cache_v_l", 9) == 0) {
                        try {
                            int il = std::stoi(dst->view_src->name + 9);
                            llama_tq_store_v(tq, il, src0->data, src0->ne[1]);
                            break; // Skip standard copy to dummy buffer
                        } catch (...) {}
                    }
                }
            }
#endif
            ggml_cuda_op_cpy(ctx, dst);
            break;'''
new_cpy = '''        case GGML_OP_CPY:
#if defined(GGML_USE_HIP) && defined(LLAMA_TURBOQUANT)
            {
                struct llama_tq_context * tq = llama_tq_get_global();
                if (tq && dst->view_src && dst->view_src->name[0] != '\\0') {
                    if (strncmp(dst->view_src->name, "cache_k_l", 9) == 0) {
                        try {
                            int il = std::stoi(dst->view_src->name + 9);
                            int token_idx = dst->offs / (dst->view_src->ne[0] * ggml_type_size(dst->view_src->type));
                            llama_tq_store_k(tq, il, token_idx, src0->data, src0->ne[1]);
                            break; // Skip standard copy to dummy buffer
                        } catch (...) {}
                    } else if (strncmp(dst->view_src->name, "cache_v_l", 9) == 0) {
                        try {
                            int il = std::stoi(dst->view_src->name + 9);
                            int token_idx = dst->offs / (dst->view_src->ne[0] * ggml_type_size(dst->view_src->type));
                            llama_tq_store_v(tq, il, token_idx, src0->data, src0->ne[1]);
                            break; // Skip standard copy to dummy buffer
                        } catch (...) {}
                    }
                }
            }
#endif
            ggml_cuda_op_cpy(ctx, dst);
            break;'''
patch_file(cuda_cu, old_cpy, new_cpy)

old_decl = '''    int llama_tq_store_k(struct llama_tq_context * ctx, int layer, const void * d_keys, int n_tokens);
    int llama_tq_store_v(struct llama_tq_context * ctx, int layer, const void * d_vals, int n_tokens);'''
new_decl = '''    int llama_tq_store_k(struct llama_tq_context * ctx, int layer, int start_token, const void * d_keys, int n_tokens);
    int llama_tq_store_v(struct llama_tq_context * ctx, int layer, int start_token, const void * d_vals, int n_tokens);'''
patch_file(cuda_cu, old_decl, new_decl)

print("Patching complete!")
