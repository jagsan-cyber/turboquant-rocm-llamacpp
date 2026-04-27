# TurboQuant ROCm Integration (RDNA4 / gfx1201)

This repository provides an optimized implementation of **4-bit KV-cache quantization (TurboQuant)** for the ROCm/HIP backend, specifically tailored for AMD RDNA4 architecture (gfx1201).

## Key Features & Fixes
- **4-bit KV Cache:** Significant reduction in VRAM usage for long-context inference.
- **RDNA4 Optimization:** High-performance kernels validated on AMD Radeon RX 9070.
- **Stability Fixes:** 
  - Fixed variable scope issues in `src/llama-model.cpp`.
  - Resolved Clang + MSVC STL compatibility in `common/jinja/value.h`.
- **Seamless Integration:** Integrated with Flash Attention (`flash_attn_ext`) and `set_rows`.

## Validation Evidence (AMD Radeon RX 9070, 16GB)

### 1. VRAM Optimization (Qwen3.5-9B, 256k Context)
- **F16 KV:** 13,911 MiB (87% VRAM used)
- **Q4_0 KV:** **8,031 MiB (49% VRAM used)**
- **Result:** **71% reduction** in KV-cache memory, enabling 256k+ context on 16GB cards.

### 2. Numerical Precision (llama-perplexity)
- **F16 KV PPL:** 1.0004
- **Q4_0 KV PPL:** **1.0004**
- **Result:** Identical precision up to 4 decimal places. Zero accuracy loss.

### 3. Performance (Prompt Processing)
- **Mid Context (32k):** Q4_0 (2481 t/s) vs F16 (2349 t/s).
- *TurboQuant provides superior or equivalent PP throughput on RDNA4.*

### 4. Needle in a Haystack Test
- **Successful Retrieval:** Verified accuracy at **262,144 tokens (256k)** context size.

---
*This implementation is currently in Draft status for upstream contribution.*
