# TurboQuant for RDNA4 (RX 9070) - v1.1.0

## Overview
TurboQuant is a KV cache quantization implementation for llama.cpp with AMD GPUs (ROCm). This version is optimized for RDNA4 GPUs (RX 9070/XT, RX 9060 XT) using 128-bit vectorized memory access.

## Key Features (v1.1.0)
- **128-bit vectorized loads** (`uint4`) for KV cache dequantization
- **128-byte memory alignment** for Infinity Cache optimization
- **No LDS buffering** - eliminates bank conflicts, reduces latency
- **`__ldg()` intrinsic** - maximizes L3 cache hit rate

## Performance
- Context processing: **1.02s for 64K tokens**
- Token generation: **5+ tok/s** on Qwen-27B-Reasoning (16GB VRAM)

## Compatible GPUs
| GPU | Architecture |
|-----|--------------|
| RX 9070 / R9700 | gfx1201 |
| RX 9060 XT | gfx1200 |
| RX 7900 XTX/XT/X | gfx1100/1101/1102 |
| RX 7800/7700 | gfx1103/1102 |

## Installation (LM Studio)

1. Find your backend folder (e.g., `%APPDATA%\lm-studio\bin\axona\llama.cpp-win-x86_64-amd-rocm-avx2-2.9.0`)
2. Backup original DLLs
3. Replace with TurboQuant v1.1.0 DLLs:
   - `llama.dll`
   - `ggml.dll`
   - `ggml-base.dll`
   - `ggml-cpu.dll`
   - `ggml-hip.dll`

## Usage

### llama.cpp CLI
```batch
llama-cli.exe -m model.gguf -ngl 99 -ctk q4_0 -ctv q4_0 -fa 1 -p 32768 -n 32 -t 7
```

### Parameters
| Flag | Description |
|------|-------------|
| `-ngl 99` | Load model to GPU |
| `-ctk q4_0` | TurboQuant key cache (4-bit) |
| `-ctv q4_0` | TurboQuant value cache (4-bit) |
| `-fa 1` | Flash attention |
| `-p 32768` | Prompt tokens (context size) |
| `-n 32` | Generation tokens |
| `-t 7` | CPU threads |

## Files
- `ggml-hip.dll` - TurboQuant HIP backend (main optimization)
- `llama.dll` - llama.cpp core
- `ggml-base.dll` - GGML base library
- `ggml-cpu.dll` - GGML CPU backend
- `ggml.dll` - GGML shared

## Build from Source
```bash
cmake -B build -G Ninja \
  -DCMAKE_C_COMPILER="clang" \
  -DCMAKE_CXX_COMPILER="clang++" \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DLLAMA_TURBOQUANT=ON \
  -DCMAKE_BUILD_TYPE=Release
```

## License
MIT (same as llama.cpp)