# TurboQuant v1.1.0 for RDNA4

## Overview
KV cache quantization for AMD GPUs (ROCm). Optimized for RDNA4 (RX 9070).

## Key Features (v1.1.0)
- 128-bit vectorized loads (uint4)
- 128-byte memory alignment
- No LDS buffering
- __ldg() intrinsic for Infinity Cache

## Performance
- Context: 1.02s for 64K tokens
- Token gen: 5+ tok/s on 27B models

## Compatible GPUs
- RX 9070 (gfx1201)
- RX 9060 XT (gfx1200)
- RX 7900 XTX/XT/X (gfx1100/1101/1102)

## Installation

### 1. Prepare ROCm Runtime (Required)
This patch requires AMD ROCm runtime libraries. Without this, GPU will not initialize.

**Option A: Using LM Studio (Recommended)**
- LM Studio already includes rocblas in its backend folder
- Simply replace the DLLs as shown below

**Option B: Standalone llama.cpp**
- Download and install AMD ROCm SDK for Windows
- Copy the `rocblas` folder (containing `library/` subfolder with .co files) to your execution directory

Directory structure:
```
your_app_folder/
├── llama.dll (replace with patched version)
├── ggml-hip.dll (replace with patched version)
├── rocblas/
│   └── library/
│       └── (various .co kernel files)
├── amdhip64_7.dll
├── libhipblas.dll
└── libhipblaslt.dll
```

### 2. Apply DLL Patch
1. Download dll.zip from Releases
2. Replace the following DLLs in your backend/execution folder:
   - llama.dll
   - ggml.dll
   - ggml-base.dll
   - ggml-cpu.dll
   - ggml-hip.dll

### 3. Environment Variable (Optional)
If using standalone llama.cpp, set:
```batch
set ROCBLAS_LIBRARY_PATH=<path_to>\rocblas\library
```

## Usage
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

## Build from Source
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git apply turboquant_changes.patch

cmake -B build -G Ninja \
  -DCMAKE_C_COMPILER="clang" \
  -DCMAKE_CXX_COMPILER="clang++" \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DLLAMA_TURBOQUANT=ON \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build
```

## Files
- dll.zip - Pre-built DLLs (llama.dll, ggml-hip.dll, etc.)
- turboquant_changes.patch - Source code patch for custom builds
- README.md - This file