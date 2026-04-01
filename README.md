# TurboQuant ROCm Build for Windows (AMD RX 9070)

A working build configuration for llama.cpp with TurboQuant (custom KV cache quantization) enabled, targeting AMD GPUs via ROCm on Windows, compatible with LM Studio.

## What is TurboQuant?

TurboQuant is a custom KV cache quantization system that intercepts attention operations to enable dynamic quantization during inference. This can improve memory efficiency and potentially performance on AMD GPUs.

## Prerequisites

- **Windows 10/11** (x64)
- **AMD HIP SDK 7.1** (or compatible version)
- **LM Studio** (latest version recommended)
- **AMD RX 9000 series GPU** (RDNA4 / gfx1201)

### Required Build Tools

- CMake 3.20+
- Ninja build system
- LLVM/Clang toolchain (included with AMD HIP SDK)
- Visual Studio 2019 Build Tools

## What We Changed

### 1. File Renaming
- Renamed `tq_hooks.cu` → `tq_hooks.hip` to fix HIP language detection on Windows

### 2. CMakeLists.txt Modifications
- Updated `ggml/src/ggml-hip/CMakeLists.txt` to reference `tq_hooks.hip`
- Added proper HIP language configuration for the .hip file

### 3. Include Path Fix
- Changed include in `tq_hooks.hip` from `ggml-cuda.h` to `../ggml-cuda/common.cuh` to access the complete `ggml_backend_cuda_context` type definition

### 4. Additional Changes
- Modified `ggml/src/CMakeLists.txt` for TurboQuant support
- Updated source files in `ggml/src/turboquant_rocm/`
- Added integration files in `src/`

## Build Instructions

### Step 1: Clone llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

### Step 2: Apply the Patch

```bash
git apply turboquant_changes.patch
```

Or manually apply the changes listed in the patch file.

### Step 3: Configure with CMake

```powershell
$env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;$env:PATH"
$env:ONEAPI_ROOT = "C:\fake"  # Hack to avoid -lm.lib linker error

cmake -S . -B build `
    -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_TOOLCHAIN_FILE="cmake\x64-windows-llvm.cmake" `
    -DGGML_HIP=ON `
    -DGGML_CPU=ON `
    -DGGML_OPENMP=OFF `
    -DLLAMA_TURBOQUANT=ON `
    -DAMDGPU_TARGETS=gfx1201
```

### Step 4: Build

```powershell
cmake --build build --parallel 4
```

### Important Notes

- **ONEAPI_ROOT Hack**: Setting `$env:ONEAPI_ROOT = "C:\fake"` bypasses a linker issue where CMake tries to link against `m.lib`. This is required for Windows builds.

- **GGML_OPENMP=OFF**: OpenMP is disabled to avoid linking errors with the Clang toolchain. The CPU backend still works without it.

- **GGML_CPU=ON**: Required even for full GPU offload - llama.cpp needs the CPU backend for buffer allocation.

## Installation in LM Studio

### Step 1: Locate Your LM Studio Backend Folder

Find the ROCm backend folder in LM Studio's extension directory:
```
C:\Users\<YourUsername>\.lmstudio\extensions\backends\llama.cpp-win-x86_64-amd-rocm-avx2-2.9.0\
```

### Step 2: Copy DLLs

Copy the following DLLs from your build output to the LM Studio backend folder:
- `ggml-base.dll`
- `ggml-cpu.dll`
- `ggml-hip.dll`
- `ggml.dll`
- `llama.dll`

### Step 3: Copy rocblas (CRITICAL!)

To prevent crashes during long prompt processing (rocBLAS errors), you MUST copy the rocblas folder from AMD HIP SDK:

1. Find the rocblas folder in your AMD HIP SDK:
   ```
   C:\Program Files\AMD\ROCm\7.1\bin\rocblas\
   ```

2. Copy the entire `rocblas` folder to the LM Studio backend folder

3. Final structure should look like:
   ```
   ...\llama.cpp-win-x86_64-amd-rocm-avx2-2.9.0\
   ├── ggml-hip.dll
   ├── ggml-base.dll
   ├── ggml-cpu.dll
   ├── ggml.dll
   ├── llama.dll
   └── rocblas\
       └── library\
           ├── TensileLibrary_lazy_gfx1201.dat
           └── ...
   ```

### Step 4: Test

1. Open LM Studio
2. Select AMD ROCm as the backend
3. Load a model with IQ1_S quantization (or other TurboQuant-compatible quantization)
4. Run inference - it should work on your RX 9070!

## Troubleshooting

### "make_cpu_buft_list: no CPU backend found"
- Make sure `ggml-cpu.dll` is copied to the backend folder

### rocBLAS Errors / Crashes
- Verify the `rocblas` folder is copied from AMD HIP SDK
- Make sure `TensileLibrary_lazy_gfx1201.dat` exists in the library folder

### Build Errors
- Ensure ONEAPI_ROOT is set to a fake path
- Verify HIP SDK is installed correctly

## Performance Notes

This build is specifically optimized for:
- **GPU**: AMD RX 9070 (RDNA4, gfx1201)
- **Backend**: ROCm 7.1
- **Quantization**: TurboQuant (IQ1_S recommended)

## Files Included

- `turboquant_changes.patch` - Source code changes
- `ggml-base.dll` - Base GGML library
- `ggml-cpu.dll` - CPU backend
- `ggml-hip.dll` - HIP/ROCm backend (TurboQuant enabled)
- `ggml.dll` - GGML meta-library
- `llama.dll` - Llama.cpp main library

## License

This build follows the same license as llama.cpp (MIT).

## Credits

- llama.cpp by Georgi Gerganov
- TurboQuant implementation
- AMD ROCm team