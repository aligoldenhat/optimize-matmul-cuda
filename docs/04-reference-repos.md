# Reference repos

Large, well-maintained projects to learn tooling (and later, kernels) from.
File paths verified on GitHub in September 2026.

## 1. Start here: made for learning

**[cpp-best-practices/cmake_template](https://github.com/cpp-best-practices/cmake_template)**
The "pyproject.toml with every option" of C++. One tool per file, short and commented.
- `cmake/CompilerWarnings.cmake`: warning list with a comment per flag
- `cmake/Sanitizers.cmake`, `cmake/StaticAnalyzers.cmake`, `cmake/Hardening.cmake`
- `cmake/Cuda.cmake`
- `ProjectOptions.cmake`: where all the on/off switches live
- `CMakePresets.json`, `.clang-tidy`, `.clang-format`

Companion book: [cpp-best-practices/cppbestpractices](https://github.com/cpp-best-practices/cppbestpractices)
(the reasoning behind those settings).
Modern CMake guide: https://cliutils.gitlab.io/modern-cmake

## 2. Closest to this project: CUDA + C++

**[NVIDIA/cccl](https://github.com/NVIDIA/cccl)** (Thrust, CUB, libcudacxx)
Has every config, including a CUDA-aware `.clangd` (the hardest one to get right).
- `.clangd`, `.clang-tidy`, `.clang-format`, `CMakePresets.json`
- `.pre-commit-config.yaml`: clang-format + ruff + mypy + more before each commit

**[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**
Real CUDA inference code, similar in spirit to an ADAS/YOLO pipeline.
- `ggml/src/ggml-cuda/CMakeLists.txt`: nvcc flags, architectures, options in a real project
- `CMakePresets.json`, `.clang-format`, `.clang-tidy`, `.pre-commit-config.yaml`
- `ggml/src/ggml-cuda/`: real hand-written matmul kernels (read after the article)

**[NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)**
NVIDIA's GEMM library: the professional version of the article. Lighter tooling
(no top-level `.clang-tidy`); read it for **kernel and template design** after finishing the article.

## 3. Very large projects: tooling at scale

**[pytorch/pytorch](https://github.com/pytorch/pytorch)**
- `.lintrunner.toml`: **closest thing to ruff config in pyproject.toml**; one file drives
  clang-format, clang-tidy, ruff, mypy and more across C++, CUDA and Python
- `.clang-tidy`: large, heavily commented
- `cmake/`: large CMake setup including CUDA detection

**[llvm/llvm-project](https://github.com/llvm/llvm-project)** (authors of clang-format / clang-tidy / clangd)
- `llvm/cmake/modules/HandleLLVMOptions.cmake`: ~78 KB of warning/sanitizer flags for many compilers
- `.clang-tidy`, `.clang-format`: how the tool authors configure their own tools

## 4. Clean C++ to read (style and templates)

- **[fmtlib/fmt](https://github.com/fmtlib/fmt)**: small, clean, modern; good for learning **templates**
- **[abseil/abseil-cpp](https://github.com/abseil/abseil-cpp)**: Google's library, Google style

## Suggested order

1. cmake_template: what each tool does, one file at a time
2. cccl: the same tools applied to CUDA
3. pytorch's `.lintrunner.toml`: how they all run together
4. cutlass: kernels, after the article

## Getting files without cloning GBs

Browse on GitHub, or do a shallow, partial clone:

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/pytorch/pytorch
```

Or fetch a single file:

```
gh api repos/NVIDIA/cccl/contents/.clangd -H "Accept: application/vnd.github.raw"
```

The article's own code is at [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
(cloned at `../SGEMM_CUDA`). It builds on this machine unchanged (its CMake prints a
deprecation warning for `find_package(CUDA)`).
