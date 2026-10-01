# Docs

Learning notes for this project: the CUDA matmul concepts, and the C++/CUDA
tooling around them.

| File | What's in it |
|---|---|
| [glossary.md](glossary.md) | GEMM, SGEMM, FLOPS, roofline, SMEM, coalescing, ... the article's vocabulary |
| [01-tooling-map.md](01-tooling-map.md) | Python tools you know → their C++/CUDA equivalents; compiler warnings; conventions |
| [02-cmake-structure.md](02-cmake-structure.md) | Why CMake is split into `cmake/*.cmake` files, `include()` vs `add_subdirectory()`, every file in cmake_template explained |
| [03-config-files.md](03-config-files.md) | What's inside `.clang-tidy`, `.clang-format`, `.clangd`, `.pre-commit-config.yaml`, compared across cmake_template, cccl, llama.cpp and this repo |
| [04-reference-repos.md](04-reference-repos.md) | Big repos worth learning from, and which files to read in each |
| [05-lessons-learned.md](05-lessons-learned.md) | Gotchas found (and verified) while setting up this repo: nvcc vs g++ warnings, arch bug, clang 21 vs 22, ... |
| [06-inspecting-kernels.md](06-inspecting-kernels.md) | PTX vs SASS, `cuobjdump` / `nvdisasm`, reading `ncu` stall reasons (MIO throttle), Godbolt, tool cheat sheet |

## How this repo's tooling fits together

```
CMakeLists.txt ──include──> cmake/CompilerWarnings.cmake   (warnings, per language)
               └─include──> cmake/Sanitizers.cmake         (ASan + UBSan, host code)
CMakePresets.json           release / debug / asan build modes
  └─ build/<preset>/compile_commands.json ──> read by clangd (via .clangd)

.clang-format   ──> clang-format   (formatter)      ┐
.clang-tidy     ──> clang-tidy     (linter)         ├─ scripts/lint.sh runs both
.clangd         ──> clangd         (editor/LSP)     ┘  clangd also runs clang-tidy live

compute-sanitizer ./build/debug/sgemm N   (memory errors *inside kernels*)
ncu / nsys                                (profiling)
```
