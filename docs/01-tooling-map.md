# Tooling map: Python → C++ / CUDA

## Equivalents

| Python | C++ / CUDA equivalent | What it does |
|---|---|---|
| uv / pip | **CMake** (+ vcpkg / Conan / CPM for libraries) | Build system. This project only needs CUDA + cuBLAS (already installed), so no package manager |
| — | **Ninja** | Faster build backend CMake can use instead of make |
| `pyproject.toml` | `CMakeLists.txt` + `CMakePresets.json` | Project definition + named build configurations |
| ruff format | **clang-format** | Formatter, configured by `.clang-format` |
| ruff check | **clang-tidy** | Linter / bug finder / modernizer, configured by `.clang-tidy` |
| ty / pyright (editor) | **clangd** | Language server: autocomplete, go-to-definition, live errors. Configured by `.clangd` |
| type checker strict mode | **compiler warnings** | The compiler *is* the type checker. See below |
| pre-commit (ruff hook) | pre-commit (clang-format hook) | Same tool, C++ hooks |
| pytest | GoogleTest / Catch2 | Not needed here: the harness comparing against cuBLAS *is* the test |
| pdb | **gdb / cuda-gdb** | Debuggers (CPU / GPU) |
| — | **compute-sanitizer** | Out-of-bounds, races, uninitialized reads *inside kernels*. Run it on every new kernel |
| — | **ASan / UBSan** (`-fsanitize=address,undefined`) | Memory errors / undefined behavior in *host* code |
| cProfile | **ncu** (one kernel) / **nsys** (whole-program timeline) | Profilers |
| pytest-cov | `--coverage` + gcov/lcov | Coverage (not needed here) |

## Compiler warnings = strict mode

Host code (g++), roughly in order of usefulness for matmul:

```
-Wall -Wextra -Wshadow
-Wconversion -Wsign-conversion   # int <-> size_t <-> float in index math
-Wdouble-promotion               # float silently becoming double
```

nvcc passes host flags on with `-Xcompiler=...` and has its own few warnings
(`-Wreorder`, `--Werror=all-warnings`). Always build with `-lineinfo` so
`ncu` and `compute-sanitizer` can point at source lines (no runtime cost).

**Important:** g++ only sees the *host* half of a `.cu` file. Kernels are
compiled by NVIDIA's device compiler, which ignores most of these warnings.
clangd / clang-tidy *do* parse device code, so they partly fill the gap.
See [05-lessons-learned.md](05-lessons-learned.md) for what this means in practice.

## Conventions / style guides

- **Google C++ Style Guide**: complete and widely used. This repo's `.clang-format` starts from it.
- **C++ Core Guidelines** (Stroustrup & Sutter): about writing *correct* modern C++, not formatting.
  clang-tidy checks many of them (`cppcoreguidelines-*`).
- **LLVM style**: common alternative for formatting (cccl uses it).

As with ruff: pick a base style in `.clang-format` and stop thinking about it.

## Installing on Ubuntu

| Source | Notes |
|---|---|
| Ubuntu apt: `clangd clang-tidy clang-format` | Default version (21 on Ubuntu 26.04). **Too old for CUDA 13 + GCC 15**, see lessons learned |
| Ubuntu apt, versioned: `clangd-22 clang-tidy-22` | Commands are named `clangd-22` etc. **Use this** |
| apt.llvm.org | LLVM's own apt repo; newest and nightly versions |
| GitHub releases (llvm-project, clangd/clangd) | Official prebuilt archives, no root needed, manual updates |
| `uv tool install clang-format` / `clang-tidy` | Community-packaged PyPI builds; no root; no clangd |

`sudo` needs a real terminal for the password prompt, so run installs in your
own terminal, not through a tool that can't prompt.
