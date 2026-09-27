# Lessons learned

Gotchas found while setting up this repo. Each one was reproduced and tested
on this machine (RTX 3050 Laptop sm_86, CUDA 13.1, GCC 15.2, CMake 4.2, Ubuntu 26.04).

## 1. `CMAKE_CUDA_ARCHITECTURES` must be set before `project()`

**Symptom:** the build compiled for **sm_75** (Turing), not sm_86. It still ran, because
the driver JIT-compiles the embedded PTX for the real GPU at startup.

**Cause:** `project(... CUDA)` fills `CMAKE_CUDA_ARCHITECTURES` with nvcc's default (75)
if it's unset. An `if(NOT DEFINED ...)` placed after `project()` is therefore always false.

**Why it matters:** code isn't tuned for Ampere, and sm_80+ features (e.g. `cp.async`,
used by later kernels) aren't available, so benchmarks would be misleading.

**Check it:**
```
cuobjdump --list-elf build/release/sgemm      # should say sm_86
grep -o -- '--generate-code=[^ "\\]*' build/release/compile_commands.json
```
After changing it, delete `build/` because the old value is cached in `CMakeCache.txt`.

## 2. `-Xcompiler` splits on commas

`-Xcompiler=-fsanitize=address,undefined` becomes `-fsanitize=address` **and** a separate
argument `undefined`, which gcc treats as an input file:
`gcc: fatal error: cannot specify '-o' with '-c' ... with multiple files`.
Fix: `-fsanitize=address -fsanitize=undefined` as separate flags.

## 3. g++ warnings are weaker in `.cu` files

nvcc's front end (cudafe) **rewrites your code** before g++ sees it. Keep the intermediate
files to see it: `nvcc -keep ...` → `*.cudafe1.cpp`. Example: `else if (...)` became
`else { if (...) }`, and braces/layout were changed.

| Warning | In `.cpp` | In `.cu` (via nvcc) |
|---|---|---|
| `-Wconversion` | ✅ | ✅ |
| `-Wduplicated-branches` | ✅ | ✅ |
| `-Wlogical-op` | ✅ | ✅ |
| `-Wduplicated-cond` | ✅ | ❌ `else if` chain no longer exists |
| `-Wmisleading-indentation` | ✅ | ❌ layout rewritten |
| `-Wpedantic`, `-Wold-style-cast` | ✅ | ❌ fire on nvcc's generated stubs |

And none of the g++ warnings see **device code** (kernels) at all. clangd / clang-tidy
parse device code, so they partly fill the gap. nvcc's own warnings (`--Werror=all-warnings`)
do work: the debug preset rejected an unused function (`#177-D declared but never referenced`).

## 4. clang 21 is too old for CUDA 13.1 + GCC 15

| Tool | clang 21 | clang 22 |
|---|---|---|
| clangd on `main.cu` using `std::vector` | ❌ false errors in `_Vector_base` (CUDA mode + GCC 15 libstdc++) | ✅ |
| clangd on a `.cuh` kernel | ✅ | ✅ |
| clang-tidy / clang++ in CUDA mode | ❌ `texture_fetch_functions.h` not found (removed in CUDA 13) | ✅ |
| clang-format | ✅ | ✅ |

`--no-cuda-version-check` only silences the version warning; it doesn't fix these.
Install `clangd-22 clang-tidy-22` and point the editor to `clangd-22`.

How it was narrowed down: a one-line `#include <vector>` file had no errors in C++ mode and
errors in CUDA mode with C++17 and C++20 alike → the flags weren't the cause, the clang version was.
clang 22 was tested without root by unpacking the `.deb`s:
`apt-get download clangd-22 ... && dpkg -x pkg.deb dir/`, then running with `LD_LIBRARY_PATH`.

## 5. clang-tidy doesn't read `.clangd`

clangd applies `.clangd`'s `Remove:` list; clang-tidy doesn't, so `clang-tidy -p build/debug`
fails on every nvcc flag (`unknown argument: '-Xcompiler=...'`).
Fix: ignore the compile database and pass clang flags after `--`:

```
clang-tidy src/main.cu -- -xcuda -std=c++20 -Isrc --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_86 ...
```

That's what `scripts/lint.sh` does. Also: `clangd --check=file` only counts **errors**; it
never prints warnings, so it's a config checker, not a linter.

## 6. What the tools caught on a naive SGEMM test

- **clang-tidy:** `std::vector<float> h(M * M, ...)` multiplies in `int`, then widens to
  `size_t` (`bugprone-implicit-widening-of-multiplication-result`). With big matrices the
  `int` multiplication overflows first. Common in matmul index math.
- **nvcc `--Werror=all-warnings`:** unused function rejected.
- **clang-format `--dry-run --Werror`:** one-line function bodies not allowed by the style.

## 7. Smaller ones

- **`.gitignore` has no end-of-line comments.** `build/  # comment` is a pattern containing
  the text `# comment`. Put comments on their own line.
- **ASan + CUDA** needs `ASAN_OPTIONS=protect_shadow_gap=0`, or the CUDA driver can't map memory.
- **make skips rebuilds** when a file changes within the same second as the last build.
  Use `cmake --build ... --clean-first` when testing.
- **`sudo` can't prompt** through a tool without a terminal (`sudo: A terminal is required`);
  run installs in your own terminal.
- **The article's repo** (`find_package(CUDA)`) still builds with CMake 4.2 but warns
  (policy CMP0146: FindCUDA removed). New code should use `find_package(CUDAToolkit)`.

## First measurements (4096×4096, article's repo, single runs)

| Kernel | GFLOPS |
|---|---|
| cuBLAS | 4174 |
| 1: naive | 59 |
| 10: warptiling | 4547 |

Laptop clocks vary with temperature; average several runs before trusting numbers.
Measured cuBLAS (~4.2 TFLOPS) is the practical ceiling, below the ~6 TFLOPS theoretical peak.
