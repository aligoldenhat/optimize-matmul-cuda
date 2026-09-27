# CMake structure

## Splitting CMake into files

A small project can keep everything in one `CMakeLists.txt`. Bigger projects
(like [cpp-best-practices/cmake_template](https://github.com/cpp-best-practices/cmake_template))
split it up. There are two mechanisms:

### `include(file.cmake)`: pull in helper code

```cmake
include(cmake/CompilerWarnings.cmake)   # runs that file right here, same variables
set_project_warnings(sgemm ON)          # call the function it defined
```

The included file usually just **defines functions**; `CMakeLists.txt` calls them.
Python analogy: `from utils import set_project_warnings`.

### `add_subdirectory(dir)`: a child project

```cmake
add_subdirectory(src)    # runs src/CMakeLists.txt
add_subdirectory(test)   # runs test/CMakeLists.txt
```

Each folder defines its own targets. Python analogy: subpackages, each with an `__init__.py`.

### Why split

- **One job per file.** Changing warnings means opening a 50-line file, not searching 500 lines.
- **Reusable.** `CompilerWarnings.cmake` can be copied unchanged into the next project.
- **`CMakeLists.txt` reads like a table of contents.** It says *what*, `cmake/` says *how*.

### When one file is fine

Under roughly 150–200 lines, with nothing reusable, one file is simpler.
This repo sits in the middle: `CMakeLists.txt` + two helpers in `cmake/`.
`add_subdirectory` becomes useful once there's more than one program (e.g. a `test/` executable).

## cmake_template's `cmake/` folder, file by file

Each file adds **one feature**. `ProjectOptions.cmake` switches them on/off with `option(...)`.

| File | What it does | Python analogy | This repo |
|---|---|---|---|
| `CompilerWarnings.cmake` | Warning flags per compiler (MSVC, clang, gcc, CUDA), each flag commented | ty strictness / ruff rule selection | ✅ `cmake/CompilerWarnings.cmake` |
| `Sanitizers.cmake` | `-fsanitize=`: address, undefined, leak, thread, memory; checks which combinations are compatible | — (runtime checkers) | ✅ ASan + UBSan |
| `StaticAnalyzers.cmake` | Runs clang-tidy and cppcheck *during the build* via `CMAKE_CXX_CLANG_TIDY` | `ruff check` on every build | ❌ `scripts/lint.sh` + clangd instead |
| `Hardening.cmake` | Security flags for shipped software: `_GLIBCXX_ASSERTIONS`, `_FORTIFY_SOURCE`, stack protector | — | ❌ benchmarking, not shipping |
| `Cuda.cmake` | CUDA target helper: `-fPIC`, separable compilation (`-rdc`), Windows fixes | — | ❌ one `.cu` file |
| `StandardProjectSettings.cmake` | Default build type, `compile_commands.json`, colored diagnostics | — | ✅ presets + `CMAKE_EXPORT_COMPILE_COMMANDS` |
| `PreventInSourceBuilds.cmake` | Refuses to build in the source folder | — | ✅ presets always use `build/` |
| `Cache.cmake` | ccache / sccache: skip recompiling unchanged files | uv's cache | ❌ small build |
| `Linker.cmake` | Pick a faster linker (lld, mold) | — | ❌ |
| `InterproceduralOptimization.cmake` | LTO: optimize across files at link time | — | ❌ doesn't help GPU kernels |
| `Tests.cmake` | Coverage (`--coverage`) | pytest-cov | ❌ |
| `LibFuzzer.cmake` | Fuzz testing support check | hypothesis (roughly) | ❌ |
| `CPM.cmake` | Download dependencies from GitHub at configure time | pip / uv | ❌ CUDA + cuBLAS are installed |
| `PackageProject.cmake`, `Doxygen.cmake`, `Emscripten.cmake`, `VCEnvironment.cmake` | Install/packaging, API docs, WebAssembly, Windows MSVC setup | `uv build` / publish | ❌ |

### Design note: INTERFACE libraries vs functions

The template puts flags into "fake" libraries:

```cmake
add_library(myproject_warnings INTERFACE)       # holds flags, no code
# ...set flags on it...
target_link_libraries(my_app PRIVATE myproject_warnings)  # my_app inherits the flags
```

Good with 20 targets. With one target, calling a function directly (what this repo does) is simpler and equivalent.

## Why not clone the whole template?

1. **It's built for C++, not CUDA.** `project()` enables only C and CXX; its CUDA warning list is a stub.
2. **Its defaults get in the way:** clang-tidy + cppcheck on every build, sanitizers always on
   (also in benchmark builds), dependencies downloaded at configure time, requires Ninja and CMake ≥ 3.29.
3. **~2,000 lines you don't understand yet**, much of it for Windows, WebAssembly, packaging, fuzzing.
4. **Disabled options don't disappear**; you still have to read past them.

Use it as a **menu**: copy a piece when you need it. The full template makes sense for a larger
C++ *application* (many libraries, tests, dependencies, cross-platform).

## This repo's CMake

```
CMakeLists.txt
  ├─ sets CMAKE_CUDA_ARCHITECTURES=86   (BEFORE project(), see lessons learned)
  ├─ C++20 / CUDA 20, extensions OFF    (-std=c++20, not gnu++20)
  ├─ find_package(CUDAToolkit)          → CUDA::cudart, CUDA::cublas
  ├─ add_executable(sgemm src/main.cu)  + -lineinfo
  ├─ set_project_warnings(...)          from cmake/CompilerWarnings.cmake
  └─ enable_sanitizers(...)             from cmake/Sanitizers.cmake

CMakePresets.json
  release  → Release, for benchmarking
  debug    → Debug + warnings as errors
  asan     → debug + ASan/UBSan   (run with ASAN_OPTIONS=protect_shadow_gap=0)
```

```
cmake --preset release && cmake --build --preset release
```
