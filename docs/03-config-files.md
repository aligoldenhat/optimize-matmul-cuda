# What's inside each config file

Compared across [cmake_template](https://github.com/cpp-best-practices/cmake_template),
[NVIDIA/cccl](https://github.com/NVIDIA/cccl), [llama.cpp](https://github.com/ggml-org/llama.cpp)
and this repo. Snapshots read in September 2026; the upstream files change over time.

## `.clang-tidy`: the linter (like ruff's `select` / `ignore`)

Checks come in **groups**. Every repo uses the same pattern: turn a group on with a
wildcard, then remove individual checks with a leading `-`:

```yaml
Checks: >
  bugprone-*,                            # enable the whole group
  -bugprone-easily-swappable-parameters  # remove one check from it
```

### The groups

| Group | Finds | This repo | llama.cpp | cccl | template |
|---|---|---|---|---|---|
| `bugprone-*` | likely bugs | ✅ | ✅ | ✅ | ✅ (via `*`) |
| `performance-*` | needless copies, slow patterns | ✅ | ✅ | ✅ | ✅ |
| `readability-*` | confusing code | ✅ | ✅ | ✅ | ✅ |
| `modernize-*` | old C++ → modern C++ | ✅ | ❌ | ✅ | ✅ |
| `cppcoreguidelines-*` | C++ Core Guidelines | ✅ | ❌ | ✅ | ✅ |
| `clang-analyzer-*` | path-sensitive analysis: null deref, leaks, use-after-free | ✅ | ✅ | ✅ | ✅ |
| `misc-*` | assorted | ✅ | ✅ | ✅ | ✅ |
| `portability-*` | platform-dependent code | ✅ | ✅ | ✅ | ✅ |
| `cert-*`, `google-*`, `concurrency-*` | security rules, Google style, threading | ❌ | ❌ | ✅ | ✅ |

Full list: https://clang.llvm.org/extra/clang-tidy/checks/list.html

### Three styles of choosing checks

- **cmake_template**: `*` (everything), then remove ~15 whole groups/checks. Strictest, noisiest.
- **cccl**: `-*` first (everything off), then add groups back explicitly. Most explicit.
- **llama.cpp / this repo**: list the wanted groups. Middle ground, good for learning.

### Checks everyone disables

All of them turn off some of: `readability-magic-numbers`, `readability-identifier-length`,
`bugprone-easily-swappable-parameters`, `cppcoreguidelines-pro-bounds-pointer-arithmetic`.
In GEMM code these fire on every line (`M, N, K`, tile sizes like `64`, `A[row * K + k]`).
The reason for each disabled check in this repo is written in `.clang-tidy`.

### Other keys

| Key | Meaning |
|---|---|
| `WarningsAsErrors` | which checks fail the run |
| `HeaderFilterRegex` | which headers to report on (so CUDA's own headers stay quiet) |
| `CheckOptions` | per-check settings (template: allow variable names `x`, `y`, `z` despite the length rule) |
| `FormatStyle: none` | don't reformat when clang-tidy applies a fix |

## `.clang-format`: the formatter (like ruff format)

```yaml
BasedOnStyle: Google       # start from a named style: Google, LLVM, Mozilla, WebKit, ...
ColumnLimit: 100           # line length
IndentWidth: 4
PointerAlignment: Left     # float* A  vs  float *A
AllowShortFunctionsOnASingleLine: Inline
IncludeBlocks: Regroup     # sort and group #includes
```

| Repo | Approach | Size |
|---|---|---|
| cccl | `BasedOnStyle: LLVM` + many overrides, 120 columns | ~200 lines |
| llama.cpp | spells out nearly every option | ~170 lines |
| cmake_template | spells out options | ~100 lines |
| this repo | `BasedOnStyle: Google` + a few overrides | 8 options |

Both approaches work. **To learn an option:** change it, run `clang-format -i file.cu`, look at `git diff`.
All options: https://clang.llvm.org/docs/ClangFormatStyleOptions.html

## `.clangd`: the language server (like ty's editor settings)

cccl's `.clangd` is the best CUDA reference. Its structure:

```yaml
If:                        # 1. only for matching files
  PathMatch: .*\.cuh?
CompileFlags:
  Add: [-xcuda]            # 2. flags to add / remove before parsing
---                        # separates blocks
CompileFlags:
  Remove: [-Xcompiler*, ...]
Diagnostics:
  Suppress: [...]          # 3. silence specific false errors
```

**Why `Remove`?** `compile_commands.json` contains **nvcc** commands, but clangd is **clang**,
which doesn't know nvcc's flags. cccl strips ~20 of them; this repo strips the ones it can produce,
grouped by kind (driver, GPU arch, codegen/debug, nvcc warnings).

**What `Add` does here:**

| Flag | Why |
|---|---|
| `-xcuda` (in the `.cuh?` block) | Headers have no entry in `compile_commands.json`; without this clangd may parse `.cuh` as plain C++ and flag `__global__`, `threadIdx` |
| `--cuda-path=/usr/local/cuda` | where CUDA is |
| `--cuda-gpu-arch=sm_86` | which GPU to target (replaces nvcc's `--generate-code`) |
| `--no-cuda-version-check`, `-Wno-unknown-cuda-version` | CUDA 13.1 is newer than clang officially knows |
| `-ferror-limit=0` | show all errors, not just the first 20 |
| `CompilationDatabase: build/debug` | which preset's `compile_commands.json` to use |

Options reference: https://clangd.llvm.org/config

## `.pre-commit-config.yaml`: checks before every commit

Same pre-commit tool as in Python. cccl runs, among others: clang-format, ruff, mypy, shellcheck,
codespell, JSON/YAML/TOML validators, trailing-whitespace fixes. llama.cpp's is small
(whitespace, YAML check, flake8). This repo doesn't have one yet; a clang-format hook would be the first step.

## `CMakePresets.json`: named build configurations

Like having several named profiles instead of remembering `-D` flags:

```json
{ "name": "debug", "inherits": "base",
  "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug", "WARNINGS_AS_ERRORS": "ON" } }
```

`"hidden": true` presets are building blocks others `inherit` from. The template's presets
cover Windows/MSVC and Linux/macOS with gcc and clang; cccl and llama.cpp have them too.
