#!/usr/bin/env bash
# Run clang-format (check only) and clang-tidy on the project's CUDA sources.
#
#   scripts/lint.sh          check formatting + run clang-tidy
#   scripts/lint.sh --fix    reformat files in place, then run clang-tidy
#
# Why not `clang-tidy -p build/debug`? compile_commands.json holds nvcc
# commands, and clang-tidy (unlike clangd) doesn't read .clangd to strip the
# nvcc-only flags. So we pass clang-compatible flags after `--` instead.
# Keep these in sync with the Add: section of .clangd.
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer the newest installed version (clang 21 can't fully parse CUDA 13 + GCC 15).
pick() { for c in "$@"; do command -v "$c" >/dev/null && { echo "$c"; return; }; done; echo "$1"; }
CLANG_TIDY=${CLANG_TIDY:-$(pick clang-tidy-22 clang-tidy)}
CLANG_FORMAT=${CLANG_FORMAT:-$(pick clang-format-22 clang-format)}

CUDA_FLAGS=(
    -xcuda -std=c++20 -Isrc
    --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_86
    --no-cuda-version-check -Wno-unknown-cuda-version
)

mapfile -t all_files < <(find src \( -name '*.cu' -o -name '*.cuh' \) | sort)
# Headers are checked through the .cu files that include them (HeaderFilterRegex).
mapfile -t cu_files < <(printf '%s\n' "${all_files[@]}" | grep '\.cu$')

# Run both tools even if the first one fails; fail at the end if either did.
status=0

echo "== $CLANG_FORMAT"
if [[ "${1:-}" == "--fix" ]]; then
    "$CLANG_FORMAT" -i "${all_files[@]}" || status=1
else
    "$CLANG_FORMAT" --dry-run --Werror "${all_files[@]}" || status=1
fi

echo "== $CLANG_TIDY"
"$CLANG_TIDY" --quiet "${cu_files[@]}" -- "${CUDA_FLAGS[@]}" || status=1

exit "$status"
