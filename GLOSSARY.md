# CUDA Matmul Glossary

Notes for reading https://siboehm.com/articles/22/CUDA-MMM

## Names

**GEMM**: **GE**neral **M**atrix **M**ultiply:

```
C = α·(A × B) + β·C
```

- A is M×K, B is K×N, C is M×N
- α (alpha), β (beta) are plain numbers (scalars). With α=1, β=0 it's just `C = A × B`. β·C lets you add to C instead of overwriting it.
- "General" = the matrices have no special structure (not symmetric, triangular, ...)

**SGEMM**: GEMM in **S**ingle precision (`float`, 32-bit). Prefixes come from BLAS:

| Prefix | Type |
|---|---|
| S | float32 |
| D | double (float64) |
| H | half (float16) |
| C / Z | complex float / complex double |

**BLAS / cuBLAS**: BLAS is a standard set of linear-algebra routines. cuBLAS is NVIDIA's heavily optimized BLAS for GPUs, and the benchmark to try to get close to.

## FLOP vs FLOPS

- **FLOP** = one floating-point operation (one add or one multiply)
- **FLOPS** = FLOPs per second (a speed)
- GFLOPS = 10⁹ per second, TFLOPS = 10¹² per second

**Work in a matmul:** each of the M·N outputs is a dot product of length K, which is K multiplies + K adds:

```
total FLOPs = 2 · M · N · K
```

Example: M=N=K=4096 → 2 · 4096³ ≈ **137 GFLOP**

**Measured performance** = FLOPs / time
Example: 50 ms → 137e9 / 0.05 ≈ **2.7 TFLOPS**

## Hardware limits (RTX 3050 Laptop, sm_86)

1. **Compute peak**: `SMs × FP32 cores/SM × 2 (a fused multiply-add = 2 FLOPs) × clock`
   16 × 128 × 2 × ~1.5 GHz ≈ **~6 TFLOPS** (laptop clocks vary)
2. **Memory bandwidth** (VRAM ↔ SMs): **~192 GB/s**

**Arithmetic intensity** = FLOPs / bytes moved from memory

- Low intensity → **memory-bound** (the naive kernel)
- High intensity → **compute-bound** (the goal)

This is the **roofline model**. Every optimization in the article comes down to:
**load each number from slow memory once, reuse it many times from fast memory** (GMEM → SMEM → registers).

## Terms

| Term | Meaning |
|---|---|
| **SM** (Streaming Multiprocessor) | A GPU "core cluster". RTX 3050 Laptop has 16. Each thread block runs on one SM |
| **Warp** | 32 threads executing the same instruction together. The real unit of execution |
| **Global memory / GMEM** | VRAM: big and slow |
| **Shared memory / SMEM** | Small (~100 KB per SM), fast, shared by a block, managed by you |
| **Registers** | Fastest memory, private to each thread |
| **Coalescing** | A warp's 32 threads reading neighboring addresses → merged into one memory transaction |
| **Tiling** | Splitting matrices into blocks (tiles) that fit in faster memory |
| **Occupancy** | Active warps per SM ÷ maximum possible. Helps hide latency, but it's not everything |
| **Bank conflicts** | SMEM has 32 banks. Threads in a warp hitting the same bank at different addresses → accesses get serialized |
| **Vectorized loads** (`float4`) | Load 4 floats in one instruction |
| **Row-major / column-major** | 2D → 1D memory layout. C/C++ = row-major, cuBLAS = column-major (watch out when checking results) |
| **Warptiling** | Tiling at the warp level too, on top of block and thread tiling |

## Kernel progression: which bottleneck each kernel fixes

Ask at each step: *which memory is the data coming from, and how many times is each value loaded?*

1. **Naive**: uncoalesced GMEM reads
2. **GMEM coalescing**: fixes coalescing, but still reads GMEM for every multiply
3. **SMEM caching (tiling)**: reuses data from SMEM; now SMEM becomes the bottleneck
4. **1D block tiling**: each thread computes several outputs, so values are reused from registers
5. **2D block tiling**: even more register reuse
6. **Vectorized memory access**: `float4` loads, transposed A in SMEM
7. **Autotuning**: search for the best tile sizes
8. **Warptiling**: tiling at the warp level as well

## Formulas to keep handy

```
FLOPs        = 2·M·N·K
TFLOPS       = FLOPs / seconds / 1e12
% of cuBLAS  = my_TFLOPS / cublas_TFLOPS
bytes (min)  = 4·(M·K + K·N + M·N)       # float32, each matrix touched once
intensity    = FLOPs / bytes
```
