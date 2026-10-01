#pragma once

#include <cuda_runtime.h>

// Kernel 1: naive SGEMM. C = alpha * A * B + beta * C, all row-major.
//   A: M x K,  B: K x N,  C: M x N
//
// One thread computes one element C[row][col]. It walks along row `row` of A
// and column `col` of B, multiplying and summing K pairs of numbers.
//
// Thread -> element mapping (as in the article): threadIdx.x picks the ROW.
// Neighboring threads in a warp therefore handle neighboring rows, which sit
// K floats apart in memory. That is slow (uncoalesced global loads), and it is
// exactly what kernel 2 fixes.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C) {
    const int row = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int col = static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y);

    // The grid is rounded up, so some threads fall outside the matrix.
    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[(row * K) + k] * B[(k * N) + col];
    }
    C[(row * N) + col] = (alpha * sum) + (beta * C[(row * N) + col]);
}

// Host-side launcher: picks the block/grid size and launches the kernel.
// Works for any M, N, K (the bounds check above handles sizes that are not
// multiples of the block size).
inline void run_sgemm_naive(int M, int N, int K, float alpha, const float* A, const float* B,
                            float beta, float* C) {
    constexpr int kBlockSize = 32;

    const dim3 block(kBlockSize, kBlockSize);  // 32 * 32 = 1024 threads per block
    // Round up, so the grid covers the whole matrix: ceil(M / 32) x ceil(N / 32) blocks.
    const dim3 grid(static_cast<unsigned int>((M + kBlockSize - 1) / kBlockSize),
                    static_cast<unsigned int>((N + kBlockSize - 1) / kBlockSize));
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
