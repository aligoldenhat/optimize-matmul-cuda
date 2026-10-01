#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// Kernel 3: shared-memory cache-blocking. C = alpha * A * B + beta * C, row-major.
//   A: M x K,  B: K x N,  C: M x N
//
// Each thread block computes one BLOCKSIZE x BLOCKSIZE tile of C. It slides
// along K in steps of BLOCKSIZE: every step, the block cooperatively copies one
// tile of A and one tile of B from slow global memory into fast shared memory
// (each thread loads one element of each), then every thread does BLOCKSIZE
// multiply-adds reading only from shared memory.
//
// Each value is now loaded from global memory once per block instead of once
// per thread, which cuts global memory traffic by a factor of BLOCKSIZE.
//
// Limitation (same as the article): there are no bounds checks, so M, N and K
// must be multiples of BLOCKSIZE. The launcher below enforces that. (That is also
// why M is unused inside the kernel: the grid size already encodes it.)
template <int BLOCKSIZE>
__global__ void sgemm_shared_mem_block([[maybe_unused]] int M, int N, int K, float alpha,
                                       const float* A, const float* B, float beta, float* C) {
    // Which BLOCKSIZE x BLOCKSIZE tile of C this block computes.
    const int cRow = static_cast<int>(blockIdx.x);
    const int cCol = static_cast<int>(blockIdx.y);

    // Shared memory: one copy per block, visible to all its threads.
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // The block is 1D (BLOCKSIZE * BLOCKSIZE threads); split threadIdx.x into a
    // (row, col) inside the tile. col = threadIdx.x % BLOCKSIZE makes neighboring
    // threads touch neighboring addresses, so global loads are coalesced.
    const int threadCol = static_cast<int>(threadIdx.x) % BLOCKSIZE;
    const int threadRow = static_cast<int>(threadIdx.x) / BLOCKSIZE;

    // Move the pointers to this block's starting tile.
    A += cRow * BLOCKSIZE * K;                         // row = cRow tile, col = 0
    B += cCol * BLOCKSIZE;                             // row = 0, col = cCol tile
    C += (cRow * BLOCKSIZE * N) + (cCol * BLOCKSIZE);  // row = cRow tile, col = cCol tile

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Each thread loads one element of the A tile and one of the B tile.
        As[(threadRow * BLOCKSIZE) + threadCol] = A[(threadRow * K) + threadCol];
        Bs[(threadRow * BLOCKSIZE) + threadCol] = B[(threadRow * N) + threadCol];

        // Wait until the whole tile is in shared memory before anyone reads it.
        __syncthreads();

        // Slide both tiles along K for the next iteration.
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // Partial dot product over this tile, from shared memory only.
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[(threadRow * BLOCKSIZE) + dotIdx] * Bs[(dotIdx * BLOCKSIZE) + threadCol];
        }

        // Wait again, so a fast thread doesn't overwrite the tiles for the next
        // iteration while a slower thread is still reading them.
        __syncthreads();
    }

    C[(threadRow * N) + threadCol] = (alpha * tmp) + (beta * C[(threadRow * N) + threadCol]);
}

// Host-side launcher: checks the sizes, picks the block/grid size, launches.
inline void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, const float* A,
                                       const float* B, float beta, float* C) {
    constexpr int kBlockSize = 32;

    if (M % kBlockSize != 0 || N % kBlockSize != 0 || K % kBlockSize != 0) {
        std::fprintf(stderr,
                     "sgemm_shared_mem_block: M, N, K must be multiples of %d (got %d, %d, %d)\n",
                     kBlockSize, M, N, K);
        std::exit(EXIT_FAILURE);
    }

    const dim3 block(kBlockSize * kBlockSize);  // 1024 threads, 1D
    const dim3 grid(static_cast<unsigned int>(M / kBlockSize),
                    static_cast<unsigned int>(N / kBlockSize));
    sgemm_shared_mem_block<kBlockSize><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
