// Your harness goes here. Suggested order:
//   1. CUDA_CHECK / CUBLAS_CHECK error macros
//   2. Parse the kernel number from argv
//   3. Allocate + randomly fill A, B, C on host; copy to device
//   4. Run cuBLAS once -> reference result (careful: cuBLAS is column-major)
//   5. Run your kernel, compare with reference (tolerance, not ==)
//   6. Time with cudaEvent: warm up, then average N runs -> GFLOPS
//
// Kernels live in src/kernels/, one file per kernel (1_naive.cuh, ...).

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                               \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                                 \
    do {                                                                                   \
        const cublasStatus_t status_ = (call);                                             \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                            \
            std::fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__, \
                         static_cast<int>(status_));                                       \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

void fill_random(std::vector<float>& m, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : m) {
        x = dist(rng);
    }
}

void cpu_sgemm(int M, int N, int K, float alpha, const std::vector<float>& A,
               const std::vector<float>& B, float beta, std::vector<float>& C) {
    const auto m = static_cast<size_t>(M);
    const auto n = static_cast<size_t>(N);
    const auto k_dim = static_cast<size_t>(K);

    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            double sum = 0.0;
            for (size_t k = 0; k < k_dim; ++k) {
                sum += static_cast<double>(A[(row * k_dim) + k]) *
                       static_cast<double>(B[(k * n) + col]);  // row-major
            }
            const size_t idx = (row * n) + col;
            C[idx] = (alpha * static_cast<float>(sum)) + (beta * C[idx]);
        }
    }
}

bool matrices_match(const std::vector<float>& ref, const std::vector<float>& out, int N) {
    const float tolerance = 1e-3f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(ref[i] - out[i]);
        if (diff > tolerance * (1.0f + std::fabs(ref[i]))) {
            std::printf("Mismatch at row %zu, col %zu: expected %f, got %f\n",
                        i / static_cast<size_t>(N), i % static_cast<size_t>(N),
                        static_cast<double>(ref[i]), static_cast<double>(out[i]));
            return false;
        }
    }
    return true;
}

int main() {
    // Non-square on purpose: catches mixed-up M/N/K.
    const int M = 128;
    const int N = 256;
    const int K = 64;

    const size_t size_A = static_cast<size_t>(M) * K;
    const size_t size_B = static_cast<size_t>(K) * N;
    const size_t size_C = static_cast<size_t>(M) * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C);

    std::mt19937 rng(42);
    fill_random(h_A, rng);
    fill_random(h_B, rng);
    fill_random(h_C, rng);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), size_C * sizeof(float), cudaMemcpyHostToDevice));

    std::printf("Allocated and copied A(%dx%d) B(%dx%d) C(%dx%d)\n", M, K, K, N, M, N);

    const float alpha = 1.0f;
    const float beta = 0.5f;

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
                             &beta, d_C, N));

    std::vector<float> h_C_cublas(size_C);
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_C_cpu = h_C;
    cpu_sgemm(M, N, K, alpha, h_A, h_B, beta, h_C_cpu);

    if (matrices_match(h_C_cpu, h_C_cublas, N)) {
        std::printf("cuBLAS matches CPU reference\n");
    } else {
        printf("cuBLAS does NOT match CPU reference\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
