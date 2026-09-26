// Your harness goes here. Suggested order:
//   1. CUDA_CHECK / CUBLAS_CHECK error macros
//   2. Parse the kernel number from argv
//   3. Allocate + randomly fill A, B, C on host; copy to device
//   4. Run cuBLAS once -> reference result (careful: cuBLAS is column-major)
//   5. Run your kernel, compare with reference (tolerance, not ==)
//   6. Time with cudaEvent: warm up, then average N runs -> GFLOPS
//
// Kernels live in src/kernels/, one file per kernel (1_naive.cuh, ...).

int main() { return 0; }
