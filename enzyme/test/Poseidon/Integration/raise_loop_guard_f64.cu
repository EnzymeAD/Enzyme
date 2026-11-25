// A product whose threads return early outside the guard
// `row >= ROWS || col >= COLS` on their global indices, profiled and raised
// onto the tensor cores. A block raises only when every one of its threads
// passes the guard; any other block runs the guard and the loop as written.
// guard_exact_kernel's grid of 16 x 16 blocks tiles its 32 x 32 extents, so
// every block raises. guard_partial_kernel's 40 x 32 extents leave its last
// row of blocks half outside the guard: the full blocks raise, the last one
// runs the loop, which gives the FP64 product, and no element outside the
// extents is written.
//
// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 \
// RUN:     %loadClangPoseidonEnzyme -mllvm --poseidon-profile-generate -mllvm -poseidon-cache=%t/cg \
// RUN:     -I%poseidon_profile_inc %s -x none %poseidon_profile_cuda_lib %poseidon_profile_lib \
// RUN:     -L%cuda_lib_dir -lcudart -lstdc++ -lm -o %t/gen.exe
// RUN: env POSEIDON_PROFILE_DIR=%t/profiles %t/gen.exe | FileCheck --check-prefix=PROFILED %s
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 %loadClangPoseidon \
// RUN:     -mllvm -poseidon-profile-use=%t/profiles -mllvm -poseidon-cache=%t/c \
// RUN:     -mllvm -poseidon-enable-matmul -mllvm -poseidon-matmul-library-calls=false \
// RUN:     -mllvm -poseidon-enable-rewrite=false -mllvm -poseidon-enable-precision=false \
// RUN:     -mllvm -poseidon-cost-model=%gpu_cost_model -mllvm -poseidon-print \
// RUN:     -I%poseidon_profile_inc %s -L%cuda_lib_dir -lcudart -o %t/opt.exe 2>&1 | FileCheck --check-prefix=RAISE %s
// RUN: %t/opt.exe | FileCheck --check-prefix=RAISED %s

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;

constexpr int COLS = 32, K = 16, ALLOC_ROWS = 48;

template <int ROWS>
__device__ __attribute__((noinline)) void
guard_body(const double *A, const double *B, double *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= ROWS || col >= COLS)
    return;
  const double *arow = A + (size_t)row * K;
  const double *bcol = B + col;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < K; ++k)
    acc += arow[k] * bcol[(size_t)k * COLS];
  C[(size_t)row * COLS + col] = acc;
}

__global__ void guard_exact_kernel(const double *A, const double *B,
                                   double *C) {
  __poseidon_fp_optimize<void>((void *)guard_body<32>, enzyme_const, A,
                               enzyme_const, B, enzyme_const, C);
}

__global__ void guard_partial_kernel(const double *A, const double *B,
                                     double *C) {
  __poseidon_fp_optimize<void>((void *)guard_body<40>, enzyme_const, A,
                               enzyme_const, B, enzyme_const, C);
}

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static const char *errorClass(double maxRel) {
  return maxRel <= 1e-12 ? "FP64" : maxRel <= 1e-2 ? "TC" : "WRONG";
}

// Launches `kernel` on a grid of 16 x 16 blocks covering `rows` x COLS and
// prints, per row of blocks, the error class of its elements against the FP64
// product, and whether the rows from `rows` on kept their value.
static void run(const char *name, void (*kernel)(const double *,
                                                 const double *, double *),
                int rows) {
  double hA[ALLOC_ROWS * K], hB[K * COLS], hC[ALLOC_ROWS * COLS];
  for (int i = 0; i < ALLOC_ROWS * K; ++i)
    hA[i] = 1.0 + 0.3 * (i % 5);
  for (int i = 0; i < K * COLS; ++i)
    hB[i] = 0.5 + 0.1 * (i % 7);
  for (int i = 0; i < ALLOC_ROWS * COLS; ++i)
    hC[i] = -1.0;

  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CK(cudaMalloc(&dA, sizeof(hA)));
  CK(cudaMalloc(&dB, sizeof(hB)));
  CK(cudaMalloc(&dC, sizeof(hC)));
  CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, hC, sizeof(hC), cudaMemcpyHostToDevice));
  const int blockRows = (rows + 15) / 16;
  kernel<<<dim3(COLS / 16, blockRows), dim3(16, 16)>>>(dA, dB, dC);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));

  printf("%s %dx%d:", name, rows, COLS);
  for (int by = 0; by < blockRows; ++by) {
    double maxRel = 0.0;
    for (int r = by * 16; r < by * 16 + 16 && r < rows; ++r)
      for (int c = 0; c < COLS; ++c) {
        double expected = 0.0;
        for (int k = 0; k < K; ++k)
          expected += hA[r * K + k] * hB[k * COLS + c];
        maxRel = std::fmax(maxRel,
                           std::fabs(hC[r * COLS + c] - expected) / expected);
      }
    printf(" %s", errorClass(maxRel));
  }
  bool untouched = true;
  for (int i = rows * COLS; i < ALLOC_ROWS * COLS; ++i)
    untouched &= hC[i] == -1.0;
  printf(", beyond %s\n", untouched ? "untouched" : "WRITTEN");
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
}

int main() {
  run("exact", guard_exact_kernel, 32);
  run("partial", guard_partial_kernel, 40);
  return 0;
}

// RAISE-COUNT-2: Applying solution for matmul[0] -> {{.*}}wmma m16n16k{{8|16}}

// PROFILED: exact 32x32: FP64 FP64, beyond untouched
// PROFILED: partial 40x32: FP64 FP64 FP64, beyond untouched

// RAISED: exact 32x32: TC TC, beyond untouched
// RAISED: partial 40x32: TC TC FP64, beyond untouched
