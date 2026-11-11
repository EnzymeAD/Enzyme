// The accuracy model's Ozaki-II product (simulateOzakiIIDgemm) against the
// runtime on the operands the model samples: tiles of the shape it simulates
// (16 x 16 with K = 2048, which takes the rectangular arm and the chunked
// strided sum of squares; K = 512, the unchunked one; and the square entry at
// N = 64), both operands row-major, elements log-uniform over the 40 binades
// below the largest (the tolerance path) or uniform (the budget path), at
// every modulus count the compiler proposes. Every product is the same bit for
// bit.
//
// REQUIRES: cuda-runtime
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -frounding-math \
// RUN:     -I%S/../../../Poseidon -c \
// RUN:     %S/../../../Poseidon/Techniques/Matmul/MatmulSimulation.cpp \
// RUN:     -o %t/sim.o
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -I%S/../../../Poseidon/Techniques/Matmul %s -x none %t/sim.o \
// RUN:     %poseidon_rt_lib -L%cuda_lib_dir -lcublas -lcudart -lstdc++ -lm \
// RUN:     -o %t/exe
// RUN: %t/exe | FileCheck %s

// CHECK: rect16x16x2048 log40: 7 moduli counts, 1792 products, 0 differ
// CHECK-NEXT: rect16x16x2048 uniform: 7 moduli counts, 1792 products, 0 differ
// CHECK-NEXT: rect16x16x512 log40: 7 moduli counts, 1792 products, 0 differ
// CHECK-NEXT: rect16x16x512 uniform: 7 moduli counts, 1792 products, 0 differ
// CHECK-NEXT: square64 log40: 7 moduli counts, 28672 products, 0 differ
// CHECK-NEXT: square64 uniform: 7 moduli counts, 28672 products, 0 differ

#include "MatmulSimulation.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int N, int lda, int ldb,
                                       int ldc, int transA, int transB,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli);
extern "C" void __poseidon_ozaki_dgemm_ex(
    double *C, const double *A, const double *B, int M, int Ncols, int K,
    int lda, int ldb, int ldc, int aColMajor, int bColMajor, int cColMajor,
    double alpha, double beta, cudaStream_t stream, int num_moduli);

static uint64_t g_state = 0x2545f4914f6cdd1dull;
static double uniform() { // [0, 1), 53 bits
  g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
  return (double)(g_state >> 11) * 0x1.0p-53;
}

// A row-major rows x cols operand with largest magnitude about `scale`.
static std::vector<double> operand(int rows, int cols, bool log40,
                                   double scale) {
  std::vector<double> v((size_t)rows * cols);
  for (double &x : v) {
    double mag =
        log40 ? scale * std::exp2(-40.0 * uniform()) : scale * uniform();
    x = uniform() < 0.5 ? -mag : mag;
  }
  return v;
}

static bool sameBits(double a, double b) {
  uint64_t x, y;
  std::memcpy(&x, &a, 8);
  std::memcpy(&y, &b, 8);
  return x == y;
}

static void run(const char *name, bool square, int M, int N, int K) {
  for (bool log40 : {true, false}) {
    std::vector<double> A = operand(M, K, log40, 0.5);
    std::vector<double> B = operand(K, N, log40, 2.0);
    double *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * 8);
    cudaMalloc(&dB, B.size() * 8);
    cudaMalloc(&dC, (size_t)M * N * 8);
    cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    std::vector<double> C((size_t)M * N), P((size_t)M * N);
    size_t products = 0, differ = 0;
    int counts = 0;
    for (unsigned nm = 8; nm <= 14; ++nm, ++counts) {
      if (square)
        __poseidon_ozaki_dgemm(dC, dA, dB, M, M, M, M, 0, 0, 1.0, 0.0, nullptr,
                               nm);
      else
        __poseidon_ozaki_dgemm_ex(dC, dA, dB, M, N, K, K, N, N, 0, 0, 0, 1.0,
                                  0.0, nullptr, nm);
      cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost);
      poseidon::simulateOzakiIIDgemm(
          nm, {A.data(), (unsigned)K, (unsigned)M, true},
          {B.data(), (unsigned)N, (unsigned)N, false}, (unsigned)K, P.data());
      for (size_t i = 0; i < C.size(); ++i, ++products)
        differ += !sameBits(C[i], P[i]);
    }
    printf("%s %s: %d moduli counts, %zu products, %zu differ\n", name,
           log40 ? "log40" : "uniform", counts, products, differ);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
}

int main() {
  run("rect16x16x2048", false, 16, 16, 2048);
  run("rect16x16x512", false, 16, 16, 512);
  run("square64", true, 64, 64, 64);
  return 0;
}
