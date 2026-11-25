// Every modulus count the Ozaki-II runtime accepts (3..14), each of which runs
// its own instantiation of the residue and CRT kernels, against the accuracy
// model's simulation of the published scheme (simulateOzakiIIDgemm), bit for
// bit: __poseidon_ozaki_dgemm in every A/B/C storage order, with beta = 0.5
// on one, at 37 x 50 x 61, at 20 x 24 x 1100, whose strided operands take the
// chunked sum of squares, and at the square 48 x 48 x 48. Rows and columns of
// the operands span 2^-30..2^30, with every 17th element of a row of A 10^6
// times larger. The same on the operands the accuracy model samples, on the
// tile it simulates (16 x 16 with K = 2048), both row-major, elements
// log-uniform over the 40 binades below the largest (the tolerance path) or
// uniform (the budget path), at every modulus count the compiler proposes.
//
// REQUIRES: cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -frounding-math \
// RUN:     -I%S/../../../Poseidon -c \
// RUN:     %S/../../../Poseidon/Techniques/Matmul/MatmulSimulation.cpp \
// RUN:     -o %t/sim.o
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=off \
// RUN:     -I%S/../../../Poseidon/Techniques/Matmul -I%poseidon_runtime_inc \
// RUN:     %s -x none %t/sim.o \
// RUN:     %poseidon_rt_lib -L%cuda_lib_dir -lcublas -lcudart -lstdc++ -lm \
// RUN:     -o %t/exe
// RUN: %t/exe | FileCheck --match-full-lines %s

// CHECK: ozaki 37x50x61: 8 layouts, 177600 products, 0 differ
// CHECK-NEXT: ozaki 20x24x1100: 8 layouts, 46080 products, 0 differ
// CHECK-NEXT: ozaki 48x48x48: 8 layouts, 221184 products, 0 differ
// CHECK-NEXT: model 16x16x2048 log40: 7 moduli, 1792 products, 0 differ
// CHECK-NEXT: model 16x16x2048 uniform: 7 moduli, 1792 products, 0 differ

#include "MatmulSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#include "GemmDispatch.h"

using poseidon::simulateOzakiIIDgemm;

namespace {

// The operands in their logical row-major shapes: A (M x K) with a scale per
// row, B (K x N) with a scale per column, C0 (M x N).
struct Operands {
  int M, N, K;
  std::vector<double> A, B, C0;
  Operands(int M, int N, int K) : M(M), N(N), K(K) {
    std::mt19937_64 g((uint64_t)M * 7919 + (uint64_t)N * 31 + K);
    std::uniform_real_distribution<double> u(-1.0, 1.0), e(-30.0, 30.0);
    A.resize((size_t)M * K);
    B.resize((size_t)K * N);
    C0.resize((size_t)M * N);
    for (int r = 0; r < M; ++r) {
      double s = std::ldexp(1.0, (int)e(g));
      for (int k = 0; k < K; ++k)
        A[(size_t)r * K + k] = u(g) * s * (k % 17 == 0 ? 1e6 : 1.0);
    }
    for (int c = 0; c < N; ++c) {
      double s = std::ldexp(1.0, (int)e(g));
      for (int k = 0; k < K; ++k)
        B[(size_t)k * N + c] = u(g) * s;
    }
    for (double &x : C0)
      x = u(g);
  }
};

// X (rows x cols, row-major) stored column-major or row-major.
std::vector<double> store(const std::vector<double> &X, int rows, int cols,
                          int colMajor) {
  std::vector<double> S(X.size());
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      S[colMajor ? (size_t)c * rows + r : (size_t)r * cols + c] =
          X[(size_t)r * cols + c];
  return S;
}

// A * B before the epilogue, row-major, as the runtime computes it: it reads a
// vector contiguously when its storage runs along k.
std::vector<double> simulate(const Operands &o, int aColMajor, int bColMajor,
                             unsigned nm) {
  std::vector<double> P((size_t)o.M * o.N);
  std::vector<double> As = store(o.A, o.M, o.K, aColMajor);
  std::vector<double> Bs = store(o.B, o.K, o.N, bColMajor);
  simulateOzakiIIDgemm(
      nm,
      {As.data(), (unsigned)(aColMajor ? o.M : o.K), (unsigned)o.M, !aColMajor},
      {Bs.data(), (unsigned)(bColMajor ? o.K : o.N), (unsigned)o.N,
       (bool)bColMajor},
      (unsigned)o.K, P.data());
  return P;
}

bool sameBits(double a, double b) {
  uint64_t x, y;
  std::memcpy(&x, &a, 8);
  std::memcpy(&y, &b, 8);
  return x == y;
}

// The number of products that differ, or 1 after a CUDA error.
size_t run(int M, int N, int K) {
  const Operands o(M, N, K);
  double *dA, *dB, *dC;
  cudaMalloc(&dA, o.A.size() * 8);
  cudaMalloc(&dB, o.B.size() * 8);
  cudaMalloc(&dC, o.C0.size() * 8);
  const int layouts = 8;
  size_t products = 0, differ = 0;
  for (int lay = 0; lay < layouts; ++lay) {
    const int ac = lay & 1, bc = (lay >> 1) & 1, cc = (lay >> 2) & 1;
    const double beta = lay == 5 ? 0.5 : 0.0;
    const std::vector<double> A = store(o.A, M, K, ac),
                              B = store(o.B, K, N, bc),
                              C0 = store(o.C0, M, N, cc);
    cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    std::vector<double> C(C0.size());
    for (unsigned nm = 3; nm <= 14; ++nm) {
      cudaMemcpy(dC, C0.data(), C0.size() * 8, cudaMemcpyHostToDevice);
      __poseidon_ozaki_dgemm(dC, dA, dB, M, N, K, ac ? M : K, bc ? K : N,
                             cc ? M : N, ac, bc, cc, 1.0, beta, nullptr, nm);
      if (cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        printf("ozaki %dx%dx%d: CUDA error\n", M, N, K);
        return 1;
      }
      const std::vector<double> P = simulate(o, ac, bc, nm);
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c, ++products) {
          const size_t i = cc ? (size_t)c * M + r : (size_t)r * N + c;
          const double ab = P[(size_t)r * N + c];
          const double want = beta == 0.0 ? ab : std::fma(beta, C0[i], ab);
          differ += !sameBits(C[i], want);
        }
    }
  }
  printf("ozaki %dx%dx%d: %d layouts, %zu products, %zu differ\n", M, N, K,
         layouts, products, differ);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return differ;
}

uint64_t g_state = 0x2545f4914f6cdd1dull;
double uniform() { // [0, 1), 53 bits
  g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
  return (double)(g_state >> 11) * 0x1.0p-53;
}

// A row-major rows x cols operand as the accuracy model samples it, with
// largest magnitude about `scale`.
std::vector<double> sampled(int rows, int cols, bool log40, double scale) {
  std::vector<double> v((size_t)rows * cols);
  for (double &x : v) {
    double mag =
        log40 ? scale * std::exp2(-40.0 * uniform()) : scale * uniform();
    x = uniform() < 0.5 ? -mag : mag;
  }
  return v;
}

// The number of products that differ over both distributions, at the modulus
// counts the compiler proposes.
size_t runSampled(int M, int N, int K) {
  size_t differ = 0;
  for (bool log40 : {true, false}) {
    std::vector<double> A = sampled(M, K, log40, 0.5);
    std::vector<double> B = sampled(K, N, log40, 2.0);
    double *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * 8);
    cudaMalloc(&dB, B.size() * 8);
    cudaMalloc(&dC, (size_t)M * N * 8);
    cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    std::vector<double> C((size_t)M * N), P((size_t)M * N);
    size_t products = 0, d = 0;
    int counts = 0;
    for (unsigned nm = 8; nm <= 14; ++nm, ++counts) {
      __poseidon_ozaki_dgemm(dC, dA, dB, M, N, K, K, N, N, 0, 0, 0, 1.0, 0.0,
                             nullptr, nm);
      cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost);
      simulateOzakiIIDgemm(nm, {A.data(), (unsigned)K, (unsigned)M, true},
                           {B.data(), (unsigned)N, (unsigned)N, false},
                           (unsigned)K, P.data());
      for (size_t i = 0; i < C.size(); ++i, ++products)
        d += !sameBits(C[i], P[i]);
    }
    printf("model %dx%dx%d %s: %d moduli, %zu products, %zu differ\n", M, N, K,
           log40 ? "log40" : "uniform", counts, products, d);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    differ += d;
  }
  return differ;
}

} // namespace

int main() {
  size_t differ = run(37, 50, 61);
  differ += run(20, 24, 1100);
  differ += run(48, 48, 48);
  differ += runSampled(16, 16, 2048);
  return differ != 0;
}
