// Every modulus count the Ozaki-II runtime accepts (3..14), each of which runs
// its own instantiation of the residue and CRT kernels, against the accuracy
// model's simulation of the published scheme (simulateOzakiIIDgemm), bit for
// bit: both arms of __poseidon_ozaki_dgemm_ex (forced), every A/B/C storage
// order with beta = 0.5 on one, at 37 x 50 x 61 (padded to 64 on the square
// arm); the rectangular arm at 20 x 24 x 1100, whose strided operands take
// the chunked sum of squares; and the square entry at N = 48. Rows and columns
// of the operands span 2^-30..2^30, with every 17th element of a row of A
// 10^6 times larger. The same on the operands the accuracy model samples, on
// the tile it simulates (16 x 16 with K = 2048, which the runtime's own choice
// of arm sends to the rectangular one), both row-major, elements log-uniform
// over the 40 binades below the largest (the tolerance path) or uniform (the
// budget path), at every modulus count the compiler proposes.
//
// REQUIRES: cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -frounding-math \
// RUN:     -I%S/../../../Poseidon -c \
// RUN:     %S/../../../Poseidon/Techniques/Matmul/MatmulSimulation.cpp \
// RUN:     -o %t/sim.o
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=off \
// RUN:     -I%S/../../../Poseidon/Techniques/Matmul %s -x none %t/sim.o \
// RUN:     %poseidon_rt_lib -L%cuda_lib_dir -lcublas -lcudart -lstdc++ -lm \
// RUN:     -o %t/exe
// RUN: %t/exe | FileCheck --match-full-lines %s

// CHECK: rect 37x50x61: 8 layouts, 177600 products, 0 differ
// CHECK-NEXT: square 37x50x61: 8 layouts, 177600 products, 0 differ
// CHECK-NEXT: rect 20x24x1100: 8 layouts, 46080 products, 0 differ
// CHECK-NEXT: entry 48x48x48: 1 layouts, 27648 products, 0 differ
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

extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int N, int lda, int ldb,
                                       int ldc, int transA, int transB,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli);
extern "C" void __poseidon_ozaki_dgemm_ex(
    double *C, const double *A, const double *B, int M, int Ncols, int K,
    int lda, int ldb, int ldc, int aColMajor, int bColMajor, int cColMajor,
    double alpha, double beta, cudaStream_t stream, int num_moduli);
extern "C" void __poseidon_ozaki_set_path(int mode);

using poseidon::simulateOzakiIIDgemm;

namespace {

enum Arm { Rect = 1, Square = 2, Entry = 3 };

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

// A * B before the epilogue, row-major, as the arm computes it. The rectangular
// arm reads a vector contiguously when its storage runs along k; the square
// arm (and the entry) reads row-major operands zero-padded to S x S.
std::vector<double> simulate(const Operands &o, Arm arm, int aColMajor,
                             int bColMajor, unsigned nm) {
  std::vector<double> P((size_t)o.M * o.N);
  if (arm == Rect) {
    std::vector<double> As = store(o.A, o.M, o.K, aColMajor);
    std::vector<double> Bs = store(o.B, o.K, o.N, bColMajor);
    simulateOzakiIIDgemm(nm,
                         {As.data(), (unsigned)(aColMajor ? o.M : o.K),
                          (unsigned)o.M, !aColMajor},
                         {Bs.data(), (unsigned)(bColMajor ? o.K : o.N),
                          (unsigned)o.N, (bool)bColMajor},
                         (unsigned)o.K, P.data());
    return P;
  }
  const int S = arm == Entry ? o.M : (std::max({o.M, o.N, o.K}) + 15) / 16 * 16;
  std::vector<double> As((size_t)S * S, 0.0), Bs((size_t)S * S, 0.0),
      Ps((size_t)S * S);
  for (int r = 0; r < o.M; ++r)
    for (int k = 0; k < o.K; ++k)
      As[(size_t)r * S + k] = o.A[(size_t)r * o.K + k];
  for (int k = 0; k < o.K; ++k)
    for (int c = 0; c < o.N; ++c)
      Bs[(size_t)k * S + c] = o.B[(size_t)k * o.N + c];
  simulateOzakiIIDgemm(nm, {As.data(), (unsigned)S, (unsigned)S, true},
                       {Bs.data(), (unsigned)S, (unsigned)S, false},
                       (unsigned)S, Ps.data());
  for (int r = 0; r < o.M; ++r)
    for (int c = 0; c < o.N; ++c)
      P[(size_t)r * o.N + c] = Ps[(size_t)r * S + c];
  return P;
}

bool sameBits(double a, double b) {
  uint64_t x, y;
  std::memcpy(&x, &a, 8);
  std::memcpy(&y, &b, 8);
  return x == y;
}

// The number of products that differ, or 1 after a CUDA error.
size_t run(const char *name, Arm arm, int M, int N, int K) {
  const Operands o(M, N, K);
  double *dA, *dB, *dC;
  cudaMalloc(&dA, o.A.size() * 8);
  cudaMalloc(&dB, o.B.size() * 8);
  cudaMalloc(&dC, o.C0.size() * 8);
  if (arm != Entry)
    __poseidon_ozaki_set_path(arm);
  const int layouts = arm == Entry ? 1 : 8;
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
      if (arm == Entry)
        __poseidon_ozaki_dgemm(dC, dA, dB, M, M, M, M, 0, 0, 1.0, beta, nullptr,
                               nm);
      else
        __poseidon_ozaki_dgemm_ex(dC, dA, dB, M, N, K, ac ? M : K, bc ? K : N,
                                  cc ? M : N, ac, bc, cc, 1.0, beta, nullptr,
                                  nm);
      if (cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        printf("%s: CUDA error\n", name);
        return 1;
      }
      const std::vector<double> P = simulate(o, arm, ac, bc, nm);
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c, ++products) {
          const size_t i = cc ? (size_t)c * M + r : (size_t)r * N + c;
          const double ab = P[(size_t)r * N + c];
          const double want = beta == 0.0 ? ab : std::fma(beta, C0[i], ab);
          differ += !sameBits(C[i], want);
        }
    }
  }
  printf("%s %dx%dx%d: %d layouts, %zu products, %zu differ\n", name, M, N, K,
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
// counts the compiler proposes, with the runtime choosing the arm.
size_t runSampled(int M, int N, int K) {
  __poseidon_ozaki_set_path(0);
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
      __poseidon_ozaki_dgemm_ex(dC, dA, dB, M, N, K, K, N, N, 0, 0, 0, 1.0, 0.0,
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
  size_t differ = run("rect", Rect, 37, 50, 61);
  differ += run("square", Square, 37, 50, 61);
  differ += run("rect", Rect, 20, 24, 1100);
  differ += run("entry", Entry, 48, 48, 48);
  differ += runSampled(16, 16, 2048);
  return differ != 0;
}
