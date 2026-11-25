// RUN: rm -rf %t && mkdir %t
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -frounding-math \
// RUN:     -I%S/../../../../Poseidon -I%S/../../../../Poseidon/Techniques/Matmul \
// RUN:     %S/../../../../Poseidon/Techniques/Matmul/Simulation.cpp %s \
// RUN:     -o %t/model
// RUN: sed -n 's|^// OZAKI\(.*digest=[0-9a-f]*\).*|// MODEL\1|p' \
// RUN:     %S/../../Inputs/gemm_runtime.cpp > %t/runtime.check
// RUN: FileCheck --check-prefix=CASES --implicit-check-not=MODEL %s < %t/runtime.check
// RUN: %t/model | FileCheck --check-prefix=MODEL --match-full-lines \
// RUN:     %t/runtime.check

// The accuracy model's Ozaki-II product (simulateOzakiIIDgemm), run on the
// host, returns the digests Inputs/gemm_runtime.cpp records for the runtime
// (operands generated the same way): a row-major square at N = 256, a skinny
// product with A row-major and B and C column-major (beta = 1), and a nearly
// square one with the opposite layouts, at 8, 10, 13 and 14 moduli. Each
// operand is laid out as the runtime reads it, since that fixes the order of
// its sum of squares.
//
// CASES-COUNT-8: // MODEL

#include "Simulation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace poseidon;

namespace {

struct Shape {
  const char *name;
  int M, Ncols, K, aColMajor, bColMajor, cColMajor;
  double beta;
};

const Shape kShapes[] = {
    {"square256", 256, 256, 256, 0, 0, 0, 0.0},
    {"rect343x192x1536", 343, 192, 1536, 0, 1, 1, 1.0},
    {"pad300", 300, 300, 300, 1, 0, 0, 0.0},
};

struct Case {
  int shape, nm;
};
const Case kCases[] = {{0, 8},  {0, 10}, {0, 13}, {0, 14},
                       {1, 10}, {1, 13}, {2, 10}, {2, 13}};

uint64_t g_state = 0x9e3779b97f4a7c15ull;
double uniform() {
  g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
  return (double)(g_state >> 11) * 0x1.0p-53;
}

std::vector<double> operand(int rows, int cols, int colMajor) {
  std::vector<double> v((size_t)rows * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      int e = (r * 7) % 9 - 4 + (c * 5) % 9 - 4;
      double x = std::ldexp(uniform() - 0.5, e);
      v[colMajor ? (size_t)c * rows + r : (size_t)r * cols + c] = x;
    }
  return v;
}

uint64_t digest(const std::vector<double> &v) {
  uint64_t h = 14695981039346656037ull;
  for (double d : v) {
    uint64_t b;
    std::memcpy(&b, &d, 8);
    for (int i = 0; i < 8; ++i) {
      h ^= (b >> (8 * i)) & 0xff;
      h *= 1099511628211ull;
    }
  }
  return h;
}

// The runtime's axpby (ozaki_rt.cu) with alpha = 1.
double axpby(double beta, double ab, double c) {
  return beta == 0.0 ? ab : std::fma(beta, c, ab);
}

// C = A B + beta C0 as __poseidon_ozakiii_dgemm computes it for `s`: a vector is
// contiguous when its storage runs along k.
std::vector<double> product(const Shape &s, int nm,
                            const std::vector<double> &A,
                            const std::vector<double> &B,
                            const std::vector<double> &C0) {
  std::vector<double> C(C0), P((size_t)s.M * s.Ncols);
  simulateOzakiIIDgemm(nm,
                       {A.data(), (unsigned)(s.aColMajor ? s.M : s.K),
                        (unsigned)s.M, !s.aColMajor},
                       {B.data(), (unsigned)(s.bColMajor ? s.K : s.Ncols),
                        (unsigned)s.Ncols, (bool)s.bColMajor},
                       s.K, P.data());
  for (int r = 0; r < s.M; ++r)
    for (int c = 0; c < s.Ncols; ++c) {
      size_t o = s.cColMajor ? (size_t)c * s.M + r : (size_t)r * s.Ncols + c;
      C[o] = axpby(s.beta, P[(size_t)r * s.Ncols + c], C0[o]);
    }
  return C;
}

} // namespace

int main() {
  for (int si = 0; si < 3; ++si) {
    const Shape &s = kShapes[si];
    std::vector<double> A = operand(s.M, s.K, s.aColMajor);
    std::vector<double> B = operand(s.K, s.Ncols, s.bColMajor);
    std::vector<double> C0 = operand(s.M, s.Ncols, s.cColMajor);
    for (const Case &c : kCases)
      if (c.shape == si)
        printf("ozaki %s nm=%d digest=%016llx\n", s.name, c.nm,
               (unsigned long long)digest(product(s, c.nm, A, B, C0)));
  }
  return 0;
}
