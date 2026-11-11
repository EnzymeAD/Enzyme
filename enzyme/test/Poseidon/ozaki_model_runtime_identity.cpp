// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -frounding-math \
// RUN:     -I%S/../../Poseidon -I%S/../../Poseidon/Techniques/Matmul \
// RUN:     %S/../../Poseidon/Techniques/Matmul/MatmulSimulation.cpp %s \
// RUN:     -o %t/model
// RUN: sed -n 's|^// OZAKI\(.*digest=[0-9a-f]*\).*|// MODEL\1|p' \
// RUN:     %S/Inputs/dispatch_rt.cpp > %t/runtime.check
// RUN: grep -c MODEL %t/runtime.check | FileCheck --check-prefix=CASES %s
// RUN: %t/model | FileCheck --check-prefix=MODEL --match-full-lines \
// RUN:     %t/runtime.check

// The accuracy model's Ozaki-II product (simulateOzakiIIDgemm), run on the
// host, returns the bits the runtime returned on an RTX 5090 for the cases
// Integration/dispatch_rt_cuda.cu holds it to (Inputs/dispatch_rt.cpp, whose
// operands are generated here the same way): the square entry at N = 256, the
// rectangular arm (A row-major, B and C column-major, beta = 1), and the
// padded square arm, at 8, 10, 13 and 14 moduli. Each operand vector is given
// with the layout in which the arm the runtime takes reads it, which decides
// the order of its sum of squares; the epilogue is the runtime's
// (poz_axpby).
//
// CASES: {{^8$}}

#include "MatmulSimulation.h"

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
  bool square;
  int M, Ncols, K, aColMajor, bColMajor, cColMajor;
  double beta;
};

const Shape kShapes[] = {
    {"square256", true, 256, 256, 256, 0, 0, 0, 0.0},
    {"rect343x192x1536", false, 343, 192, 1536, 0, 1, 1, 1.0},
    {"pad300", false, 300, 300, 300, 1, 0, 0, 0.0},
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

// poz_axpby with alpha = 1.
double axpby(double beta, double ab, double c) {
  return beta == 0.0 ? ab : std::fma(beta, c, ab);
}

// C = A B + beta C0 as __poseidon_ozaki_dgemm(_ex) computes it for `s`.
std::vector<double> product(const Shape &s, int nm,
                            const std::vector<double> &A,
                            const std::vector<double> &B,
                            const std::vector<double> &C0) {
  std::vector<double> C(C0);
  auto store = [&](int r, int c, double ab) {
    size_t o = s.cColMajor ? (size_t)c * s.M + r : (size_t)r * s.Ncols + c;
    C[o] = axpby(s.beta, ab, C0[o]);
  };
  int S = (std::max({s.M, s.Ncols, s.K}) + 15) / 16 * 16;
  double padWaste = (double)S * S * S / ((double)s.M * s.Ncols * s.K);
  std::vector<double> P;
  if (!s.square && padWaste >= 1.05) {
    // poz_run_rect: a vector is contiguous when its storage runs along k.
    P.resize((size_t)s.M * s.Ncols);
    simulateOzakiIIDgemm(nm,
                         {A.data(), (unsigned)(s.aColMajor ? s.M : s.K),
                          (unsigned)s.M, !s.aColMajor},
                         {B.data(), (unsigned)(s.bColMajor ? s.K : s.Ncols),
                          (unsigned)s.Ncols, (bool)s.bColMajor},
                         s.K, P.data());
    for (int r = 0; r < s.M; ++r)
      for (int c = 0; c < s.Ncols; ++c)
        store(r, c, P[(size_t)r * s.Ncols + c]);
    return C;
  }
  // The square pipeline on row-major operands, repacked and zero-padded to
  // S x S by the _ex entry (poz_pack, poz_crop): rows of A contiguous,
  // columns of B strided.
  int N = s.square ? s.M : S;
  std::vector<double> As((size_t)N * N, 0.0), Bs((size_t)N * N, 0.0);
  for (int r = 0; r < s.M; ++r)
    for (int k = 0; k < s.K; ++k)
      As[(size_t)r * N + k] =
          A[s.aColMajor ? (size_t)k * s.M + r : (size_t)r * s.K + k];
  for (int k = 0; k < s.K; ++k)
    for (int c = 0; c < s.Ncols; ++c)
      Bs[(size_t)k * N + c] =
          B[s.bColMajor ? (size_t)c * s.K + k : (size_t)k * s.Ncols + c];
  P.resize((size_t)N * N);
  simulateOzakiIIDgemm(nm, {As.data(), (unsigned)N, (unsigned)N, true},
                       {Bs.data(), (unsigned)N, (unsigned)N, false},
                       (unsigned)N, P.data());
  for (int r = 0; r < s.M; ++r)
    for (int c = 0; c < s.Ncols; ++c)
      store(r, c, P[(size_t)r * N + c]);
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
