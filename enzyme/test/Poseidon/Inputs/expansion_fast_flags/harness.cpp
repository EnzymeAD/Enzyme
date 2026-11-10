// Runs the expanded ex2_add and ex2_mul of expansion_fast_flags.cpp and
// expansion_fast_math_attributes.cpp on the host against MultiFloats.jl's
// two-limb addition and multiplication (mfadd and mfmul at Val{2},
// ../expansion_match/multifloats_subset.h); compile without contraction.
#include "../expansion_match/multifloats_subset.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" double ex2_add(double, double);
extern "C" double ex2_mul(double, double);

namespace {

struct Pair {
  float x[2];
};

Pair split(double x) {
  float hi = (float)x;
  return {{hi, (float)(x - (double)hi)}};
}

double join(Pair p) { return (double)p.x[0] + (double)p.x[1]; }

Pair mfadd(Pair x, Pair y) {
  Pair r;
  mfref::mfadd(x.x, y.x, r.x);
  return r;
}

Pair mfmul(Pair x, Pair y) {
  Pair r;
  mfref::mfmul(x.x, y.x, r.x);
  return r;
}

uint64_t bits(double d) {
  uint64_t b;
  memcpy(&b, &d, 8);
  return b;
}

uint64_t g_state = 0x9e3779b97f4a7c15ull;
double operand() {
  g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
  double m = 1.0 + (double)(g_state >> 11) * 0x1.0p-53;
  int e = (int)((g_state >> 3) % 61) - 30;
  return (g_state & 1 ? -1.0 : 1.0) * std::ldexp(m, e);
}

} // namespace

int main() {
  const int n = 100000;
  long add = 0, mul = 0;
  for (int i = 0; i < n; ++i) {
    double a = i ? operand() : 0.1, b = i ? operand() : 0.2;
    add += bits(ex2_add(a, b)) == bits(join(mfadd(split(a), split(b))));
    mul += bits(ex2_mul(a, b)) == bits(join(mfmul(split(a), split(b))));
  }
  printf("ex2_add(0.1, 0.2) = %a\n", ex2_add(0.1, 0.2));
  printf("sum %ld of %d, product %ld of %d equal to the reference\n", add, n,
         mul, n);
  return add != n || mul != n;
}
