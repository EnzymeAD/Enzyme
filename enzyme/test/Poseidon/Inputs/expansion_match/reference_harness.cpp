// Runs the raw-component functions emit.cpp writes (expansion-match-raw) on
// the host against the external routines they reproduce, bit for bit (NaN
// payload and sign excepted): n = 3, 4 against mX_real (mx_real_subset.h),
// n = 2 add, sub and mul against MultiFloats.jl's two-limb mfadd and mfmul
// (below). Prints, per routine, the cases that differ, and the two-component
// results at the special operands where mX_real and MultiFloats.jl agree.
// Compile without contraction.
#include "gen.h"
#include "mx_real_subset.h"

#include <cmath>
#include <cstdio>

using namespace expansion_match;

extern "C" {
#define DECL(n)                                                                \
  void p##n##_add(const float *, const float *, float *);                      \
  void p##n##_sub(const float *, const float *, float *);                      \
  void p##n##_mul(const float *, const float *, float *);                      \
  void p##n##_sqr(const float *, float *);                                     \
  void p##n##_div(const float *, const float *, float *);                      \
  void p##n##_sqrt(const float *, float *);                                    \
  void p##n##_neg(const float *, float *);                                     \
  void p##n##_fma(const float *, const float *, const float *, float *);       \
  void p##n##_fmasq(const float *, const float *, float *);
DECL(2)
DECL(3)
DECL(4)
}

namespace {

// MultiFloats.jl 3.2.6+8 (commit 28041a3, MIT License), Val{2}: two_sum and
// fast_two_sum (src/MultiFloats.jl 313-321, 1047-1052), two_prod (1070-1074),
// mfadd (src/mfadd.jl 10-24) and mfmul (src/mfmul.jl 10-23), transcribed.
void twoSum(float a, float b, float &s, float &e) {
  s = a + b;
  float ap = s - b, bp = s - ap;
  e = (a - ap) + (b - bp);
}
void fastTwoSum(float a, float b, float &s, float &e) {
  s = a + b;
  e = b - (s - a);
}
void mfadd(const float *x, const float *y, float *o) {
  float a, b, c, d;
  twoSum(x[0], y[0], a, b);
  twoSum(x[1], y[1], c, d);
  fastTwoSum(a, c, a, c);
  b += d;
  b += c;
  fastTwoSum(a, b, o[0], o[1]);
}
void mfmul(const float *x, const float *y, float *o) {
  float p00 = x[0] * y[0];
  float e00 = std::fma(x[0], y[0], -p00);
  float p01 = x[0] * y[1];
  float p10 = x[1] * y[0];
  p01 += p10;
  e00 += p01;
  fastTwoSum(p00, e00, o[0], o[1]);
}

int failures = 0;

template <typename Emitted, typename Reference>
void check(int n, const char *what, Emitted emitted, Reference reference) {
  long cases = 0, differ = 0;
  forEachCase(n, 10000, [&](const Case &k) {
    float e[4] = {}, r[4] = {};
    emitted(k, e);
    reference(k, r);
    ++cases;
    for (int i = 0; i < n; ++i)
      if (!sameFloat(e[i], r[i])) {
        if (differ++ < 3)
          printf(
              "  differs: %s class %s component %d emitted %a reference %a\n",
              what, className(k.cls), i, e[i], r[i]);
        break;
      }
  });
  failures += differ != 0;
  printf("n=%d %s: %ld cases, %ld differ\n", n, what, cases, differ);
}

template <int N> struct Emitted;
#define EMITTED(N)                                                             \
  template <> struct Emitted<N> {                                              \
    static void add(const Case &k, float *o) { p##N##_add(k.a, k.b, o); }      \
    static void sub(const Case &k, float *o) { p##N##_sub(k.a, k.b, o); }      \
    static void mul(const Case &k, float *o) { p##N##_mul(k.a, k.b, o); }      \
    static void sqr(const Case &k, float *o) { p##N##_sqr(k.a, o); }           \
    static void div(const Case &k, float *o) { p##N##_div(k.a, k.b, o); }      \
    static void sqrt(const Case &k, float *o) { p##N##_sqrt(k.a, o); }         \
    static void neg(const Case &k, float *o) { p##N##_neg(k.a, o); }           \
    static void fma(const Case &k, float *o) { p##N##_fma(k.a, k.b, k.c, o); } \
    static void fmasq(const Case &k, float *o) { p##N##_fmasq(k.a, k.c, o); }  \
  };
EMITTED(2)
EMITTED(3)
EMITTED(4)

template <int N> void againstMXReal() {
  using E = Emitted<N>;
  check(N, "add", E::add,
        [](const Case &k, float *o) { mxref::add(k.a, k.b, o, N); });
  check(N, "sub", E::sub,
        [](const Case &k, float *o) { mxref::sub(k.a, k.b, o, N); });
  check(N, "mul", E::mul,
        [](const Case &k, float *o) { mxref::mul(k.a, k.b, o, N); });
  check(N, "a*a", E::sqr,
        [](const Case &k, float *o) { mxref::mul(k.a, k.a, o, N); });
  check(N, "div", E::div,
        [](const Case &k, float *o) { mxref::div(k.a, k.b, o, N); });
  check(N, "sqrt", E::sqrt,
        [](const Case &k, float *o) { mxref::sqrt(k.a, o, N); });
  check(N, "neg", E::neg,
        [](const Case &k, float *o) { mxref::neg(k.a, o, N); });
  check(N, "fma", E::fma, [](const Case &k, float *o) {
    float p[4];
    mxref::mul(k.a, k.b, p, N);
    mxref::add(p, k.c, o, N);
  });
  check(N, "fma(a,a,c)", E::fmasq, [](const Case &k, float *o) {
    float p[4];
    mxref::mul(k.a, k.a, p, N);
    mxref::add(p, k.c, o, N);
  });
}

void show(const char *what, const float *r) {
  printf("%s = %a %a\n", what, r[0], r[1]);
}

} // namespace

int main() {
  againstMXReal<3>();
  againstMXReal<4>();
  using E = Emitted<2>;
  check(2, "add", E::add, [](const Case &k, float *o) { mfadd(k.a, k.b, o); });
  check(2, "sub", E::sub, [](const Case &k, float *o) {
    float nb[2] = {-k.b[0], -k.b[1]};
    mfadd(k.a, nb, o);
  });
  check(2, "mul", E::mul, [](const Case &k, float *o) { mfmul(k.a, k.b, o); });

  float r[2];
  const float zeros[4][2] = {
      {0.0f, 0.0f}, {-0.0f, 0.0f}, {0.0f, -0.0f}, {-0.0f, -0.0f}};
  for (const auto &z : zeros) {
    p2_sqrt(z, r);
    char what[64];
    snprintf(what, sizeof what, "n=2 sqrt(%a, %a)", z[0], z[1]);
    show(what, r);
  }
  const float x[2] = {-10.5f, 0.0f}, y[2] = {-14.0f, 0.0f};
  p2_div(x, y, r);
  show("n=2 (-10.5, 0) / (-14, 0)", r);
  return failures;
}
