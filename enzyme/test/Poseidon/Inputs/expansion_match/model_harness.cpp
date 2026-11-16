// model_harness <dir>: runs the code emit.cpp's expansion-match-model pass
// emitted on the host and compares it, bit for bit (NaN payload and sign
// excepted), with what the pass's accuracy model computed for the same inputs
// (<dir>/raw.bin, <dir>/app.bin): the raw-component routines and boundary
// conversions at n = 2, 3, 4 on the eight operand classes, and the FP64 and
// FP32 functions (single operations and chains) materialized by
// applyExpansion. Compile without contraction, with -I<dir> for fns.inc.
#include "gen.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

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
  void p##n##_fmasq(const float *, const float *, float *);                    \
  void p##n##_from_f64(double, float *);                                       \
  double p##n##_to_f64(const float *);                                         \
  float p##n##_to_f32(const float *);
DECL(2)
DECL(3)
DECL(4)
#define FN(name, n, T, kind) T name(T, T, T);
#include "fns.inc"
#undef FN
}

namespace {

struct Raw {
  void (*add)(const float *, const float *, float *);
  void (*sub)(const float *, const float *, float *);
  void (*mul)(const float *, const float *, float *);
  void (*sqr)(const float *, float *);
  void (*div)(const float *, const float *, float *);
  void (*sqrt)(const float *, float *);
  void (*neg)(const float *, float *);
  void (*fma)(const float *, const float *, const float *, float *);
  void (*fmasq)(const float *, const float *, float *);
  void (*fromF64)(double, float *);
  double (*toF64)(const float *);
  float (*toF32)(const float *);
};
#define RAW(n)                                                                 \
  Raw {                                                                        \
    p##n##_add, p##n##_sub, p##n##_mul, p##n##_sqr, p##n##_div, p##n##_sqrt,   \
        p##n##_neg, p##n##_fma, p##n##_fmasq, p##n##_from_f64, p##n##_to_f64,  \
        p##n##_to_f32                                                          \
  }
const Raw kRaw[3] = {RAW(2), RAW(3), RAW(4)};
const char *const kOps[] = {"add",  "sub", "mul", "a*a",        "div",
                            "sqrt", "neg", "fma", "fma(a,a,c)", "from_f64"};

struct Count {
  long cases = 0, differ = 0;
};
std::map<std::string, Count> counts;

void tally(const std::string &key, bool same) {
  Count &c = counts[key];
  ++c.cases;
  c.differ += !same;
}

bool sameLimbs(const float *x, const float *y, unsigned n) {
  for (unsigned i = 0; i < n; ++i)
    if (!sameFloat(x[i], y[i]))
      return false;
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  const std::string dir = argv[1];

  FILE *raw = fopen((dir + "/raw.bin").c_str(), "rb");
  if (!raw)
    return 2;
  for (unsigned n = 2; n <= 4; ++n) {
    const Raw &R = kRaw[n - 2];
    for (long i = 0; i < 10000L * kClasses; ++i) {
      Case k;
      float model[10][4];
      double toF64;
      float toF32;
      if (fread(&k, sizeof k, 1, raw) != 1 || fread(model, 4, 40, raw) != 40 ||
          fread(&toF64, 8, 1, raw) != 1 || fread(&toF32, 4, 1, raw) != 1)
        return 2;
      float e[10][4] = {};
      R.add(k.a, k.b, e[0]);
      R.sub(k.a, k.b, e[1]);
      R.mul(k.a, k.b, e[2]);
      R.sqr(k.a, e[3]);
      R.div(k.a, k.b, e[4]);
      R.sqrt(k.a, e[5]);
      R.neg(k.a, e[6]);
      R.fma(k.a, k.b, k.c, e[7]);
      R.fmasq(k.a, k.c, e[8]);
      R.fromF64((double)k.a[0] + (double)k.a[1], e[9]);
      const std::string N = "n=" + std::to_string(n) + " ";
      for (int op = 0; op < 10; ++op)
        tally(N + kOps[op], sameLimbs(e[op], model[op], n));
      tally(N + "to_f64", sameDouble(R.toF64(k.a), toF64));
      tally(N + "to_f32", sameFloat(R.toF32(k.a), toF32));
    }
  }
  fclose(raw);

  FILE *app = fopen((dir + "/app.bin").c_str(), "rb");
  if (!app)
    return 2;
  std::vector<double> pts(3 * 4000);
  if (fread(pts.data(), 8, pts.size(), app) != pts.size())
    return 2;
  auto run = [&](unsigned n, const char *kind, auto fn, auto cast) {
    for (size_t i = 0; i < 4000; ++i) {
      double model;
      if (fread(&model, 8, 1, app) != 1)
        exit(2);
      double r =
          fn(cast(pts[3 * i]), cast(pts[3 * i + 1]), cast(pts[3 * i + 2]));
      tally("n=" + std::to_string(n) + " " + kind, sameDouble(r, model));
    }
  };
#define FN(name, n, T, kind) run(n, kind, name, [](double v) { return (T)v; });
#include "fns.inc"
#undef FN
  fclose(app);

  int failures = 0;
  for (auto &[key, c] : counts) {
    printf("%s: %ld cases, %ld differ\n", key.c_str(), c.cases, c.differ);
    failures += c.differ != 0;
  }
  return failures;
}
