// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi

// A fixed-point loop marked by __enzyme_set_fixed_point, as a Fortran
// !DIR$ ENZYME FIXED_POINT directive lowers: Enzyme outlines one iteration
// as the step. The loop is tested in its header (DO WHILE) or at its end,
// and carries locals (the error, an iteration count, a tolerance it
// tightens) from one iteration to the next. The gradient must equal the
// gradient through every iteration of the same loop without the marker.

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define N 5

void __enzyme_autodiff(void *, ...);
void __enzyme_set_fixed_point(double reduction, int64_t max_iters,
                              void *control, ...);
extern int enzyme_dup;
extern int enzyme_const;

double u[N], p[N];
int64_t iters;

__attribute__((noinline)) static double iterate(double tol) {
  double tmp[N], err = 0;
  for (int k = 0; k < N; k++)
    tmp[k] = 0.25 * cos(u[k] - 0.5 * u[(k + 1) % N]) + p[k] * p[k];
  for (int k = 0; k < N; k++) {
    err = fmax(err, fabs(tmp[k] - u[k]));
    u[k] = tmp[k];
  }
  return err > tol ? err : 0;
}

static void setup(const double *x) {
  for (int k = 0; k < N; k++) {
    u[k] = 0;
    p[k] = x[k];
  }
}

static double loss(void) {
  double s = 0;
  for (int k = 0; k < N; k++)
    s += u[k] * u[k] * p[k];
  return s;
}

#define MARK                                                                   \
  __enzyme_set_fixed_point(1e-30, (int64_t)-1, (void *)0, u,                   \
                           (int64_t)sizeof(u))

// DO WHILE: tested before each iteration.
#define HEADER(name, mark)                                                     \
  static double name(const double *x) {                                        \
    setup(x);                                                                  \
    double err = 1, tol = 1e-8;                                                \
    int64_t n = 0;                                                             \
    while (err > 0) {                                                          \
      mark;                                                                    \
      err = iterate(tol);                                                      \
      tol = fmax(tol * 0.1, 1e-15);                                            \
      n++;                                                                     \
    }                                                                          \
    iters = n;                                                                 \
    return loss();                                                             \
  }

// Tested after each iteration.
#define LATCH(name, mark)                                                      \
  static double name(const double *x) {                                        \
    setup(x);                                                                  \
    double err;                                                                \
    do {                                                                       \
      mark;                                                                    \
      err = iterate(1e-15);                                                    \
    } while (err > 0);                                                         \
    return loss();                                                             \
  }

HEADER(header_plain, (void)0)
HEADER(header_fp, MARK)
LATCH(latch_plain, (void)0)
LATCH(latch_fp, MARK)

// The gradients d0 (plain) and d (marked), from the same x.
static int check(const char *name, const double *d0, const double *d,
                 int64_t n0, const double *u0) {
  int failures = 0;
  for (int k = 0; k < N; k++) {
    double err = fabs(d[k] - d0[k]) / (fabs(d0[k]) + 1e-300);
    if (!(err < 1e-12) || d0[k] == 0) {
      printf("%s: dx[%d] = %.17g, expected %.17g\n", name, k, d[k], d0[k]);
      failures++;
    }
    if (u[k] != u0[k]) {
      printf("%s: u[%d] after the gradient differs\n", name, k);
      failures++;
    }
  }
  if (iters != n0) {
    printf("%s: %lld iterations, expected %lld\n", name, (long long)iters,
           (long long)n0);
    failures++;
  }
  return failures;
}

#define COMPARE(name, plain, fp)                                               \
  {                                                                            \
    double x[N], d0[N] = {0}, d[N] = {0}, u0[N];                               \
    for (int k = 0; k < N; k++)                                                \
      x[k] = 0.3 + 0.1 * k;                                                    \
    __enzyme_autodiff((void *)plain, enzyme_dup, x, d0);                       \
    int64_t n0 = iters;                                                        \
    for (int k = 0; k < N; k++)                                                \
      u0[k] = u[k];                                                            \
    __enzyme_autodiff((void *)fp, enzyme_dup, x, d);                           \
    failures += check(name, d0, d, n0, u0);                                    \
  }

int main(void) {
  int failures = 0;
  COMPARE("header", header_plain, header_fp);
  COMPARE("latch", latch_plain, latch_fp);
  if (failures) {
    printf("%d failures\n", failures);
    return 1;
  }
  printf("ok\n");
  return 0;
}
