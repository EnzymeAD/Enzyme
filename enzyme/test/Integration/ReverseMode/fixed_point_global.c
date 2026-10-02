// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi

// A fixed-point loop whose state and parameters are globals, as MITgcm keeps
// them in COMMON blocks, between code that overwrites the state afterwards:
// the reverse pass must linearize at the converged state, not at what the
// later code left. Compared with the gradient through every iteration (and
// with finite differences, by hand). Globals are active only with
// -enzyme-global-activity, as for MITgcm.

#include <enzyme/fixed_point.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define N 6

void __enzyme_autodiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

double u[N], p[N];

__attribute__((noinline)) static int step(int64_t i, double *tol) {
  (void)i;
  double tmp[N], err = 0;
  for (int k = 0; k < N; k++)
    tmp[k] = 0.3 * tanh(u[k] + 0.7 * u[(k + 1) % N] - 0.2 * u[(k + 5) % N]) +
             p[k] * u[(k + 2) % N] * 0.2 + p[k];
  for (int k = 0; k < N; k++) {
    err = fmax(err, fabs(tmp[k] - u[k]));
    u[k] = tmp[k];
  }
  return err > *tol;
}

static void setup(const double *x) {
  for (int k = 0; k < N; k++) {
    u[k] = 0;
    p[k] = x[k];
  }
}

// After the solve, use the state and then overwrite it.
static double finish(void) {
  double s = 0;
  for (int k = 0; k < N; k++) {
    s += sin(u[k]) * p[k];
    u[k] = 0.5 * u[k] * u[k];
  }
  for (int k = 0; k < N; k++)
    s += u[k];
  return s;
}

static double plain(const double *x, double *tol) {
  setup(x);
  int64_t i = 0;
  while (step(i++, tol))
    ;
  return finish();
}

static double fixed(const double *x, double *tol) {
  setup(x);
  __enzyme_fixed_point((void *)step, enzyme_fp_state, u, (int64_t)sizeof(u),
                       enzyme_fp_reduction, 1e-30, tol);
  return finish();
}

int main(void) {
  double tol = 1e-15;
  double x[N], dx0[N] = {0}, dx[N] = {0};
  for (int k = 0; k < N; k++)
    x[k] = 0.1 + 0.05 * k;
  __enzyme_autodiff((void *)plain, enzyme_dup, x, dx0, enzyme_const, &tol);
  double u0[N];
  for (int k = 0; k < N; k++)
    u0[k] = u[k];
  __enzyme_autodiff((void *)fixed, enzyme_dup, x, dx, enzyme_const, &tol);
  int failures = 0;
  for (int k = 0; k < N; k++) {
    double err = fabs(dx[k] - dx0[k]) / (fabs(dx0[k]) + 1e-300);
    if (!(err < 1e-12)) {
      printf("dx[%d] = %.17g, expected %.17g\n", k, dx[k], dx0[k]);
      failures++;
    }
    if (u[k] != u0[k]) {
      printf("u[%d] = %.17g after the gradient, expected %.17g\n", k, u[k],
             u0[k]);
      failures++;
    }
  }
  if (failures) {
    printf("%d failures\n", failures);
    return 1;
  }
  printf("ok\n");
  return 0;
}
