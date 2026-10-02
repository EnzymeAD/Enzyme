// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// A nonlinear fixed point u = C sin(A u) + p, iterated to convergence through
// __enzyme_fixed_point. Its gradient comes from the adjoint iteration at the
// converged state, and must equal the gradient through every iteration of
// the plain loop (which converges to the same value), with the state after it
// as the plain loop leaves it. The step also writes an output y from the
// state, which the loss reads.

#include <enzyme/fixed_point.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 4
#define C 0.4

void __enzyme_autodiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

static const double A[N][N] = {{1.0, 0.5, 0.0, -0.3},
                               {0.2, 1.0, 0.4, 0.0},
                               {0.0, -0.6, 1.0, 0.3},
                               {0.5, 0.0, 0.1, 1.0}};

// One iteration: whether it changed u by more than *tol.
__attribute__((noinline)) static int step(int64_t i, double *u, double *p,
                                          double *y, double *tol) {
  (void)i;
  double tmp[N];
  for (int k = 0; k < N; k++) {
    double s = 0;
    for (int j = 0; j < N; j++)
      s += A[k][j] * u[j];
    tmp[k] = C * sin(s) + p[k] * p[k];
  }
  double err = 0;
  for (int k = 0; k < N; k++) {
    y[k] = tmp[k] * u[k];
    err = fmax(err, fabs(tmp[k] - u[k]));
    u[k] = tmp[k];
  }
  return err > *tol;
}

static double loss(const double *u, const double *y) {
  double s = 0;
  for (int k = 0; k < N; k++)
    s += u[k] * u[k] * u[k] + y[k];
  return s;
}

static double plain(double *u, double *p, double *y, double *tol) {
  int64_t i = 0;
  while (step(i++, u, p, y, tol))
    ;
  return loss(u, y);
}

static double fixed(double *u, double *p, double *y, double *tol) {
  __enzyme_fixed_point((void *)step, enzyme_fp_state, u,
                       (int64_t)(N * sizeof(double)), enzyme_fp_reduction,
                       1e-30, u, p, y, tol);
  return loss(u, y);
}

static int controlCalls;
static double firstCumul;

// Tapenade's protocol, at most three adjoint iterations.
static int control(double *cumul, double *reduction) {
  (void)reduction;
  if (controlCalls++ == 0)
    firstCumul = *cumul;
  return controlCalls < 4;
}

static double controlled(double *u, double *p, double *y, double *tol) {
  __enzyme_fixed_point((void *)step, enzyme_fp_state, u,
                       (int64_t)(N * sizeof(double)), enzyme_fp_control,
                       control, u, p, y, tol);
  return loss(u, y);
}

static void init(double *u, double *p) {
  for (int k = 0; k < N; k++) {
    u[k] = 0.1 * k;
    p[k] = 0.3 + 0.2 * k;
  }
}

int main(void) {
  int failures = 0;
  double tol = 1e-15;

  double u0[N], p0[N], y0[N] = {0}, du0[N] = {0}, dp0[N] = {0}, dy0[N] = {0};
  init(u0, p0);
  __enzyme_autodiff((void *)plain, enzyme_dup, u0, du0, enzyme_dup, p0, dp0,
                    enzyme_dup, y0, dy0, enzyme_const, &tol);

  double u[N], p[N], y[N] = {0}, du[N] = {0}, dp[N] = {0}, dy[N] = {0};
  init(u, p);
  __enzyme_autodiff((void *)fixed, enzyme_dup, u, du, enzyme_dup, p, dp,
                    enzyme_dup, y, dy, enzyme_const, &tol);

  for (int k = 0; k < N; k++) {
    double err = fabs(dp[k] - dp0[k]) / (fabs(dp0[k]) + 1e-300);
    if (!(err < 1e-12)) {
      printf("dp[%d] = %.17g, expected %.17g\n", k, dp[k], dp0[k]);
      failures++;
    }
    if (du[k] != 0.0) {
      printf("du[%d] = %g, expected 0\n", k, du[k]);
      failures++;
    }
    if (dy[k] != 0.0) {
      printf("dy[%d] = %g, expected 0\n", k, dy[k]);
      failures++;
    }
  }
  if (memcmp(u, u0, sizeof(u))) {
    printf("state after the gradient differs\n");
    failures++;
  }

  // A control function that stops after three adjoint iterations: the
  // gradient is then only approximate, but close.
  double dpc[N] = {0};
  init(u, p);
  memset(du, 0, sizeof(du));
  memset(dy, 0, sizeof(dy));
  __enzyme_autodiff((void *)controlled, enzyme_dup, u, du, enzyme_dup, p, dpc,
                    enzyme_dup, y, dy, enzyme_const, &tol);
  if (controlCalls != 4 || firstCumul != -1.0) {
    printf("control called %d times, first with %g\n", controlCalls,
           firstCumul);
    failures++;
  }
  double diff = 0, ref = 0;
  for (int k = 0; k < N; k++) {
    diff = fmax(diff, fabs(dpc[k] - dp0[k]));
    ref = fmax(ref, fabs(dp0[k]));
  }
  if (!(diff > 0 && diff < 0.1 * ref)) {
    printf("three adjoint iterations: error %g of %g\n", diff, ref);
    failures++;
  }

  if (failures) {
    printf("%d failures\n", failures);
    return 1;
  }
  printf("ok\n");
  return 0;
}
