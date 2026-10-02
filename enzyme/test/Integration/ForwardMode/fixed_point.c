// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// Forward mode over __enzyme_fixed_point: the tangent is iterated at the
// converged state, s <- A s + B h, until it stops changing. It must equal the
// tangent through every iteration of the plain loop, for the state and for an
// output the step writes, with the primal state as the plain loop leaves it.
// For a Newton iteration, whose step has a zero Jacobian in the state at the
// fixed point, two tangent passes suffice: one, and one to see it converged.

#include <enzyme/fixed_point.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 4
#define C 0.4

void __enzyme_fwddiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

static const double A[N][N] = {{1.0, 0.5, 0.0, -0.3},
                               {0.2, 1.0, 0.4, 0.0},
                               {0.0, -0.6, 1.0, 0.3},
                               {0.5, 0.0, 0.1, 1.0}};

// One iteration of u = C sin(A u) + p^2: whether it changed u by more than
// *tol.
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

static void plain(double *u, double *p, double *y, double *tol) {
  int64_t i = 0;
  while (step(i++, u, p, y, tol))
    ;
}

static void fixed(double *u, double *p, double *y, double *tol) {
  __enzyme_fixed_point((void *)step, enzyme_fp_state, u,
                       (int64_t)(N * sizeof(double)), enzyme_fp_reduction,
                       1e-30, u, p, y, tol);
}

// Newton's iteration for the cube roots of a.
__attribute__((noinline)) static int newton(int64_t i, double *x, double *a,
                                            double *tol) {
  (void)i;
  double err = 0;
  for (int k = 0; k < N; k++) {
    double next = (2 * x[k] + a[k] / (x[k] * x[k])) / 3;
    err = fmax(err, fabs(next - x[k]));
    x[k] = next;
  }
  return err > *tol;
}

static int controlCalls;
static double controlRef = -1;

// Tapenade's protocol, with the built-in relative test.
static int control(double *cumul, double *reduction) {
  controlCalls++;
  if (*cumul < 0) {
    controlRef = -1;
    return 1;
  }
  if (controlRef < 0) {
    controlRef = *cumul;
    return *cumul > 0;
  }
  return *cumul > *reduction * controlRef;
}

static void roots(double *x, double *a, double *tol) {
  __enzyme_fixed_point((void *)newton, enzyme_fp_state, x,
                       (int64_t)(N * sizeof(double)), enzyme_fp_reduction,
                       1e-24, enzyme_fp_control, control, x, a, tol);
}

static void init(double *u, double *p) {
  for (int k = 0; k < N; k++) {
    u[k] = 0.1 * k;
    p[k] = 0.3 + 0.2 * k;
  }
}

static int near(double a, double b, double rtol) {
  return fabs(a - b) <= rtol * fabs(b) + 1e-300;
}

int main(void) {
  int failures = 0;
  double tol = 1e-15;
  double dp[N] = {1.0, -0.5, 0.25, 2.0};

  double u0[N], p0[N], y0[N] = {0}, du0[N] = {0}, dy0[N] = {0};
  init(u0, p0);
  __enzyme_fwddiff((void *)plain, enzyme_dup, u0, du0, enzyme_dup, p0, dp,
                   enzyme_dup, y0, dy0, enzyme_const, &tol);

  // The tangent of the initial guess has no effect.
  double u[N], p[N], y[N] = {0}, du[N] = {5, 5, 5, 5}, dy[N] = {0};
  init(u, p);
  __enzyme_fwddiff((void *)fixed, enzyme_dup, u, du, enzyme_dup, p, dp,
                   enzyme_dup, y, dy, enzyme_const, &tol);

  for (int k = 0; k < N; k++) {
    if (!near(du[k], du0[k], 1e-12)) {
      printf("du[%d] = %.17g, expected %.17g\n", k, du[k], du0[k]);
      failures++;
    }
    if (!near(dy[k], dy0[k], 1e-10)) {
      printf("dy[%d] = %.17g, expected %.17g\n", k, dy[k], dy0[k]);
      failures++;
    }
  }
  if (memcmp(u, u0, sizeof(u))) {
    printf("state after the tangent differs\n");
    failures++;
  }

  double x[N], a[N], dx[N] = {0}, da[N] = {1, 1, 1, 1};
  for (int k = 0; k < N; k++) {
    x[k] = 1;
    a[k] = 2.0 + k;
  }
  __enzyme_fwddiff((void *)roots, enzyme_dup, x, dx, enzyme_dup, a, da,
                   enzyme_const, &tol);
  for (int k = 0; k < N; k++) {
    double exact = cbrt(a[k]) / (3 * a[k]);
    if (!near(dx[k], exact, 1e-12)) {
      printf("dx[%d] = %.17g, expected %.17g\n", k, dx[k], exact);
      failures++;
    }
  }
  // The start of the protocol, and two passes.
  if (controlCalls != 3) {
    printf("control called %d times\n", controlCalls);
    failures++;
  }

  if (failures) {
    printf("%d failures\n", failures);
    return 1;
  }
  printf("ok\n");
  return 0;
}
