// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-global-activity=1 | %lli - ; fi

// A fixed-point loop of a constant trip count, small enough that -O2 would
// fully unroll it before Enzyme runs, leaving the marker outside any loop.
// Enzyme keeps marked loops from being unrolled. The iteration contracts
// fast enough to converge within the count, so the gradient must equal the
// gradient through every iteration of the same loop without the marker.

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define N 4
#define ITERS 16

void __enzyme_autodiff(void *, ...);
void __enzyme_set_fixed_point(double reduction, int64_t max_iters,
                              void *control, ...);
extern int enzyme_dup;

double u[N], p[N];

__attribute__((noinline)) static void iterate(void) {
  double tmp[N];
  for (int k = 0; k < N; k++)
    tmp[k] = 0.05 * sin(u[k] + u[(k + 1) % N]) + p[k] * p[k];
  for (int k = 0; k < N; k++)
    u[k] = tmp[k];
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

static double plain(const double *x) {
  setup(x);
  for (int i = 0; i < ITERS; i++)
    iterate();
  return loss();
}

static double fixed(const double *x) {
  setup(x);
  for (int i = 0; i < ITERS; i++) {
    __enzyme_set_fixed_point(1e-30, (int64_t)-1, (void *)0, u,
                             (int64_t)sizeof(u));
    iterate();
  }
  return loss();
}

int main(void) {
  double x[N], d0[N] = {0}, d[N] = {0};
  for (int k = 0; k < N; k++)
    x[k] = 0.3 + 0.1 * k;
  __enzyme_autodiff((void *)plain, enzyme_dup, x, d0);
  __enzyme_autodiff((void *)fixed, enzyme_dup, x, d);
  int failures = 0;
  for (int k = 0; k < N; k++)
    if (!(fabs(d[k] - d0[k]) <= 1e-12 * fabs(d0[k]))) {
      printf("dx[%d] = %.17g, expected %.17g\n", k, d[k], d0[k]);
      failures++;
    }
  if (failures)
    return 1;
  printf("ok\n");
  return 0;
}
