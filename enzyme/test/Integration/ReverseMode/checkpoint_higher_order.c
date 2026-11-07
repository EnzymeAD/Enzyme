// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// A Hessian-vector product through a checkpointed loop, reverse mode over
// forward mode: the forward-mode derivative of the loop is differentiated
// again, and must give what the plain loop gives. (It is not checkpointed
// yet: the outer reverse pass keeps the tangent loop's tape.)

#include "../test_utils.h"
#include <enzyme/checkpoint.h>
#include <math.h>
#define N 4
void __enzyme_autodiff(void *, ...);
double __enzyme_fwddiff(void *, ...);
extern int enzyme_dup, enzyme_const, enzyme_dupnoneed;

__attribute__((noinline)) static void step(int64_t i, double *u) {
  double tmp[N];
  for (int k = 0; k < N; k++)
    tmp[k] =
        u[k] + 0.1 * u[(k + 1) % N] * u[k] + 0.01 * sin(u[k] + 0.1 * (double)i);
  for (int k = 0; k < N; k++)
    u[k] = tmp[k];
}
static double loss(const double *u) {
  double s = 0;
  for (int k = 0; k < N; k++)
    s += u[k] * u[k] * u[k];
  return s;
}
static double plain(double *u, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    step(i, u);
  return loss(u);
}
static double ckpt(double *u, int64_t n, EnzymeCkptConfig *c) {
  __enzyme_checkpoint_for((void *)step, 0, n, enzyme_scheme, &EnzymeCkptRevolve,
                          c, enzyme_checkpoint_region, u,
                          (int64_t)(N * sizeof(double)), u);
  return loss(u);
}

// JVP of the loss along v, from a copy of x
__attribute__((noinline)) static double jvp_plain(const double *x,
                                                  const double *v, int64_t n) {
  double u[N], du[N];
  for (int k = 0; k < N; k++) {
    u[k] = x[k];
    du[k] = v[k];
  }
  return __enzyme_fwddiff((void *)plain, enzyme_dup, u, du, enzyme_const, n);
}
__attribute__((noinline)) static double
jvp_ckpt(const double *x, const double *v, int64_t n, EnzymeCkptConfig *c) {
  double u[N], du[N];
  for (int k = 0; k < N; k++) {
    u[k] = x[k];
    du[k] = v[k];
  }
  return __enzyme_fwddiff((void *)ckpt, enzyme_dup, u, du, enzyme_const, n,
                          enzyme_const, c);
}
int main(void) {
  int64_t n = 7;
  EnzymeCkptConfig c = {2, 0, NULL, 0, NULL};
  double x[N] = {0.5, 0.7, 0.9, 1.1}, v[N] = {1, -1, 0.5, 2};
  double Hv_plain[N] = {0}, Hv_ckpt[N] = {0};
  __enzyme_autodiff((void *)jvp_plain, enzyme_dup, x, Hv_plain, enzyme_const, v,
                    enzyme_const, n);
  __enzyme_autodiff((void *)jvp_ckpt, enzyme_dup, x, Hv_ckpt, enzyme_const, v,
                    enzyme_const, n, enzyme_const, &c);
  for (int k = 0; k < N; k++)
    APPROX_EQ(Hv_ckpt[k], Hv_plain[k], 1e-10);
  return 0;
}
