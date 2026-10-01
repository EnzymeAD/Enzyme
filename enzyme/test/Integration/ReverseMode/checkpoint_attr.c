// RUN: if [ %llvmver -ge 17 ]; then %clang -std=c2x -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -std=c2x -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -std=c2x -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -std=c2x -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -x c++ -std=c++17 -DCXX -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// A loop annotated for checkpointing, as in Enzyme-MLIR and Reactant:
// [[enzyme_checkpointing_enable("binomial" or "regular", count)]] on the for
// statement. Its state is a global, a malloc'ed array, a local array and a
// scalar carried from one iteration to the next. The gradient and the value
// must be those of the same loop without the annotation, for every mode,
// including the default (periodic, with the square root of the number of
// steps as the budget).

#include <enzyme/checkpoint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 4
#ifdef CXX
#define CKPT(...) [[enzyme::checkpointing_enable(__VA_ARGS__)]]
#define CKPT0 [[enzyme::checkpointing_enable]]
extern "C" void __enzyme_autodiff(void *, ...);
#else
#define CKPT(...) [[enzyme_checkpointing_enable(__VA_ARGS__)]]
#define CKPT0 [[enzyme_checkpointing_enable]]
void __enzyme_autodiff(void *, ...);
#endif
extern int enzyme_dup, enzyme_const;
double g;

#define BODY                                                                   \
  {                                                                            \
    double tmp[N];                                                             \
    for (int k = 0; k < N; k++)                                                \
      tmp[k] = u[k] + 0.1 * u[(k + 1) % N] * g + 0.01 * sin(u[k] + 0.1 * i);  \
    for (int k = 0; k < N; k++)                                                \
      u[k] = tmp[k];                                                           \
    g = 0.9 * g + 0.1 * u[0];                                                  \
    acc += u[1] * g;                                                           \
  }

#define RUN(NAME, ATTR)                                                        \
  __attribute__((noinline)) double NAME(const double *x, long n) {             \
    double *u = (double *)malloc(N * sizeof(double));                          \
    for (int k = 0; k < N; k++)                                                \
      u[k] = x[k];                                                             \
    double acc = 0;                                                            \
    g = 0.5;                                                                   \
    ATTR for (long i = 0; i < n; i++) BODY                                     \
    double r = acc + g;                                                        \
    for (int k = 0; k < N; k++)                                                \
      r += u[k] * u[k];                                                        \
    free(u);                                                                   \
    return r;                                                                  \
  }

RUN(plain, )
RUN(binomial, CKPT("binomial", 2))
RUN(regular, CKPT("regular", 3))
RUN(dflt, CKPT0)

#define GRAD(NAME)                                                             \
  static void grad_##NAME(long n, double *dx) {                                \
    double x[N] = {0.3, 0.7, 1.1, 1.5};                                        \
    memset(dx, 0, N * sizeof(double));                                         \
    __enzyme_autodiff((void *)NAME, enzyme_dup, x, dx, enzyme_const, n);       \
  }
GRAD(plain)
GRAD(binomial)
GRAD(regular)
GRAD(dflt)
typedef double (*fn)(const double *, long);
typedef void (*gfn)(long, double *);

int main(void) {
  long steps[] = {0, 1, 2, 7, 20};
  fn fs[] = {binomial, regular, dflt};
  gfn gs[] = {grad_binomial, grad_regular, grad_dflt};
  const char *names[] = {"binomial", "regular", "default"};
  int failures = 0;
  for (unsigned a = 0; a < 5; a++) {
    double want[N], got[N];
    double x[N] = {0.3, 0.7, 1.1, 1.5};
    grad_plain(steps[a], want);
    for (unsigned s = 0; s < 3; s++) {
      if (fs[s](x, steps[a]) != plain(x, steps[a])) {
        printf("%s n=%ld: primal differs\n", names[s], steps[a]);
        failures++;
      }
      gs[s](steps[a], got);
      for (int k = 0; k < N; k++)
        if (!(fabs(got[k] - want[k]) <= 1e-12 * (fabs(want[k]) + 1e-300))) {
          printf("%s n=%ld: dx[%d] = %.17g, expected %.17g\n", names[s],
                 steps[a], k, got[k], want[k]);
          failures++;
        }
    }
  }
  if (failures)
    return printf("%d failures\n", failures), 1;
  printf("ok\n");
  return 0;
}
