// RUN: %clang -std=c11 -O2 -flto=thin -DSINGLE_TU -c %s -emit-llvm -o %t.thin.bc
// RUN: %opt %loadLLDEnzymeOpt -passes="internalize,thinlto<O2>" -internalize-public-api-list=main %t.thin.bc -S -o - | %lli -
// RUN: %clang -std=c11 -O2 -flto -DSINGLE_TU -c %s -emit-llvm -o %t.full.bc
// RUN: %opt %loadLLDEnzymeOpt -passes="internalize,lto<O2>" -internalize-public-api-list=main %t.full.bc -S -o - | %lli -
// RUN: %clang -std=c11 -O2 -flto=thin -DSQUARE_DEF -c %s -emit-llvm -o %t.def.bc
// RUN: %clang -std=c11 -O2 -flto=thin -c %s -emit-llvm -o %t.use.bc
// RUN: llvm-link %t.def.bc %t.use.bc -o %t.linked.bc
// RUN: %opt %loadLLDEnzymeOpt -passes="internalize,thinlto<O2>" -internalize-public-api-list=main %t.linked.bc -S -o - | %lli -

#include <stdio.h>
#include <stdlib.h>

#include "../test_utils.h"

double __enzyme_autodiff(void *, double);

#if defined(SINGLE_TU) || defined(SQUARE_DEF)
#ifdef SINGLE_TU
__attribute__((noinline))
#endif
void square_(const double *src, double *dest) {
  *dest = *src * *src;
}

int augment = 0;
void *augment_square_(const double *src, const double *d_src, double *dest,
                      double *d_dest) {
  augment++;
  // Intentionally incorrect to distinguish custom gradient from AD of square_.
  *dest = 7.0;
  *d_dest = 11.0;
  return NULL;
}

int gradient = 0;
void gradient_square_(const double *src, double *d_src, const double *dest,
                      const double *d_dest, void *tape) {
  gradient++;
  // Intentionally incorrect to distinguish custom gradient from AD of square_.
  *d_src = 13.0;
}

void *__enzyme_register_gradient_square[] = {
    (void *)square_,
    (void *)augment_square_,
    (void *)gradient_square_,
};
#endif

#if !defined(SQUARE_DEF)
extern int augment;
extern int gradient;
void square_(const double *src, double *dest);

double square(double x) {
  double y;
  square_(&x, &y);
  return y;
}

double dsquare(double x) { return __enzyme_autodiff((void *)square, x); }

int main(void) {
  double res = dsquare(3.0);
  APPROX_EQ(res, 13.0, 1e-10);
  if (augment != 1 || gradient != 1)
    abort();
  return 0;
}
#endif
