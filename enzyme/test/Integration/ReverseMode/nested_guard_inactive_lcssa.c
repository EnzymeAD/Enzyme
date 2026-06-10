// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -

#include "../test_utils.h"
#include <stdbool.h>
#include <string.h>

void __enzyme_autodiff(void *, ...);

__attribute__((noinline)) double kernel(double *fvec, bool fan) {
  double unit_rad[3];
  double radius = 0.0;

  if (fan) {
    memcpy(&unit_rad, fvec, 8);
    for (int i = 0; i < 2; ++i) {
      radius += fvec[i];
    }
    radius = __builtin_sqrt(radius);

    if (radius > 0.0) {
      for (int i = 0; i < 2; ++i) {
        unit_rad[i] /= radius;
      }
    }
  }

  return fvec[0];
}

int main(void) {
  double fvec[3] = {1.0, 2.0, 3.0};
  double dfvec[3] = {0.0, 0.0, 0.0};
  bool fan = false;

  __enzyme_autodiff((void *)kernel, enzyme_dup, fvec, dfvec, enzyme_const, fan);

  APPROX_EQ(dfvec[0], 1.0, 1e-10);
  APPROX_EQ(dfvec[1], 0.0, 1e-10);
  APPROX_EQ(dfvec[2], 0.0, 1e-10);
  return 0;
}
