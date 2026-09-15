// RUN: %clang++ -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang++ -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang++ -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang++ -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -

#include "../test_utils.h"
#include <cmath>
#include <cstring>

extern "C" {
void __enzyme_autodiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;
}

__attribute__((noinline)) double first(double *fvec, bool fan) {
  double unit_rad[3];
  double radius = 0.0;
  if (fan) {
    memcpy(&unit_rad, fvec, 8);
    for (int i = 0; i < 2; ++i)
      radius += fvec[i];
    radius = std::sqrt(radius);
    if (radius > 0.0)
      for (int i = 0; i < 2; ++i)
        unit_rad[i] /= radius;
  }
  return fvec[0];
}

__attribute__((noinline)) void second(double *fvec, bool fan, double *out) {
  double unit_rad[3] = {0.0, 0.0, 0.0};
  double radius = 0.0;
  if (fan) {
    for (int i = 0; i < 3; ++i) {
      unit_rad[i] = fvec[i];
      radius += unit_rad[i] * unit_rad[i];
    }
    radius = std::sqrt(radius);
    if (radius > 0.0)
      for (int i = 0; i < 3; ++i)
        unit_rad[i] /= radius;
  }
  *out = fvec[0] + fvec[1] + fvec[2];
}

__attribute__((noinline)) void check(bool fan) {
  double fvec[3] = {1.0, 2.0, 3.0};
  double dfvec[3] = {0.0, 0.0, 0.0};
  double out = 0.0, dout = 1.0;
  __enzyme_autodiff((void *)second, enzyme_dup, fvec, dfvec, enzyme_const, fan,
                    enzyme_dup, &out, &dout);
  for (int i = 0; i < 3; ++i)
    APPROX_EQ(dfvec[i], 1.0, 1e-12);

  // The first example only initializes one element of unit_rad. Exercise the
  // skipped path, where the original program does not read uninitialized data.
  if (!fan) {
    for (int i = 0; i < 3; ++i)
      dfvec[i] = 0.0;
    __enzyme_autodiff((void *)first, enzyme_dup, fvec, dfvec, enzyme_const,
                      fan);
    APPROX_EQ(dfvec[0], 1.0, 1e-12);
    APPROX_EQ(dfvec[1], 0.0, 1e-12);
    APPROX_EQ(dfvec[2], 0.0, 1e-12);
  }
}

int main() {
  check(false);
  check(true);
  return 0;
}
