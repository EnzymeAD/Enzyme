// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// The forward derivative of x / y must not square y: y * y overflows for
// |y| > ~1e154 (and underflows for |y| < ~1e-154) although x / y and its
// derivative are representable. Under nested forward mode each level squares
// the denominator again, so y^(2^n) overflows at modest n and |y|, giving
// Inf / Inf = NaN.

#include "../test_utils.h"

extern double __enzyme_fwddiff(void *, ...);

double f(double x, double y) { return x / y; }

// d/dx (x / y)
double dfdx(double x, double y) {
  return __enzyme_fwddiff((void *)f, x, 1.0, y, 0.0);
}

// d/dy (x / y)
double dfdy(double x, double y) {
  return __enzyme_fwddiff((void *)f, x, 0.0, y, 1.0);
}

// d^2/dy^2 (x / y) = 2 x / y^3
double d2fdy2(double x, double y) {
  return __enzyme_fwddiff((void *)dfdy, x, 0.0, y, 1.0);
}

// d^2/dx dy (x / y) = -1 / y^2
double d2fdxdy(double x, double y) {
  return __enzyme_fwddiff((void *)dfdx, x, 0.0, y, 1.0);
}

int main() {
  // Old rule: (dx * y) / (y * y) = 1e160 / Inf = 0.
  double y = 1e160;
  APPROX_EQ(dfdx(2.0, y) * y, 1.0, 1e-12);
  // Old rule: (-dy * x) / (y * y) = -1e160 / Inf = -0.
  APPROX_EQ(dfdy(y, y) * y, -1.0, 1e-12);

  // Old rule: y * y underflows to a subnormal and loses precision.
  y = 1e-160;
  APPROX_EQ(dfdx(2.0, y) * y, 1.0, 1e-12);

  // Old rule: the second-order denominator (y * y) * (y * y) overflows to
  // Inf, giving 0.
  y = 1e80;
  APPROX_EQ(d2fdy2(1.0, y) * (y * y) * y, 2.0, 1e-12);
  APPROX_EQ(d2fdxdy(1.0, y) * y * y, -1.0, 1e-12);
  return 0;
}
