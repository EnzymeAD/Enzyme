// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

#include <stdio.h>

#include "../test_utils.h"

extern void __enzyme_autodiff(void *, ...);

void f(float *x, float *y, float *out) {
  *out = __builtin_powf(*x, *y);
}

int main() {
  float x = 0.0f;
  float y = 1.0f;
  float dx = 0.0f;
  float dy = 0.0f;
  float out = 0.0f;
  float dout = 1.0f;

  __enzyme_autodiff((void *)f, &x, &dx, &y, &dy, &out, &dout);

  APPROX_EQ(out, 0.0f, 1e-7);
  APPROX_EQ(dx, 1.0f, 1e-7);
  APPROX_EQ(dy, 0.0f, 1e-7);
  return 0;
}
