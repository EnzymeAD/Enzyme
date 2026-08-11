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

extern float __enzyme_fwddiff(float (*)(float, float), float, float, float,
                              float);

float f(float x, float y) { return __builtin_powf(x, y); }

int main() {
  float x = 0.0f;
  float y = 1.0f;
  float dx = __enzyme_fwddiff(f, x, 1.0f, y, 0.0f);
  float dy = __enzyme_fwddiff(f, x, 0.0f, y, 1.0f);

  APPROX_EQ(f(x, y), 0.0f, 1e-7);
  APPROX_EQ(dx, 1.0f, 1e-7);
  APPROX_EQ(dy, 0.0f, 1e-7);
  return 0;
}
