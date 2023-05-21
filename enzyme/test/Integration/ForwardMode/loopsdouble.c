// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

double __enzyme_fwddiff(void*, ...);

void compute_loops(float* a, float* b, float* ret) {
  double sum0 = 0.0;
  for (int i = 0; i < 100; i++) {
    double sum1 = 0.0;
    for (int j = 0; j < 100; j++) {
      sum1 += *a+*b;
    }
    sum0 += sum1;
  }
  *ret = sum0;
}

int main(int argc, char** argv) {
  float a = 2.0;
  float b = 3.0;

  float da = 1.0;
  float db = 1.0;

  float ret = 0;
  float dret = 1.0;

  __enzyme_fwddiff(compute_loops, &a, &da, &b, &db, &ret, &dret);

  APPROX_EQ(dret, 100*100*2.0f, 1e-10);

  printf("hello! %f, res2 %f, da: %f, db: %f\n", ret, dret, a, b);
  return 0;
}
