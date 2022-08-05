// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli -

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "test_utils.h"

double __enzyme_autodiff(void*, ...);

double log1p_like_function(double a) {
  return a;
}

double test(double a) {
  return log1p_like_function(a);
}

void* __enzyme_function_like[2] = {(void*)log1p_like_function, "log1p"}; 

int main(int argc, char** argv) {

  double out = __enzyme_autodiff(test, 2.0); 
  APPROX_EQ(out, 1/3.0, 1e-10); 

  return 0;
}
