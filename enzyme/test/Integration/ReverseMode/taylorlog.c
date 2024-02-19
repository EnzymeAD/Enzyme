// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 

#include "../test_utils.h"

#define __builtin_autodiff __enzyme_autodiff

double __enzyme_autodiff(void*, double, unsigned);

// May be needed if not built with compiler-rt
double __powidf2(double a, int b) {
  const int recip = b < 0;
  double r = 1;
  while (1) {
    if (b & 1)
      r *= a;
    b /= 2;
    if (b == 0)
      break;
    a *= a;
  }
  return recip ? 1 / r : r;
}

static double taylorlog(double x, unsigned SINCOSN) {
  double sum = 0;
  for(int i=1; i<=SINCOSN; i++) {
    sum += __builtin_pow(x, i) / i;
    //sum += pow(x, i) / i;
  }
  return sum;
}

int main(int argc, char** argv) {
  
  double ret = __builtin_autodiff(taylorlog, 0.5, 10000);

  APPROX_EQ(ret, 2.0, 1e-7);

  return 0;
}
