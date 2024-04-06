// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

#include "../test_utils.h"

double __enzyme_error_estimate(void *, ...);

double compute(double a, double b) { return a + b; }

void compute_ptr(double *a, double *b, double *ret) { *ret = *a + *b; }

int main(int argc, char **argv) {
  double a = 99.0;
  double b = 127.0;

  double err_a = 0.0; //(double*) malloc(sizeof(double));
  double err_b = 0.0; //(double*) malloc(sizeof(double));

  double ret_ptr = 0;
  double err_ret_ptr = 0.0;

  double err_ret = __enzyme_error_estimate(compute, a, err_a, b, err_b);

  __enzyme_error_estimate(compute_ptr, &a, &err_a, &b, &err_b, &ret_ptr,
                          &err_ret_ptr);

  APPROX_EQ(err_ret_ptr, 2.84217e-14, 1e-17);
  APPROX_EQ(err_ret_ptr, err_ret, 1e-17);

  printf("ret = %e, err_ret = %e, err_ret_ptr = %e\n", ret_ptr, err_ret,
         err_ret_ptr);

  return 0;
}
