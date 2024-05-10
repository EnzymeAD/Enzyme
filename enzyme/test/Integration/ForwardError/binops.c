// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

double sin(double);
double cos(double);
double fabs(double);

extern double __enzyme_error_estimate(void *, ...);

int errorLogCount = 0;

void enzymeLogError(double res, double err, const char *opcodeName,
                    const char *calleeName, const char *moduleName,
                    const char *functionName, const char *blockName) {
  ++errorLogCount;
  printf("Res = %e, Error = %e, Op = %s, Callee = %s, Module = %s, Function = "
         "%s, BasicBlock = %s\n",
         res, err, opcodeName, calleeName, moduleName, functionName, blockName);
}

// An example from https://dl.acm.org/doi/10.1145/3371128
double fun(double x) {
  double v1 = cos(x);
  double v2 = 1 - v1;
  double v3 = x * x;
  double v4 = v2 / v3;
  double v5 = sin(v4); // Inactive -- logger is not invoked.

  printf("v1 = %.18e, v2 = %.18e, v3 = %.18e, v4 = %.18e, v5 = %.18e\n", v1, v2,
         v3, v4, v5);

  return v4;
}

int main() {
  double res = fun(1e-7);
  double error = __enzyme_error_estimate((void *)fun, 1e-7, 0.0);
  printf("res = %.18e, abs error = %.18e, rel error = %.18e\n", res, error,
         fabs(error / res));
  APPROX_EQ(error, 2.2222222222e-2, 1e-4);
  TEST_EQ(errorLogCount, 4);
}
