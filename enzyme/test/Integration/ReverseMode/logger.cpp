// RUN: %clang++ -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"
#include <cmath>

extern double __enzyme_autodiff(void *, ...);

int errorLogCount = 0;

void enzymeLogGrad(double res, double grad, const char *opcodeName,
                   const char *calleeName, const char *moduleName,
                   const char *functionName, unsigned blockIdx,
                   unsigned instIdx, unsigned numOperands, double *operands) {
  ++errorLogCount;
  printf("Res = %e, Grad = %e, Op = %s, Callee = %s, Module = %s, Function = "
         "%s, BlockIdx = %u, InstIdx = %u\n",
         res, grad, opcodeName, calleeName, moduleName, functionName, blockIdx,
         instIdx);
  for (int i = 0; i < numOperands; ++i) {
    printf("Operand[%d] = %e\n", i, operands[i]);
  }
}

double fun(double x) {
  double v1 = x * 3;
  double v2 = 1 - v1;
  double v3 = x * x;
  double v4 = v2 / v3;
  double v5 = v3 + v4;

  printf("v1 = %.18e, v2 = %.18e, v3 = %.18e, v4 = %.18e, v5 = %.18e\n", v1, v2,
         v3, v4, v5);

  return v5;
}

int main() {
  double x = 2.0;
  double res = fun(x);
  double grad_x = __enzyme_autodiff((void *)fun, x);
  printf("res = %.18e, grad = %.18e\n", res, grad_x);
  APPROX_EQ(grad_x, 4.5, 1e-4);
  TEST_EQ(errorLogCount, 5);
}
