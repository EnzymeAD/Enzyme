// RUN: %clang++ -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include <cmath>
#include <iostream>
#include <string>

#include "../test_utils.h"

extern double __enzyme_error_estimate(void *, ...);

int valueLogCount = 0;
int errorLogCount = 0;

void enzymeLogValue(const char *id, double res, unsigned numOperands,
                    double *operands) {
  ++valueLogCount;
  std::cout << "Id = " << id << ", Res = " << res << "\n";
  for (int i = 0; i < numOperands; ++i) {
    std::cout << "\tOperand[" << i << "] = " << operands[i] << "\n";
  }
}

void enzymeLogError(const char *id, double err) {
  ++errorLogCount;
  std::cout << "Id = " << id << ", Err = " << err << "\n";
}

// An example from https://dl.acm.org/doi/10.1145/3371128
double fun(double x) {
  double v1 = cos(x);
  double v2 = 1 - v1;
  double v3 = x * x;
  double v4 = v2 / v3;
  double v5 = sin(v4); // Inactive -- logger is not invoked.

  std::cout << "v1 = " << v1 << ", v2 = " << v2 << ", v3 = " << v3
            << ", v4 = " << v4 << ", v5 = " << v5 << "\n";

  return v4;
}

int main() {
  double res = fun(1e-7);
  double error = __enzyme_error_estimate((void *)fun, 1e-7, 0.0);
  std::cout << "res = " << res << ", abs error = " << error
            << ", rel error = " << fabs(error / res) << "\n";
  APPROX_EQ(error, 2.2222222222e-2, 1e-4);
  TEST_EQ(valueLogCount, 4);
  TEST_EQ(errorLogCount, 4);
}
