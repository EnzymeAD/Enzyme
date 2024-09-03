// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

double sin(double);
double cos(double);
double fabs(double);

extern double __enzyme_error_estimate(void *, ...);

int valueLogCount = 0;
int errorLogCount = 0;

void enzymeLogValue(const char *id, double res, unsigned numOperands,
                    double *operands) {
  ++valueLogCount;
  printf("Id = %s, Res = %.18e\n", id, res);
  for (int i = 0; i < numOperands; ++i) {
    printf("\tOperand[%d] = %.18e\n", i, operands[i]);
  }
}

void enzymeLogError(const char *id, double err) {
  ++errorLogCount;
  printf("Id = %s, Err = %.18e\n", id, err);
}

// An example from https://dl.acm.org/doi/10.1145/3371128
double fun(double x) {
  double v1 = cos(x);
  double v2 = 1 - v1;
  double v3 = x * x;
  double v4 = v2 / v3;
  double v5 = sin(v4);

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
  TEST_EQ(valueLogCount, 4); // TODO: should be 5
  TEST_EQ(errorLogCount, 4);
}
