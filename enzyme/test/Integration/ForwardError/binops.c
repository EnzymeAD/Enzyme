// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include "../test_utils.h"

extern double __enzyme_error_estimate(void *, ...);

double fun(double x1, double x2)
{
    double w1 = x1;
    double w2 = x2;
    double w3 = w1 * w2;
    double w4 = 1.0 / w1;
    double w5 = w3 + w4;
    return w5;
}

int main()
{
    double error = __enzyme_error_estimate((void *)fun, 3.0, 0.0, 2.7, 0.0);
    printf("Found floating point error of %f\n", error);
    APPROX_EQ(error, 0, 1e-10);
}
