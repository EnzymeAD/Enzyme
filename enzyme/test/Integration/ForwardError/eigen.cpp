// RUN: %clang++ -mllvm -force-vector-width=1 -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

#include "../test_utils.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

double __enzyme_error_estimate(double(double), double, double);

double square(double x) {
  Eigen::Vector3d v(x, x * x, x * x * x);
  v *= 2;
  return v[1];
}

double err_square(double x) { return __enzyme_error_estimate(square, x, 0.0); }

int main() {
  double x = 999.0;
  double res = err_square(x);
  APPROX_EQ(res, 2.32831e-10, 1e-8);
  printf("err_square(%e)=%e\n", x, res);
  return 0;
}