// COM: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// COM: %clang -O2 -ffast-math %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// COM: %clang -O1 -g %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include <math.h>

#include "../test_utils.h"

#define N 10

double simple_add(double a, double b) {
    return a + b;
}
double intrinsics(double a, double b) {
    return sqrt(a) * pow(b, 2);
}
// TODO
double constt(double a, double b) {
    return 2;
}
double compute(double *A, double *B, double *C, int n) {
  for (int i = 0; i < n; i++) {
      C[i] = A[i] * 2;
  }
  return C[0];
}

typedef double (*fty)(double *, double *, double *, int);

typedef double (*fty2)(double, double);

extern fty __enzyme_truncate_func_2(...);
extern fty2 __enzyme_truncate_func(...);
extern double __enzyme_truncate_value(...);
extern double __enzyme_expand_value(...);

#define FROM 64
#define TO 32

#define TEST(F) do {


int main() {

    {
        double a = 1;
        APPROX_EQ(
            __enzyme_expand_value(
                __enzyme_truncate_value(a, FROM, TO) , FROM, TO),
            a, 1e-10);
    }

    {
        double a = 2;
        double b = 3;
        double truth = simple_add(a, b);
        a = __enzyme_truncate_value(a, FROM, TO);
        b = __enzyme_truncate_value(b, FROM, TO);
        double trunc = __enzyme_expand_value(__enzyme_truncate_func(simple_add, FROM, TO)(a, b), FROM, TO);
        APPROX_EQ(trunc, truth, 1e-5);
    }
    {
        double a = 2;
        double b = 3;
        double truth = intrinsics(a, b);
        a = __enzyme_truncate_value(a, FROM, TO);
        b = __enzyme_truncate_value(b, FROM, TO);
        double trunc = __enzyme_expand_value(__enzyme_truncate_func(intrinsics, FROM, TO)(a, b), FROM, TO);
        APPROX_EQ(trunc, truth, 1e-5);
    }
    // {
    //     double a = 2;
    //     double b = 3;
    //     double truth = intrinsics(a, b);
    //     a = __enzyme_truncate_value(a, FROM, TO);
    //     b = __enzyme_truncate_value(b, FROM, TO);
    //     double trunc = __enzyme_expand_value(__enzyme_truncate_func(constt, FROM, TO)(a, b), FROM, TO);
    //     APPROX_EQ(trunc, truth, 1e-5);
    // }

    // double A[N];
    // double B[N];
    // double C[N];
    // double D[N];


    // for (int i = 0; i < N; i++) {
    //     A[i] = 1 + i % 5;
    //     B[i] = 1 + i % 3;
    // }

    // compute(A, B, D, N);

    // for (int i = 0; i < N; i++) {
    //     A[i] = __enzyme_truncate_value(A[i], 64, 32);
    //     B[i] = __enzyme_truncate_value(B[i], 64, 32);
    // }

    // __enzyme_truncate_func_2(compute, 64, 32)(A, B, C, N);

    // for (int i = 0; i < N; i++) {
    //     C[i] = __enzyme_expand_value(C[i], 64, 32);
    // }

    // for (int i = 0; i < N; i++) {
    //     printf("%d\n", i);
    //     APPROX_EQ(D[i], C[i], 1e-5);
    // }

}
