// clang-format off
// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang                -DTRUNC_OP -O0                %s -o %s.a.out %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -lm -lmpfr && %s.a.out ; fi
// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang                -DTRUNC_OP -O2    -ffast-math %s -o %s.a.out %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -lm -lmpfr && %s.a.out ; fi
// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang                           -O1 -g             %s -o %s.a.out %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -lm -lmpfr && %s.a.out ; fi
// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang    -DTRUNC_MEM -DTRUNC_OP -O2                %s -o %s.a.out %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -lm -lmpfr && %s.a.out ; fi
// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -g -DTRUNC_MEM -DTRUNC_OP -O2                %s -o %s.a.out %newLoadClangEnzyme -include enzyme/fprt/mpfr.h -lm -lmpfr && %s.a.out ; fi

// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang    -DTRUNC_MEM -DTRUNC_OP -O2                %s -o %s.a.out %newLoadClangEnzyme -include enzyme/fprt/mpfr-test.h -lm -lmpfr && %s.a.out ; fi
// RUN: if [ %llvmver -ge 12 ] && [ %hasMPFR == "yes" ] ; then %clang -g -DTRUNC_MEM -DTRUNC_OP -O2                %s -o %s.a.out %newLoadClangEnzyme -include enzyme/fprt/mpfr-test.h -lm -lmpfr && %s.a.out ; fi

#include <math.h>

#include "../test_utils.h"

#define N 10

double simple_add(double a, double b) {
    return a + b;
}
double simple_cmp(double a, double b) {
    if (a > b)
        return a * 2;
    else
        return b + a;
}
double intrinsics(double a, double b) {
    return sqrt(a) * pow(b, 2);
}
// TODO trunc mem mode
double constt(double a, double b) {
    return 2;
}
void const_store(double *a) {
    *a = 2.0;
}
double phinode(double a, double b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (exp(a + b) - exp(a)) / b;
        b /= 10;
    }
    return sum;
}
double compute(double *A, double *B, double *C, int n) {
  for (int i = 0; i < n; i++) {
    C[i] = A[i] * 2 + B[i] * sqrt(A[i]);
  }
  return C[0];
}
double intcast(int a) {
    double d = (double) a;
    return d / 3.14;
}

typedef double (*fty)(double *, double *, double *, int);

typedef double (*fty2)(double, double);

template <typename fty> fty *__enzyme_truncate_mem_func(fty *, int, int);
template <typename fty> fty *__enzyme_truncate_op_func(fty *, int, int);
extern double __enzyme_truncate_mem_value(...);
extern double __enzyme_expand_mem_value(...);

#define FROM 64
#define TO 32

#define TEST(F) do {


int main() {

    #ifdef TRUNC_MEM
    {
        double a = 1;
        APPROX_EQ(
            __enzyme_expand_mem_value(
                __enzyme_truncate_mem_value(a, FROM, TO) , FROM, TO),
            a, 1e-10);
    }

    {
        double a = 2;
        double b = 3;
        double truth = simple_cmp(a, b);
        a = __enzyme_truncate_mem_value(a, FROM, TO);
        b = __enzyme_truncate_mem_value(b, FROM, TO);
        double trunc = __enzyme_expand_mem_value(__enzyme_truncate_mem_func(simple_cmp, FROM, TO)(a, b), FROM, TO);
        APPROX_EQ(trunc, truth, 1e-5);
    }
    {
        double a = 2;
        double b = 3;
        double truth = simple_add(a, b);
        a = __enzyme_truncate_mem_value(a, FROM, TO);
        b = __enzyme_truncate_mem_value(b, FROM, TO);
        double trunc = __enzyme_expand_mem_value(__enzyme_truncate_mem_func(simple_add, FROM, TO)(a, b), FROM, TO);
        APPROX_EQ(trunc, truth, 1e-5);
    }
    {
        double a = 2;
        double b = 3;
        double truth = intrinsics(a, b);
        a = __enzyme_truncate_mem_value(a, FROM, TO);
        b = __enzyme_truncate_mem_value(b, FROM, TO);
        double trunc = __enzyme_expand_mem_value(__enzyme_truncate_mem_func(intrinsics, FROM, TO)(a, b), FROM, TO);
        APPROX_EQ(trunc, truth, 1e-5);
    }
    {
        double a = 2;
        double b = 3;
        double truth = constt(a, b);
        a = __enzyme_truncate_mem_value(a, FROM, TO);
        b = __enzyme_truncate_mem_value(b, FROM, TO);
        double trunc = __enzyme_expand_mem_value(__enzyme_truncate_mem_func(constt, FROM, TO)(a, b), FROM, TO);
        APPROX_EQ(trunc, truth, 1e-5);
    }
    {
        double a = 2;
        double b = 3;
        double truth = phinode(a, b, 10);
        a = __enzyme_truncate_mem_value(a, FROM, TO);
        b = __enzyme_truncate_mem_value(b, FROM, TO);
        double trunc = __enzyme_expand_mem_value(__enzyme_truncate_mem_func(phinode, FROM, TO)(a, b, 10), FROM, TO);
        APPROX_EQ(trunc, truth, 20.0);
    }
    {
        double truth = 0;
        const_store(&truth);
        double a = 0;
        __enzyme_truncate_mem_func(const_store, FROM, TO)(&a);
        a = __enzyme_expand_mem_value(a, FROM, TO);
        APPROX_EQ(a, truth, 1e-5);
    }
    {
        __enzyme_truncate_mem_func(intcast, FROM, TO)(64);
    }
    #endif

    #ifdef TRUNC_OP
    {
        double A[N];
        double B[N];
        double C[N];
        double D[N];


        for (int i = 0; i < N; i++) {
            A[i] = 1 + i % 5;
            B[i] = 1 + i % 3;
        }

        compute(A, B, D, N);

        // for (int i = 0; i < N; i++) {
        //     A[i] = __enzyme_truncate_mem_value(A[i], 64, 32);
        //     B[i] = __enzyme_truncate_mem_value(B[i], 64, 32);
        // }

        __enzyme_truncate_op_func(compute, 64, 32)(A, B, C, N);

        // for (int i = 0; i < N; i++) {
        //     C[i] = __enzyme_expand_mem_value(C[i], 64, 32);
        // }

        for (int i = 0; i < N; i++) {
            APPROX_EQ(D[i], C[i], 1e-5);
        }
    }
    #endif

}
