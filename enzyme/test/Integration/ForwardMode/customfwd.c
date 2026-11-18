// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi

#include "../test_utils.h"

int enzyme_dup;
int enzyme_const;
double __enzyme_fwddiff(void*, ...);

static void square_(const double *src, double *dest) { *dest = *src * *src; }

int derivative = 0;
void derivative_square_(const double* src, const double *d_src, const double* dest, double* d_dest) {
    derivative++;
    // intentionally incorrect for debugging
    *d_dest = 100; 
}

void* __enzyme_register_derivative_square[] = {
    (void*)square_,
    (void*)derivative_square_,
};


double square(double x) {
    double y;
    square_(&x, &y);
    return y;
}

double dsquare(double x) {
    return __enzyme_fwddiff((void*)square, x, 1.0);
}

__attribute__((noinline)) int my_func(int x_size, const double *x, int *fx_size,
                                      double *fx) {
  derivative -= 1000;
  *fx_size = 1;
  fx[0] = 0.0;
  for (int i = 0; i < x_size; i++)
    fx[0] += x[i];
  return 0;
}

int derivative_my_func(int x_size, int d_x_size, const double *x,
                       const double *dx, int *fx_size, int *d_fx_size,
                       double *fx, double *dfx) {
  derivative++;
  *fx_size = 1;
  fx[0] = 0.0;
  dfx[0] = 0.0;
  for (int i = 0; i < x_size; i++) {
    fx[0] += x[i];
    dfx[0] += dx[i];
  }
  return 0;
}

void *__enzyme_register_derivative_my_func[] = {
    (void *)my_func,
    (void *)derivative_my_func,
};

int call_my_func(int x_size, const double *x, int *fx_size, double *fx) {
  int info = my_func(x_size, x, fx_size, fx);
  if (info != 0)
    return info;
  fx[0] *= 2.0;
  return 0;
}

int main() {
    double res = dsquare(3.0);
    printf("res=%f derivative=%d\n", res, derivative);
    APPROX_EQ(res, 100.0, 1e-10);
    APPROX_EQ(derivative, 1.0, 1e-10);

    double x[2] = {1.0, 2.0};
    double dx[2] = {3.0, 5.0};
    double fx[1] = {0.0};
    double dfx[1] = {0.0};
    int fx_size = 0;

    __enzyme_fwddiff((void *)my_func, enzyme_const, 2, enzyme_dup, x, dx,
                     enzyme_const, &fx_size, enzyme_dup, fx, dfx);
    APPROX_EQ(derivative, 2.0, 1e-10);
    APPROX_EQ(fx[0], 3.0, 1e-10);
    APPROX_EQ(dfx[0], 8.0, 1e-10);

    fx[0] = 0.0;
    dfx[0] = 0.0;
    __enzyme_fwddiff((void *)call_my_func, enzyme_const, 2, enzyme_dup, x, dx,
                     enzyme_const, &fx_size, enzyme_dup, fx, dfx);
    APPROX_EQ(derivative, 3.0, 1e-10);
    APPROX_EQ(fx[0], 6.0, 1e-10);
    APPROX_EQ(dfx[0], 16.0, 1e-10);
}
