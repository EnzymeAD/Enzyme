// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include "test_utils.h"

// Fast inverse sqrt
// Code taken from https://en.wikipedia.org/wiki/Fast_inverse_square_root
float Q_rsqrt( float number )
{
  int32_t i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y  = number;
  i  = * ( int32_t * ) &y;                    // evil floating point bit level hacking
  i  = 0x5f3759df - ( i >> 1 );               // what the [...]?
  y  = * ( float * ) &i;
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
  return y;
}


double invmag(double* __restrict__ A, int n) {
  double sumsq = 0;
  for (int i=0; i<n; i++) {
    sumsq += A[i] * A[i];
  }
  return Q_rsqrt(sumsq);
}

//// (1) An interface that works only because the usage of Q_rsqrt inside invmag is simple enough.

// Returns { optional tape, original return (if pointer), shadow return (if pointer) }
void aug_rsqrt(float x) {
  // Nothing need to be done in augmented forward pass
}

// Arguments: all pointers duplicated, gradient of the return, tape (if provided)
float rev_rsqrt(float x, float grad_out) {
  // derivative of x^(-1/2) = -1/2 x^(-3/2)
  return -grad_out * Q_rsqrt(x) / (2 * x);
}

void* __enzyme_register_gradient_rsqrt[3] = { (void*)Q_rsqrt, (void*)aug_rsqrt, (void*)rev_rsqrt };

//// (2) That the above aug_rsqrt works is really a fluke. This is a correct interface in general.
double rsqrt2(double x) {return Q_rsqrt(x);}
double aug_rsqrt2(double x) {return Q_rsqrt(x);}
double rev_rsqrt2(double x, double y_b) {return -y_b * Q_rsqrt(x) / (2 * x);}
void* __enzyme_register_gradient_rsqrt2[3] = { (void*)rsqrt2, (void*)aug_rsqrt2, (void*)rev_rsqrt2};

double invmag2(double* __restrict__ A, int n) {
  double sumsq = 0;
  for (int i=0; i<n; i++)
    sumsq += A[i] * A[i];
  return rsqrt2(sumsq);
}

//// (3) This is another possible interface for more complicated functions.
void rsqrt3(double* y, double* x) {
    *y = Q_rsqrt(*x);
}
void* aug_rsqrt3(double* y, double* y_b, double* x, double* x_b) {
    double* tape = malloc(sizeof(double));
    *tape = *x;
    *y = Q_rsqrt(*x);
    return tape;
}
void rev_rsqrt3(double* y, double* y_b, double* x, double* x_b, void* tape) {
    double x0 = *(double*)tape; // x points to junk; original input *x must be obtained from the tape
    *x_b -= (*y_b) * Q_rsqrt(x0) / (2 * x0);
    *y_b = 0;   // Since y is used purely as an output, changes in y do not affect the calculation
    free(tape);
}
void* __enzyme_register_gradient_rsqrt3[3] = { (void*)rsqrt3, (void*)aug_rsqrt3, (void*)rev_rsqrt3};

double invmag3(double* __restrict__ A, int n) {
  double sumsq = 0;
  for (int i=0; i<n; i++)
    sumsq += A[i] * A[i];
  double res;
  rsqrt3(&res, &sumsq);
  return res;
}


void __enzyme_autodiff(void*, ...);

int main(int argc, char *argv[]) {
  int n = 3;

  double *A  = (double*)malloc(sizeof(double) * n);
  double *A2 = (double*)malloc(sizeof(double) * n);
  double *A3 = (double*)malloc(sizeof(double) * n);
  for(int i=0; i<n; i++)
    A[i] = A2[i] = A3[i] = i+1;

  double *grad_A  = (double*)malloc(sizeof(double) * n);
  double *grad_A2 = (double*)malloc(sizeof(double) * n);
  double *grad_A3 = (double*)malloc(sizeof(double) * n);
  for(int i=0; i<n; i++)
    grad_A[i] = grad_A2[i] = grad_A3[i] = 0;

  __enzyme_autodiff((void*)invmag , A , grad_A , n);
  __enzyme_autodiff((void*)invmag2, A2, grad_A2, n);
  __enzyme_autodiff((void*)invmag3, A3, grad_A3, n);

  for(int i=0; i<n; i++) {
    printf("A [%d]=%f dA [%d]=%f\n", i, A [i], i, grad_A [i]);
    printf("A2[%d]=%f dA2[%d]=%f\n", i, A2[i], i, grad_A2[i]);
    printf("A3[%d]=%f dA3[%d]=%f\n", i, A3[i], i, grad_A3[i]);
  }
  
  double im  = invmag(A,  n);
  im  = im *im *im;
  for(int i=0; i<n; i++) {
    APPROX_EQ(grad_A[i] , -A[i]*im, 1e-3);
    APPROX_EQ(grad_A2[i], -A[i]*im, 1e-3);
    APPROX_EQ(grad_A3[i], -A[i]*im, 1e-3);
  }
}
