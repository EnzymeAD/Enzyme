// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=0 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=0 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=0 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=0 | %lli - ; fi

#include "../test_utils.h"
#include "../blas_inline.h"

#include <stdio.h>

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

void __enzyme_autodiff(void*, ...);

const size_t n = 10;

static char N = 'N';
static int ten = 10;
static double one = 1.0;
static double zero = 0.0;
double simulate(double* A) {
  double *out = (double*)malloc(sizeof(double)*n*n);
  dgemm_(&N, &N, &ten, &ten, &ten, &one, A, &ten, A, &ten, &zero, &out[0], &ten);
  return out[0];//P1(0, 0);
}

int main(int argc, char **argv) {

    double A[n * n];
    double Adup[n * n];
    double Adup_fd[n * n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[n*i + j] = j == i ? 0.3 : 0.1;
            Adup[n*i + j] = 0.0;
            Adup_fd[n*i + j] = 0.0;
        }
    }
  
    double delta = 0.001;
    delta = delta * delta;

    double fx = simulate(A);
    printf("f(A) = %f\n", fx);
   
    __enzyme_autodiff((void *)simulate, enzyme_dup, &A[0], &Adup[0]);
    
    for (int i = 0; i < n*n; i++) {
        printf("dA[%d]=%f\n", i, Adup[i]);
    }
    for (int i = 0; i < n*n; i++) {
        A[i] += delta / 2;
        double fx2 = simulate(A);
        A[i] -= delta;
        double fx3 = simulate(A);
        A[i] += delta/2;
        
        Adup_fd[i] = (fx2 - fx3) / delta;

        printf("dA_fd[%d]=%f\n", i, Adup_fd[i]);
    
        APPROX_EQ(Adup[i], Adup_fd[i], 1e-6);
    }

    return 0;
}
