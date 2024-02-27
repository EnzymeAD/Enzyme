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

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

#include <assert.h>

void __enzyme_autodiff(void*, ...);

const size_t n = 20;

#include <string.h>
struct Prod {
    double* out;
    double alpha;
};

__attribute__((noinline))
void mul(struct Prod* P, double* __restrict__ rhs) {
  double* tmp= (double*)malloc(sizeof(double)*n*n);
  memset(tmp, 0, n*n*sizeof(double));
  char N = 'N';
  int ten = n;
  double one = 1.0;
  double zero = 0.0;

  dgemm_(&N, &N, &ten, &ten, &ten, &one, rhs, &ten, rhs, &ten, &one, tmp, &ten);
  dgemm_(&N, &N, &ten, &ten, &ten, &one, tmp, &ten, rhs, &ten, &zero, P->out, &ten);
  P->alpha = 0;
  return;
}

double simulate(double* P) {
  struct Prod M;
  M.out = (double*)malloc(sizeof(double)*n*n);
  M.alpha = 1.0;
  mul(&M, P);
  return M.out[0];
  // double *out = (double*)malloc(sizeof(double)*n*n);
  // dgemm_(&N, &N, &ten, &ten, &ten, &one, P1.data(), &ten, P.data(), &ten, &zero, &out[0], &ten);
  // return P1(0, 0);
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
   
    // if (argc == 2) {
    __enzyme_autodiff((void *)simulate, enzyme_dup, &A[0], &Adup[0]);
    printf("dP(0,0) = %f, dP(0,1) = %f, dP(1,0) = %f\n", Adup[0], Adup[1], Adup[2]);
    //}

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
