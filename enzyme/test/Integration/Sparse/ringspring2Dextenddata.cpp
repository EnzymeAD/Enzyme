// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions-ffast-math  -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions-ffast-math  -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions-ffast-math  -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

// everything should be always inline

#include <stdio.h>
#include <assert.h>
#include <vector>


#include<math.h>

#include "matrix.h"

template<typename T>
__attribute__((always_inline))
static double f(size_t N, double* pos) {
    double e = 0.;
    for (size_t i = 0; i < N; i ++) {
        __builtin_assume(i < 1000000000);
        double vx = pos[2 * i];
        double vy = pos[2 * i + 1];

        double wx = pos[2 * i + 2];
        double wy = pos[2 * i + 3];
        e += (wx - vx) * (wx - vx) + (wy - vy) * (wy - vy);
    }
    return e;
}

template<typename T>
__attribute__((always_inline))
static void grad_f(size_t N, T* input, T* dinput) {
    __enzyme_autodiff<void>((void*)f<T>, enzyme_const, N, enzyme_dup, input, dinput);
}

template<typename T>
__attribute__((always_inline))
static void never_store(T val, int64_t idx, T* input, size_t N) {
    assert(0 && "this is a read only input, why are you storing here...");
}

__attribute__((always_inline))
static double mod_load(int64_t idx, double* input, size_t N) {
    idx /= sizeof(double);
    return input[idx % N];
}

template<typename T>
__attribute__((noinline))
std::vector<Triple<T>> hess_f(size_t N, T* input) {
    std::vector<Triple<T>> triplets;
    // input = __enzyme_todense((void*)mod_load, (void*)never_store, input, N);
    __builtin_assume(N > 0);
    for (size_t i=0; i<N; i++) {
        __builtin_assume(i < 100000000);
        T* d_input = __enzyme_todense<T*>((void*)ident_load<T>, (void*)ident_store<T>, i);
        T* d_dinput = __enzyme_todense<T*>((void*)sparse_load<T>, (void*)sparse_store<T>, i, &triplets);

       __enzyme_fwddiff<void>((void*)grad_f<T>, 
                            enzyme_const, N,
                            enzyme_dup, input, d_input,
                            enzyme_dupnoneed, (T*)0x1, d_dinput);

    }
    return triplets;
}

int main(int argc, char** argv) {
    size_t N = 30;

    if (argc >= 2) {
         N = atoi(argv[1]);
    }

    double *x = (double*)malloc(sizeof(double) * (2 * N + 2));
    for (int i = 0; i < N; ++i) {
        double angle = 2 * M_PI * i / N;
        x[2 * i] = cos(angle) ;//+ normal(generator);
        x[2 * i + 1] = sin(angle) ;//+ normal(generator);
    }
    x[2 * N] = x[0];
    x[2 * N + 1] = x[1];


  struct timeval start, end;
  gettimeofday(&start, NULL);
  
  auto res = hess_f(N, x);

  gettimeofday(&end, NULL);
    
  printf("Number of elements %ld\n", res.size());
  
  printf("Runtime %0.6f\n", tdiff(&start, &end));

  if (N <= 30) {
  for (auto & tup : res)
      printf("%ld, %ld = %f\n", tup.row, tup.col, tup.val);
  }

  return 0;
}
