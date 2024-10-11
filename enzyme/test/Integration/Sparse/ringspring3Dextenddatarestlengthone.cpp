// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -lt 18 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -lt 18 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -lt 18 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <random>

#include<math.h>

#include "matrix.h"

template<typename T>
__attribute__((always_inline))
static double f(size_t N, T* __restrict__ pos) {
    double e = 0.;
    __builtin_assume(N != 0);
    for (size_t j = 0; j < N/3; j ++) {
        size_t i = 3 * j;
        T vx = pos[i];
        T vy = pos[i + 1];
        T vz = pos[i + 2];
        
        T wx = pos[i + 3];
        T wy = pos[i + 4];
        T wz = pos[i + 5];
        T distance = (wx - vx) * (wx - vx) + (wy - vy) * (wy - vy) + (wz - vz) * (wz - vz);
        T rest_len_one_dist = (sqrt(distance) - 1) * (sqrt(distance) - 1);
        e += rest_len_one_dist;
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

    double *x = (double*)malloc(sizeof(double) * (3 * N + 3));
    for (int i = 0; i < N; ++i) {
        double angle = 2 * M_PI * i / N;
        x[3 * i] = cos(angle) ;//+ normal(generator);
        x[3 * i + 1] = sin(angle) ;//+ normal(generator);
        x[3 * i + 2] = 0;//normal(generator);
    }
    x[3 * N] = x[0];
    x[3 * N + 1] = x[1];
    x[3 * N + 2] = x[2];


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

