// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

// everything should be always inline


#include <stdio.h>
#include <assert.h>
#include <vector>


#include<math.h>

#include "matrix.h"

template<typename T>
__attribute__((always_inline))
static T f(size_t N, T* input) {
    double out = 0;
    // __builtin_assume(!((N-1) == 0));
    for (size_t i=0; i<N; i++) {
        //double sub = input[i] - input[i+1]; 
        // out += sub * sub;
        T sub = (input[i+1] - input[i]) * (input[i+1] - input[i]);
        out += (sqrt(sub) - 1)*(sqrt(sub) - 1);
    }
    return out;
}

template<typename T>
__attribute__((always_inline))
static void grad_f(size_t N, T* input, T* dinput) {
    __enzyme_autodiff<void>((void*)f<T>, enzyme_const, N, enzyme_dup, input, dinput);
}

template<typename T>
__attribute__((always_inline))
double ringident_load(int64_t idx, size_t i, size_t N) {
    idx /= sizeof(double);
    // return (double)( ( (idx == N) ? 0 : idx) == i);
    return (double)((idx != N && idx == i) || (idx == N && 0 == i));
    // return (double)( idx % N == i);
}
template<typename T>
__attribute__((always_inline))
void never_store(T val, int64_t idx, T* input, size_t N) {
    assert(0 && "this is a read only input, why are you storing here...");
}

template<typename T>
__attribute__((always_inline))
double mod_load(int64_t idx, double* input, size_t N) {
    idx /= sizeof(double);
    return input[idx % N];
}

template<typename T>
__attribute__((noinline))
std::vector<Triple<T>> hess_f(size_t N, T* input) {
    std::vector<Triple<T>> triplets;
    input = __enzyme_todense<T*>((void*)mod_load<T>, (void*)never_store<T>, input, N);
    __builtin_assume(N > 0);
    __builtin_assume(N != 1);
    for (size_t i=0; i<N; i++) {
        __builtin_assume(i < 100000000);
        T* d_input = __enzyme_todense<T*>((void*)ringident_load<T>, (void*)never_store<T>, i, N);
        T* d_dinput = __enzyme_todense<T*>((void*)sparse_load<T>, (void*)sparse_store<T>, i, &triplets);

       __enzyme_fwddiff<void>((void*)grad_f<T>, 
                            enzyme_const, N,
                            enzyme_dup, input, d_input,
                            enzyme_dupnoneed, (T*)0x1, d_dinput);

    }
    return triplets;
}


int main(int argc, char** argv) {
    size_t N = 8;

    if (argc >= 2) {
         N = atoi(argv[1]);
    }

    double *x = (double*)malloc(sizeof(double) * N);
    for (int i=0; i<N; i++) x[i] = (i + 1) * (i + 1);


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
