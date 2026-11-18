// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -lt 18 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -lt 18 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -lt 18 ]; then %clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -ffast-math -mllvm -enable-load-pre=0  -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

// /usr/local/Cellar/llvm@15/15.0.7/bin/clang++ -fno-exceptions  -ffast-math -mllvm -enable-load-pre=0 -std=c++11 -O3 /Users/jessemichel/research/sparse_project/benchmark_enzyme/Enzyme/enzyme/test/Integration/Sparse/ringspring3Drestlengthone.cpp -fpass-plugin=/Users/jessemichel/research/sparse_project/benchmark_enzyme/Enzyme/enzyme/build15D/Enzyme/ClangEnzyme-15.dylib -Xclang -load -Xclang /Users/jessemichel/research/sparse_project/benchmark_enzyme/Enzyme/enzyme/build15D/Enzyme/ClangEnzyme-15.dylib -mllvm -enzyme-auto-sparsity=1 -o ringspring3Drestlengthone.out


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
    for (size_t j = 0; j < N; j ++) {
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

template<typename T>
__attribute__((always_inline))
static T mod_load(int64_t idx, T* input, size_t N) {
    idx /= sizeof(T);
    return input[idx % N];
}

template<typename T>
__attribute__((noinline))
std::vector<Triple<T>> hess_f(size_t N, T* input) {
    std::vector<Triple<T>> triplets;
    input = __enzyme_post_sparse_todense<T*>((void*)mod_load<T>, (void*)never_store<T>, input, N);
    __builtin_assume(N > 0);
    for (size_t i=0; i<N; i++) {
        __builtin_assume(i < 100000000);
        T* d_input = __enzyme_todense<T*>((void*)ident_load<T>, (void*)ident_store<T>, i);
        T* d_dinput = __enzyme_todense<T*>((void*)sparse_load_modn<T>, (void*)sparse_store_modn<T>, i, N, &triplets);

       __enzyme_fwddiff<void>((void*)grad_f<T>, 
                            enzyme_const, N,
                            enzyme_dup, input, d_input,
                            enzyme_dupnoneed, (T*)0x1, d_dinput);

    }
    return triplets;
}

/*
__attribute__((noinline))
std::vector<triple> hess_f2(size_t N, double* input) {
    std::vector<triple> triplets;
    input = 
    ((void*)mod_load, (void*)never_store, input, N);
    hess_f(N, input);
}
*/

template<typename T>
__attribute__((noinline))
std::vector<Triple<T>> handrolled_hess_f(size_t n, T* data) {
    std::vector<Triple<T>> triplets;
    int k = 3 * n;
    for (int i = 0; i < n; ++i) {
        int row = 3 * i;
        double x = data[row % k];
        double y = data[(row + 1) % k];
        double z = data[(row + 2) % k];
        double a = data[(row + 3) % k];
        double b = data[(row + 4) % k];
        double c = data[(row + 5) % k];

        double f = pow(x - a, 2) + pow(y - b, 2) + pow(z - c, 2);
        // double f = (x - a) * (x - a) + (y - b) * (y - b) + (z - c) * (z - c);
        double g = 2 * (1 - 1 / sqrt(f));

        // Diagonal terms
        #pragma clang loop unroll(full)
        for (int j = 0; j < 6; ++j) {
            T val = g;
#ifdef BENCHMARK
            if (val > -1e-10 && val < 1e-10) continue;
#endif
            triplets.emplace_back((row + j) % k, (row + j) % k, val);
        }

        #pragma clang loop unroll(full)
        for (int j = 0; j < 6; ++j) {
            T val = -g;
#ifdef BENCHMARK
            if (val > -1e-10 && val < 1e-10) continue;
#endif
            triplets.emplace_back((row + ((j + 3) % 6)) % k, (row + j) % k, val);
        }

        // Cross terms
        std::vector<double> half_grad_f = {x - a, y - b, z - c, a - x, b - y, c - z};
        // double half_grad_f[] = {x - a, y - b, z - c, a - x, b - y, c - z};
        #pragma clang loop unroll(full)
        for (int j = 0; j < 6; ++j) {
            #pragma clang loop unroll(full)
            for (int l = 0; l < 6; ++l) {
                T val = 2 * half_grad_f[j] * half_grad_f[l] * pow(f, -1.5);
                // T val = 2 * half_grad_f[j] * half_grad_f[l] / ( f * sqrt(f) );
#ifdef BENCHMARK
                if (val > -1e-10 && val < 1e-10) continue;
#endif
                triplets.emplace_back((row + j) % k, (row + l) % k, val);
            }
        }
    }
    return triplets;
}

// int argc, char** argv
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

    for (int i=0; i<10; i++) 
    {
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

  gettimeofday(&start, NULL);
  auto hand_res = handrolled_hess_f(N, x);
  gettimeofday(&end, NULL);
  
  printf("Handrolled Number of elements %ld\n", hand_res.size());
  printf(" Runtime %0.6f\n", tdiff(&start, &end));
    }

  return 0;
}

