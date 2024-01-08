// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-auto-sparsity=1  | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-auto-sparsity=1 | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-auto-sparsity=1 -S | %lli - ; fi

// everything should be always inline

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <random>


#include<math.h>

struct triple {
    size_t row;
    size_t col;
    double val;
    triple(triple&&) = default;
    triple(size_t row, size_t col, double val) : row(row), col(col), val(val) {}
};


size_t N;

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

extern void __enzyme_autodiff(void *, ...);

extern void __enzyme_fwddiff(void *, ...);

extern double* __enzyme_todense(void *, ...) noexcept;


__attribute__((always_inline))
static double f(size_t N, double* pos) {
    double e = 0.;
    for (size_t i = 0; i < N; i += 2) {
        double vx = pos[i];
        double vy = pos[i + 1];

        double wx = pos[i + 2];
        double wy = pos[i + 3];
        e += (wx - vx) * (wx - vx) + (wy - vy) * (wy - vy);
    }
    return e;
}


__attribute__((always_inline))
static void grad_f(size_t N, double* input, double* dinput) {
    __enzyme_autodiff((void*)f, enzyme_const, N, enzyme_dup, input, dinput);
}

__attribute__((always_inline))
void ident_store(double , int64_t idx, size_t i) {
    assert(0 && "should never load");
}

__attribute__((always_inline))
double ident_load(int64_t idx, size_t i, size_t N) {
    idx /= sizeof(double);
    return (double)(idx == i);// ? 1.0 : 0.0;
}

__attribute__((enzyme_sparse_accumulate))
void inner_store(int64_t row, int64_t col, double val, std::vector<triple> &triplets) {
    printf("row=%d col=%d val=%f\n", row, col % N, val);
    // assert(abs(val) > 0.00001);
    triplets.emplace_back(row % N, col % N, val);
}

__attribute__((always_inline))
void sparse_store(double val, int64_t idx, size_t i, size_t N, std::vector<triple> &triplets) {
    if (val == 0.0) return;
    idx /= sizeof(double);
    inner_store(i, idx, val, triplets);
}

__attribute__((always_inline))
double sparse_load(int64_t idx, size_t i, size_t N, std::vector<triple> &triplets) {
    return 0.0;
}

__attribute__((always_inline))
void never_store(double val, int64_t idx, double* input, size_t N) {
    assert(0 && "this is a read only input, why are you storing here...");
}

__attribute__((always_inline))
double mod_load(int64_t idx, double* input, size_t N) {
    idx /= sizeof(double);
    return input[idx % N];
}

__attribute__((noinline))
std::vector<triple> hess_f(size_t N, double* input) {
    std::vector<triple> triplets;
    // input = __enzyme_todense((void*)mod_load, (void*)never_store, input, N);
    __builtin_assume(N > 0);
    for (size_t i=0; i<N; i++) {
        __builtin_assume(i < 100000000);
        double* d_input = __enzyme_todense((void*)ident_load, (void*)ident_store, i, N);
        double* d_dinput = __enzyme_todense((void*)sparse_load, (void*)sparse_store, i, N, &triplets);

       __enzyme_fwddiff((void*)grad_f, 
                            enzyme_const, N,
                            enzyme_dup, input, d_input,
                            enzyme_dupnoneed, (double*)0x1, d_dinput);

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
// int argc, char** argv
int __attribute__((always_inline)) main() {
    std::mt19937 generator(0); // Seed the random number generator
    std::uniform_real_distribution<double> normal(0, 0.05);


    if (argc != 2) {
        printf("Usage: %s <size>\n", argv[0]);
        return 1;
    }

    // size_t N = atoi(argv[1]);
    size_t N = 16;

    double x[2 * N + 2];
    for (int i = 0; i < N; ++i) {
        double angle = 2 * M_PI * i / N;
        x[2 * i] = cos(angle) + normal(generator);
        x[2 * i + 1] = sin(angle) + normal(generator);
    }
    x[2 * N] = x[0];
    x[2 * N + 1] = x[1];
    auto res = hess_f(N, &x[0]);

    printf("%ld\n", res.size());
  
    for (auto & tup : res)
        printf("%ld, %ld = %f\n", tup.row, tup.col, tup.val);
  
    return 0;
}

