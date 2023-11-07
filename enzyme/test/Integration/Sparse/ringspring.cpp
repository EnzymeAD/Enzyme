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


#include<math.h>

struct triple {
    size_t row;
    size_t col;
    double val;
    triple(triple&&) = default;
    triple(size_t row, size_t col, double val) : row(row), col(col), val(val) {}
};


size_t N = 8;

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

extern void __enzyme_autodiff(void *, ...);

extern void __enzyme_fwddiff(void *, ...);

extern double* __enzyme_todense(void *, ...) noexcept;


/// Compute energy
double f(size_t N, double* input) {
    double out = 0;
    // __builtin_assume(!((N-1) == 0));
    for (size_t i=0; i<N; i++) {
        //double sub = input[i] - input[i+1]; 
        // out += sub * sub;
        double sub = (input[i+1] - input[i]) * (input[i+1] - input[i]);
        out += (sqrt(sub) - 1)*(sqrt(sub) - 1);
    }
    return out;
}

/// Perform dinput += gradient(f)
void grad_f(size_t N, double* input, double* dinput) {
    __enzyme_autodiff((void*)f, enzyme_const, N, enzyme_dup, input, dinput);
}


void ident_store(double , int64_t idx, size_t i) {
    assert(0 && "should never load");
}

__attribute__((always_inline))
double ident_load(int64_t idx, size_t i, size_t N) {
    idx /= sizeof(double);
    return (double)(idx % N == i % N);// ? 1.0 : 0.0;
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
    input = __enzyme_todense((void*)mod_load, (void*)never_store, input, N);
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


int main() {
  // size_t N = 8;
  double x[N];
  for (int i=0; i<N; i++) x[i] = (i + 1) * (i + 1);

  auto res = hess_f(N, &x[0]);


  printf("%ld\n", res.size());

  for (auto & tup : res)
      printf("%ld, %ld = %f\n", tup.row, tup.col, tup.val);

  return 0;
}
