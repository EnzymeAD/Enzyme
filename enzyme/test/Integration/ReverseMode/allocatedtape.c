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

#include "test_utils.h"

extern int enzyme_dup, enzyme_allocated, enzyme_tape;
void __enzyme_autodiff(void*, ...);
void* __enzyme_augmentfwd(void*, ...);
void __enzyme_reverse(void*, ...);

#define TAPE_SIZE 100   // v0.69 wants 24 bytes, so 100 should be more than enough

// jac(f)(x) is [1 2*x1; 0 -1] as a function of the original inputs
void f(double* x) {
    x[0] += x[1]*x[1];
    x[1] = -x[1];
}

void f_aug(double* x, double* x_b, void* tape) {
    __enzyme_augmentfwd(f, enzyme_dup, x, x_b,
                enzyme_allocated, TAPE_SIZE, enzyme_tape, tape);
}

void f_rev(double* x, double* x_b, void* tape) {
    __enzyme_reverse(f, enzyme_dup, x, x_b,
                enzyme_allocated, TAPE_SIZE, enzyme_tape, tape);
}

int main() {
    double x0 = 1.2, x1 = 3.4;
    double x0_b = 5.6, x1_b = 7.8;
    double x[2] = {x0, x1};
    double x_b[2] = {x0_b, x1_b};
    char tape[TAPE_SIZE];

    f_aug(x, x_b, tape);
    f_rev(x, x_b, tape);

    APPROX_EQ(x[0],   x0 + x1*x1,              1e-10);
    APPROX_EQ(x[1],   -x1,                     1e-10);
    APPROX_EQ(x_b[0], x0_b*(1)    + x1_b*(0),  1e-10);
    APPROX_EQ(x_b[1], x0_b*(2*x1) + x1_b*(-1), 1e-10);
}
