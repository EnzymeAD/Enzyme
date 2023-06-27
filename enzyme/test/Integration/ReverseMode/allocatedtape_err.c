// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O0 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O1 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O3 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O0 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O1 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O3 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi

#include <math.h>

extern int enzyme_allocated, enzyme_tape;
void __enzyme_reverse(void*, ...);

void f(double* x) {
    x[0] = sin(x[0]);
}

void f_rev(double* x, double* x_b, void* tape) {
    __enzyme_reverse(f, x, x_b, enzyme_allocated, 1, enzyme_tape, tape); // expected-error {{Enzyme: need 8 bytes have 1 bytes}}
}

