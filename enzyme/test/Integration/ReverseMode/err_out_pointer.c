// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O0 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O1 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 10 ]; then %clang -std=c11 -g -O3 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O0 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O1 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O3 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi

// A pointer argument marked enzyme_out, instead of enzyme_dup with a shadow.
// This is the mistake behind https://github.com/EnzymeAD/Enzyme/issues/2623
// (a C++ reference parameter), which used to fail with an unrelated error from
// inside the derivative.

int enzyme_out;
extern double __enzyme_autodiff(void*, ...);

double square(double* x) {
  return x[0] * x[0];
}

double dsquare(double* x) {
  return __enzyme_autodiff((void*)square, enzyme_out, x); // expected-error {{Enzyme: enzyme_out applied to argument 0 of square, which is a pointer (or reference) rather than a value passed by value.}}
}
