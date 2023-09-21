// RUN: %clang -std=c11 -g -O0 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify
// RUN: %clang -std=c11 -g -O1 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify
// RUN: %clang -std=c11 -g -O2 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify
// RUN: %clang -std=c11 -g -O3 %s -S -emit-llvm -o -  %loadClangEnzyme -Xclang -verify
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O0 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O1 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O2 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -g -O3 %s -S -emit-llvm -o -  %newLoadClangEnzyme -Xclang -verify; fi

extern double __enzyme_fwddiff(void*, double, double);

double unknown(double in);

double g(double in) {
    return unknown(in); // expected-error {{Enzyme: No forward mode derivative found for unknown}}
}

double square(double x, double dx) {
    return __enzyme_fwddiff((void*)g, x, dx);
}
