// RUN: if [ %llvmver -ge 12 ]; then ls && %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi

#include "../test_utils.h"

double pow(double, double);

__attribute__((noinline))
unsigned long long factorial(unsigned long long x) {
    if (x == 0) return 1;
    return x * factorial(x-1);
}

double my_sin(double x) {
    double result = 0;
    unsigned long long N = 12;
    for(unsigned long long i=0; i<=N; i++) {
        if (i % 2 == 0) continue;
        result += pow(x, i) / factorial(i) * (i % 4 == 1 ? 1 : -1);
    }
    return result;

}

unsigned long long __enzyme_iter(unsigned long long, unsigned long long);

double __enzyme_autodiff(void*, double);

double my_sin2(double x) {
    double result = 0;
    unsigned long long N = __enzyme_iter(12, 1);
    for(unsigned long long i=0; i<=N; i++) {
        if (i % 2 == 0) continue;
        result += pow(x, i) / factorial(i) * (i % 4 == 1 ? 1 : -1);
    }
    return result;
}

double d_mysin2(double x) {
    return __enzyme_autodiff(my_sin2, x);
}
double dd_mysin2(double x) {
    return __enzyme_autodiff(d_mysin2, x);
}
double ddd_mysin2(double x) {
    return __enzyme_autodiff(dd_mysin2, x);
}

double dddd_mysin2(double x) {
    return __enzyme_autodiff(ddd_mysin2, x);
}


int main() {
    double x = 1.23;
    printf("my_sin(x=%f)=%e\n", x, my_sin(x));
    printf("my_sin2(x=%f)=%e\n", x, my_sin2(x));
    APPROX_EQ(my_sin2(x), my_sin(x), 10e-10);
    printf("dd_my_sin2(x=%f)=%e\n", x, dd_mysin2(x));
    APPROX_EQ(dd_mysin2(x), -my_sin(x), 10e-10);
    printf("dddd_my_sin2(x=%f)=%e\n", x, dddd_mysin2(x));
    APPROX_EQ(dddd_mysin2(x), my_sin(x), 10e-10);
}
