// RUN: if [ %llvmver -ge 12 ]; then %clang -std=c11 -c %s %newLoadClangEnzyme -o -; fi

#include <stdlib.h>

extern int enzyme_const, enzyme_out;
double __enzyme_autodiff(void*, ...);

union DoublePtr {
    long addrInt;
    double *ptr;
};

__attribute__((enzyme_notypeanalysis))
static union DoublePtr dPtr;

double divide(double x, double y) {
    double tmp = x / y;

    // Access pointer part of dPtr
    *dPtr.ptr = x;

    // Access long int part of dPtr. Without enzyme_notypeanalysis,
    // this would cause a type analysis conflict.
    return tmp + dPtr.addrInt;
}

double d_divide(double x) {
    double y = 3.0;
    return __enzyme_autodiff((void*)divide, enzyme_out, x, enzyme_const, y);
}