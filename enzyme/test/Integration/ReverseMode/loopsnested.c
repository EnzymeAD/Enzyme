// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli - 
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

#include "../test_utils.h"

__attribute__((always_inline))
inline void doA(double x[2])
{
    double t;
    t = 4*__builtin_cos(x[0] + x[1]);
    x[0] = 3*__builtin_sin(2*x[0] + x[1]);
    x[1] = t;
}

__attribute__((always_inline))
inline void doB(double x[2])
{
    double t;
    t = 4*__builtin_sin(2*x[0] + x[1]);
    x[0] = 4*__builtin_cos(x[0] + 2*x[1]);
    x[1] = t;
}

// for input x = {0.9, 0.7}
void fcheck(double x[2])
{
    doA(x);
    doA(x);
    doB(x);
    doB(x);
    doA(x);
    doB(x);
}

void f(double x[2])
{
A:  doA(x);
    if (x[0]*x[0] < x[1])
        return;
    else if (x[0] > 0)
        goto A;
    else
        goto B;

B: doB(x);
   if (x[0]*x[0] < x[1])
      return;
   else if (x[0] > 0)
      goto A;
   else
      goto B;
}

extern void* __enzyme_augmentfwd(void*, double*, double*);
extern void __enzyme_reverse(void*, double*, double*, void*);
extern void __enzyme_autodiff(void*, double*, double*);

int main()
{
    double x[2], x_b[2], y[2], y_b[2];
    x[0] = y[0] = 0.9;
    x[1] = y[1] = 0.7;
    x_b[0] = y_b[0] = 1;
    x_b[1] = y_b[1] = 2;

    void* tape = __enzyme_augmentfwd((void*)f, x, x_b);
    __enzyme_reverse((void*)f, x, x_b, tape);
    __enzyme_autodiff((void*)fcheck, y, y_b);
    APPROX_EQ(x[0], y[0], 1e-10);
    APPROX_EQ(x[1], y[1], 1e-10);
    APPROX_EQ(x_b[0], y_b[0], 1e-10);
    APPROX_EQ(x_b[1], y_b[1], 1e-10);
}
