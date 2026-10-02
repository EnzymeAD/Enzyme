// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

// The shadow of a global that points to another global points to the
// shadow of that global, in each lane: here a descriptor, as Fortran keeps
// for an array, whose shadow holds the same size and the address of the
// shadow of the data.

#include "../test_utils.h"

typedef struct {
  double d1, d2;
} Tangents;

extern void *__enzyme_context(int);
extern void *__enzyme_shadow(void *, void *, int);
extern double __enzyme_fwddiff(double (*)(double), ...);
extern Tangents __enzyme_fwddiff2(double (*)(double), ...)
    __asm__("__enzyme_fwddiff");
extern int enzyme_context;

double data[3] = {1.0, 2.0, 3.0};

struct {
  double *base;
  long size;
} desc = {data, 3};

__attribute__((noinline)) double sum(double x) {
  double s = 0;
  for (long i = 0; i < desc.size; i++)
    s += desc.base[i] * x;
  return s;
}

int main() {
  void *ctx = __enzyme_context(1);
  // The shadow descriptor exists before any derivative needs it.
  double **dbase = (double **)__enzyme_shadow(ctx, &desc.base, 0);
  long *dsize = (long *)__enzyme_shadow(ctx, &desc.size, 0);
  double *ddata = (double *)__enzyme_shadow(ctx, &data, 0);
  APPROX_EQ((double)*dsize, 3.0, 1e-10);
  APPROX_EQ((double)(*dbase == ddata), 1.0, 1e-10);

  ddata[0] = 1.0;
  ddata[2] = 1.0;
  double t = __enzyme_fwddiff(sum, enzyme_context, ctx, 2.0, 0.0);
  APPROX_EQ(t, (1.0 + 1.0) * 2.0, 1e-10);

  void *ctx2 = __enzyme_context(2);
  double **dbase1 = (double **)__enzyme_shadow(ctx2, &desc.base, 1);
  double *ddata1 = (double *)__enzyme_shadow(ctx2, &data, 1);
  APPROX_EQ((double)(*dbase1 == ddata1), 1.0, 1e-10);
  double *ddata0 = (double *)__enzyme_shadow(ctx2, &data, 0);
  ddata0[1] = 1.0;
  ddata1[2] = 1.0;
  Tangents t2 = __enzyme_fwddiff2(sum, enzyme_context, ctx2, 2.0, 0.0, 0.0);
  APPROX_EQ(t2.d1, 2.0, 1e-10);
  APPROX_EQ(t2.d2, 2.0, 1e-10);
  return 0;
}
