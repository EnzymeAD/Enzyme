// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Seed and read the shadows of globals that Enzyme creates itself, in
// contexts of width 1 and width 2, through __enzyme_shadow.

#include "../test_utils.h"

typedef struct {
  double d1, d2;
} Tangents;

extern void *__enzyme_context(int);
extern void *__enzyme_shadow(void *, void *, int);
extern double __enzyme_fwddiff(double (*)(double), ...);
extern Tangents __enzyme_fwddiff2(double (*)(double), ...)
    __asm__("__enzyme_fwddiff");
extern double __enzyme_autodiff(double (*)(double), ...);
extern int enzyme_context;

double g = 2.0;

// Like a Fortran COMMON block: the query names a member at an offset.
struct {
  double a;
  double b;
} blk = {1.0, 3.0};

double f(double x) { return g * x * x + blk.b * x; }

int main() {
  void *ctx = __enzyme_context(1);
  double *dg = (double *)__enzyme_shadow(ctx, &g, 0);
  double *db = (double *)__enzyme_shadow(ctx, &blk.b, 0);

  // Reverse mode accumulates the adjoints of the globals.
  *dg = 0.0;
  *db = 0.0;
  double dx = __enzyme_autodiff(f, enzyme_context, ctx, 3.0);
  APPROX_EQ(dx, 2 * 2.0 * 3.0 + 3.0, 1e-10);
  APPROX_EQ(*dg, 3.0 * 3.0, 1e-10);
  APPROX_EQ(*db, 3.0, 1e-10);

  // Forward mode reads the tangents seeded in the globals' shadows.
  *dg = 1.0;
  *db = 0.0;
  double t = __enzyme_fwddiff(f, enzyme_context, ctx, 3.0, 0.0);
  APPROX_EQ(t, 9.0, 1e-10);

  // A context of width 2 has one shadow per lane, and states the width of
  // the derivatives requested in it.
  void *ctx2 = __enzyme_context(2);
  double *dg0 = (double *)__enzyme_shadow(ctx2, &g, 0);
  double *dg1 = (double *)__enzyme_shadow(ctx2, &g, 1);
  double *db0 = (double *)__enzyme_shadow(ctx2, &blk.b, 0);
  double *db1 = (double *)__enzyme_shadow(ctx2, &blk.b, 1);
  *dg0 = 1.0;
  *dg1 = 0.0;
  *db0 = 0.0;
  *db1 = 1.0;
  Tangents t2 = __enzyme_fwddiff2(f, enzyme_context, ctx2, 3.0, 0.0, 0.0);
  APPROX_EQ(t2.d1, 9.0, 1e-10);
  APPROX_EQ(t2.d2, 3.0, 1e-10);

  // The shadows of the first context are unchanged.
  APPROX_EQ(*dg, 1.0, 1e-10);
  return 0;
}
