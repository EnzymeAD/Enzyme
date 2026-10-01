// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Seed and read the shadows of globals that Enzyme creates itself, at
// width 1 and width 2, through __enzyme_shadow.

#include "../test_utils.h"

typedef struct {
  double d1, d2;
} Tangents;

extern void *__enzyme_shadow(void *, int, int);
extern double __enzyme_fwddiff(double (*)(double), ...);
extern Tangents __enzyme_fwddiff2(double (*)(double), ...)
    __asm__("__enzyme_fwddiff");
extern double __enzyme_autodiff(double (*)(double), ...);
extern int enzyme_width;

double g = 2.0;

// Like a Fortran COMMON block: the query names a member at an offset.
struct {
  double a;
  double b;
} blk = {1.0, 3.0};

double f(double x) { return g * x * x + blk.b * x; }

int main() {
  double *dg = (double *)__enzyme_shadow(&g, 1, 0);
  double *db = (double *)__enzyme_shadow(&blk.b, 1, 0);

  // Reverse mode accumulates the adjoints of the globals.
  *dg = 0.0;
  *db = 0.0;
  double dx = __enzyme_autodiff(f, 3.0);
  APPROX_EQ(dx, 2 * 2.0 * 3.0 + 3.0, 1e-10);
  APPROX_EQ(*dg, 3.0 * 3.0, 1e-10);
  APPROX_EQ(*db, 3.0, 1e-10);

  // Forward mode reads the tangents seeded in the globals' shadows.
  *dg = 1.0;
  *db = 0.0;
  double t = __enzyme_fwddiff(f, 3.0, 0.0);
  APPROX_EQ(t, 9.0, 1e-10);

  // Width 2 has its own shadows, one per lane.
  double *dg0 = (double *)__enzyme_shadow(&g, 2, 0);
  double *dg1 = (double *)__enzyme_shadow(&g, 2, 1);
  double *db0 = (double *)__enzyme_shadow(&blk.b, 2, 0);
  double *db1 = (double *)__enzyme_shadow(&blk.b, 2, 1);
  *dg0 = 1.0;
  *dg1 = 0.0;
  *db0 = 0.0;
  *db1 = 1.0;
  Tangents t2 = __enzyme_fwddiff2(f, enzyme_width, 2, 3.0, 0.0, 0.0);
  APPROX_EQ(t2.d1, 9.0, 1e-10);
  APPROX_EQ(t2.d2, 3.0, 1e-10);

  // The width 1 shadow is unchanged.
  APPROX_EQ(*dg, 1.0, 1e-10);
  return 0;
}
