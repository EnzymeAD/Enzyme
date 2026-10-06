// The store in outer iteration i+1 overwrites exactly the range that the
// load read in outer iteration i, so the loaded values must be cached for
// the reverse pass. At -O0/-O1 the store address is an AddRec of the inner
// loop only (`off` is loaded from memory), and overwritesToMemoryReadByLoop
// wrongly reports no overwrite: the reverse pass reloads the clobbering 3.0
// and every gradient in the clobbered ranges comes out as 2 * 3.0 = 6.

// RUN: %clang -std=c11 -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - | FileCheck %s --check-prefixes=COMMON,BUG
// RUN: %clang -std=c11 -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - | FileCheck %s --check-prefixes=COMMON,BUG
// RUN: %clang -std=c11 -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - | FileCheck %s --check-prefixes=COMMON,CHECK
// RUN: %clang -std=c11 -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli - | FileCheck %s --check-prefixes=COMMON,CHECK

#include <stdio.h>

extern double __enzyme_autodiff(void *, ...);
int enzyme_const;

double f(double *A, long *offs, long N, long M) {
  double s = 0;
  if (N <= 0 || M <= 0)
    return 0;
  for (long i = 0; i < N; i++) {
    long off = offs[i];
    for (long j = 0; j < M; j++) {
      double v = A[off + M + 1 + j];
      s += v * v;
      A[off + j] = 3.0;
    }
  }
  return s;
}

int main(void) {
  enum { N = 3, M = 4, SZ = 20 };
  double A[SZ], dA[SZ];
  long offs[N];
  for (int k = 0; k < SZ; k++) {
    A[k] = k + 1.0;
    dA[k] = 0.0;
  }
  // offs[i + 1] = offs[i] + M + 1: each store range is the previous read range.
  for (int i = 0; i < N; i++)
    offs[i] = (long)i * (M + 1);
  __enzyme_autodiff((void *)f, A, dA, enzyme_const, offs, (long)N, (long)M);
  for (int k = 0; k < SZ; k++)
    printf("dA[%d] = %g\n", k, dA[k]);
  return 0;
}

// The exact gradient is dA[k] = 2 * (k + 1) over the read ranges [5, 9),
// [10, 14) and [15, 19), and 0 elsewhere.

// COMMON: dA[0] = 0
// COMMON-NEXT: dA[1] = 0
// COMMON-NEXT: dA[2] = 0
// COMMON-NEXT: dA[3] = 0
// COMMON-NEXT: dA[4] = 0
// CHECK-NEXT: dA[5] = 12
// CHECK-NEXT: dA[6] = 14
// CHECK-NEXT: dA[7] = 16
// CHECK-NEXT: dA[8] = 18
// BUG-NEXT: dA[5] = 6
// BUG-NEXT: dA[6] = 6
// BUG-NEXT: dA[7] = 6
// BUG-NEXT: dA[8] = 6
// COMMON-NEXT: dA[9] = 0
// CHECK-NEXT: dA[10] = 22
// CHECK-NEXT: dA[11] = 24
// CHECK-NEXT: dA[12] = 26
// CHECK-NEXT: dA[13] = 28
// BUG-NEXT: dA[10] = 6
// BUG-NEXT: dA[11] = 6
// BUG-NEXT: dA[12] = 6
// BUG-NEXT: dA[13] = 6
// COMMON-NEXT: dA[14] = 0
// COMMON-NEXT: dA[15] = 32
// COMMON-NEXT: dA[16] = 34
// COMMON-NEXT: dA[17] = 36
// COMMON-NEXT: dA[18] = 38
// COMMON-NEXT: dA[19] = 0
