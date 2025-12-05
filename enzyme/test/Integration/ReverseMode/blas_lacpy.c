// RUN: if [ %llvmver -ge 12 ]; then %clang -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi

#include "../blas_inline.h"
#include "../test_utils.h"

#include <stdio.h>

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

void __enzyme_autodiff(void *, int *, double *, double *, int *, double *,
                       double *, int *, char);

int ROWS = 20;
int COLS = 10;

__attribute__((noinline)) void simulate(double *dst, double *src, char uplo) {
  int one = 1;
  dlacpy_(&uplo, &ROWS, &COLS, src, &ROWS, dst, &ROWS);
}

#define IDX(A, r, c) (A)[(r) + ROWS * (c)]

void set(double *dx, double *dy) {
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      IDX(dx, i, j) = (COLS * i + j) + 1;
      IDX(dy, i, j) = (COLS * i * 10 - j * 101);
    }
  }
}

void dump(double *dx, const char *name) {
  printf("Array %s:\n", name);
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      printf("%+05.0f\t", IDX(dx, i, j));
    }
    printf("\n");
  }
  printf("\n");
}

double fabs (double a);

#define dump(A) dump(A, #A)

int min(int a, int b) {
  if (a < b)
    return a;
  else
    return b;
}

int main(int argc, char **argv) {

  double dx0[ROWS * COLS];
  double dy0[ROWS * COLS];

  set((double*)dx0, (double*)dy0);

  double dx[ROWS * COLS];
  double dy[ROWS * COLS];

  double tx[ROWS * COLS];
  double ty[ROWS * COLS];

  for (int i = 0; i < 3; i++) {
    char uplo = "ULG"[i];

    set((double*)dx, (double*)dy);

    simulate((double*)dx, (double*)dy, uplo);

    for (int i = 0; i < ROWS; i++) {
      for (int j = 0; j < COLS; j++) {
        APPROX_EQ(IDX(dy, i, j), IDX(dy0, i, j), 1e-10);
        bool old = false;
        if (uplo == 'U')
          old = i > min(j, ROWS);
        else if (uplo == 'L')
          old = i < j;

        double dxv = IDX(dx, i, j);
        double dxv0 = IDX(dx0, i, j);
        double dyv0 = IDX(dy0, i, j);

        double res = old ? dxv0 : dyv0;

        if (fabs(res - dxv) > 1e-10) {
          printf("i=%d j=%d old=%d dxv=%f dxv0=%f dyv0=%f\n", i, j, old, dxv,
                 dxv0, dyv0);
        }
        APPROX_EQ(dxv, res, 1e-10);
      }
    }

    set(dx, dy);

    __enzyme_autodiff((void *)simulate, &enzyme_dup, (double*)&ty, (double*)&dy, &enzyme_dup, (double*)&tx,
                      (double*)&dx, (double*)&enzyme_const, uplo);

    for (int i = 0; i < ROWS; i++) {
      for (int j = 0; j < COLS; j++) {
        bool old = false;
        if (uplo == 'U')
          old = i > min(j, ROWS);
        else if (uplo == 'L')
          old = i < j;

        double dxv = IDX(dx, i, j);
        double dxv0 = IDX(dx0, i, j);
        double dyv0 = IDX(dy0, i, j);

        double res = old ? dxv0 : (dxv0 + dyv0);

        if (fabs(res - dxv) > 1e-10) {
          dump(dx0);
          dump(dy0);
          // should be dx += dy
          // dy .= 0

          dump(dx);
          dump(dy);
          printf("uplo=%c i=%d j=%d old=%d dxv=%f dxv0=%f dyv0=%f\n", uplo, i,
                 j, old, dxv, dxv0, dyv0);
        }
        APPROX_EQ(dxv, res, 1e-10);

        if (old) {
          APPROX_EQ(IDX(dy, i, j), IDX(dy0, i, j), 1e-10);
        } else {
          APPROX_EQ(IDX(dy, i, j), 0.0, 1e-10);
        }
      }
    }
  }

  return 0;
}
