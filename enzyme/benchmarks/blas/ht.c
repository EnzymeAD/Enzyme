#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "helper.h"

double ht(double *X, double *Y, double *W, double *T, double *Rt, int m) {
  double *Xbar = (double *) malloc(sizeof(double) * 4 * m);
  double *Z = (double *) malloc(sizeof(double) * 4 * m);
  double *V = (double *) malloc(sizeof(double) * 3 * m);
  memset(Z, 0, sizeof(double) * 4 * m);
  for (int i = 0; i != 22; ++i) {
    for (int j = 0; j != m; ++j) {
      cblas_dcopy(3, X + 3 * j, 1, Xbar + 4 * j, 1);
      Xbar[4 * j + 3] = 1;
      cblas_dscal(4, W[i + j * 22], Xbar + 4 * j, 1);
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, m, 4, 1.0, T + i * 16, 4, Xbar, 4, 1.0, Z, 4);
  }
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, m, 4, 1.0, Rt, 3, Z, 4, 0.0, V, 3);
  cblas_daxpy(3 * m, -1.0, V, 1, Y, 1);
  return sqnm(V, 3 * m);
}

extern void __enzyme_autodiff(void *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, int);

int main() {
  const int m = 10;
  double X[3 * m] = {0.0}, Y[3 * m] = {0.0}, W[22 * m] = {0.0}, T[22 * 16] = {0.0}, Rt[12] = {0.0};
  randinit(X, 3 * m);
  randinit(Y, 3 * m);
  randinit(W, 22 * m);
  randinit(T, 22 * 16);
  randinit(Rt, 12);
  double _X[3 * m] = {0.0}, _Y[3 * m] = {0.0}, _W[22 * m] = {0.0}, _T[22 * 16] = {0.0}, _Rt[12] = {0.0};
  ht(X, Y, W, T, Rt, m);
  __enzyme_autodiff((void *) ht, X, _X, Y, _Y, W, _W, T, _T, Rt, _Rt, m);
  return 0;
}