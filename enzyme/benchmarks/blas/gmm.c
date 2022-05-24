#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "helper.h"

double logsumexp(double *v, int n) {
  double s = 0;
  for (int i = 0; i != n; ++i) {
    s += exp(v[i]);
  }
  return log(s);
}

void buildQ(double *Q, double *l, double *q, int n) {
  for (int i = 0; i != n; ++i) {
    Q[i * n + i] = exp(q[i]);
    for (int j = i + 1; j != n; ++j) {
      int skip = (i + 1) * i / 2;
      Q[i * n + j] = l[i * n + j - skip];
    }
  }
}

extern void __enzyme_autodiff(void *, double *, double *, double *, double *, double *, double *, double *, double *, int, int, int);

double loss(double *alpha, double *ms, double *qs, double *ls, int n, int k, int d) {
  int d2 = d * (d - 1) / 2;
  double *tmp = (double *) malloc(sizeof(double) * k);
  double *eqs = (double *) malloc(sizeof(double) * d * k);
  double *Q = (double *) malloc(sizeof(double) * d * d);
  double *mvresult = (double *) malloc(sizeof(double) * d);
  double *mmresult = (double *) malloc(sizeof(double) * d * n);
  memset(tmp, 0, k * sizeof(double));
  memset(eqs, 0, d * k * sizeof(double));
  memset(Q, 0, d * d * sizeof(double));
  memset(mvresult, 0, d * sizeof(double));
  memset(mmresult, 0, d * n * sizeof(double));

  double *X = (double *) malloc(sizeof(double) * d * n);
  memset(X, 0, d * n * sizeof(double));

  for (int i = 0; i != k; ++i) {
    tmp[i] = n * (alpha[k] + sum(qs + i * d, d));
    buildQ(Q, ls + i * d2, qs + i * d, d);
    cblas_dgemv(CblasColMajor, CblasNoTrans, d, d, 1.0, Q, d, ms + i * d, 1, 0.0, mvresult, 1);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, n, 1.0, Q, d, X, d, 0.0, mmresult, d);
    for (int j = 0; j != n; ++j) {
      cblas_daxpy(d, -1.0, mvresult, 1, mmresult + j * d, 1);
    }
    tmp[i] -= sqnm(mmresult, d * n) / 2;
  }
  double l = logsumexp(tmp, k) - n * logsumexp(alpha, k);
  for (int i = 0; i != d * k; ++i) {
    eqs[i] = exp(qs[i]);
  }
  l += cblas_ddot(d * k, eqs, 1, eqs, 1) / 2;
  l += cblas_ddot(d2 * k, ls, 1, ls, 1) / 2;
  l -= sum(qs, d * k);
  return l;
}

int main() {
  const int n = 100, k = 6, d = 100;
  const int vsize = d * k;
  const int lsize = ((d * (d - 1)) / 2) * k;
  double alpha[k] = {1, 2, 3, 4, 5, 6}, qs[vsize] = {0.0}, ls[lsize] = {0.0}, ms[vsize] = {0.0};
  // randinit(qs, vsize);
  // randinit(ms, vsize);
  // randinit(ls, lsize);
  printf("Hello world!\n");
  double _alpha[k] = {0.0}, _qs[vsize] = {0.0}, _ls[lsize] = {0.0}, _ms[vsize] = {0.0};
  __enzyme_autodiff((void *) loss, alpha, _alpha, ms, _ms, qs, _qs, ls, _ls, n, k, d);
  printf("%f\n", _alpha[0]);
  printf("Hello world!\n");
  return 0;
}
