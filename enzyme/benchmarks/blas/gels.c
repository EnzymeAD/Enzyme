#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include "helper.h"

void dgelq(int m, int n, double *a, int lda, double *tau, double *work) {
  int k = MIN(m, n);
  double aii;
  for (int i = 0; i != k; ++i) {
    dlarfg(n - i, a + i + i * lda, a + i + MIN(i + 1, n - 1) * lda, lda, tau + i);
    if (i < m - 1) {
      aii = a[i + i * lda];
      a[i + i * lda] = 1.0;
      dlarf(CblasRight, m - i - 1, n - i, a + i + i * lda, lda, tau[i], a + (i + 1) + i * lda, lda, work);
      a[i + i * lda] = aii;
    }
  }
}

double wrapper(int n, double *a) {
  double *tau = (double *) malloc(sizeof(double) * n);
  double *work = (double *) malloc(sizeof(double) * n);
  dgelq(n, n, a, n, tau, work);
  return sum(a, n * n);
}

extern void __enzyme_autodiff(void *, int, double *, double *);

int main() {
  const int n = 10;
  double a[n * n], _a[n * n] = {0.0};
  randinit(a, n * n);
  __enzyme_autodiff((void *) wrapper, n, a, _a);
}
