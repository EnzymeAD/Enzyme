#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include "helper.h"

void dgebrd(int m, int n, double *a, int lda, double *d, double *e, double *tauq, double *taup, double *work) {
  for (int i = 0; i != n; ++i) {
    dlarfg(m - i, a + i + i * lda, a + MIN(i + 1, m - 1) + i * lda, 1, tauq + i);
    d[i] = a[i + i * lda];
    a[i + i * lda] = 1.0;
    if (i < n - 1) {
      dlarf(CblasLeft, m - i, n - i - 1, a + i + i * lda, 1, tauq[i], a + i + (i + 1) * lda, lda, work);
    }
    a[i + i * lda] = d[i];
    if (i < n - 1) {
      dlarfg(n - i - 1, a + i + (i + 1) * lda, a + i + MIN(i + 2, n - 1) * lda, lda, taup + i);
      e[i] = a[i + (i + 1) * lda];
      a[i + (i + 1) * lda] = 1.0;
      dlarf(CblasRight, m - i - 1, n - i - 1, a + i + (i + 1) * lda, lda, taup[i], a + (i + 1) + (i + 1) * lda, lda, work);
      a[i + (i + 1) * lda] = e[i];
    } else {
      taup[i] = 0.0;
    }
  }
}

double wrapper(int n, double *a) {
  double *d = (double *) malloc(sizeof(double) * n);
  double *e = (double *) malloc(sizeof(double) * n);
  double *tauq = (double *) malloc(sizeof(double) * n);
  double *taup = (double *) malloc(sizeof(double) * n);
  double *work = (double *) malloc(sizeof(double) * n);
  dgebrd(n, n, a, n, d, e, tauq, taup, work);
  return sum(a, n * n);
}

extern void __enzyme_autodiff(void *, int, double *, double *);

int main() {
  const int n = 10;
  double a[n * n], _a[n * n] = {0.0};
  randinit(a, n * n);
  __enzyme_autodiff((void *) wrapper, n, a, _a);
}
