#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include "helper.h"

void dsytd(int n, double *a, int lda, double *d, double *e, double *tau) {
  double taui, alpha;
  if (n <= 0) return;
  for (int i = 0; i != n - 1; ++i) {
    dlarfg(n - i - 1, a + (i + 1) + i * lda, a + MIN(i + 2, n - 1) + i * lda, 1, &taui);
    e[i] = a[i + 1 + i * lda];
    if (taui != 0) {
      a[i + 1 + i * lda] = 1.0;
      cblas_dgemv(CblasColMajor, CblasNoTrans, n - i - 1, n - i - 1, taui, a + (i + 1) + (i + 1) * lda, lda, a + (i + 1) + i * lda, 1, 0.0, tau + i, 1);
      alpha = -0.5 * taui * cblas_ddot(n - i - 1, tau + i, 1, a + (i + 1) + i * lda, 1);
      cblas_daxpy(n - i - 1, alpha, a + (i + 1) + i * lda, 1, tau + i, 1);
      cblas_dsyr2(CblasColMajor, CblasLower, n - i - 1, -1.0, a + (i + 1) + i * lda, 1, tau + i, 1, a + (i + 1) + (i + 1) * lda, lda);
      a[i + 1 + i * lda] = e[i];
    }
    d[i] = a[i + i * lda];
    tau[i] = taui;
  }
  d[n - 1] = a[(n - 1) + (n - 1) * lda];
}

double wrapper(int n, double *a) {
  double *d = (double *) malloc(sizeof(double) * n);
  double *e = (double *) malloc(sizeof(double) * n);
  double *tau = (double *) malloc(sizeof(double) * n);
  dsytd(n, a, n, d, e, tau);
  return sum(a, n * n);
}

extern void __enzyme_autodiff(void *, int, double *, double *);

int main() {
  const int n = 10;
  double a[n * n], _a[n * n] = {0.0};
  randinit(a, n * n);
  __enzyme_autodiff((void *) wrapper, n, a, _a);
}
