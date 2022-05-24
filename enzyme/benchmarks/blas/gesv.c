#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include "helper.h"

int idamax(int n, double *x, int incx) {
  if (n <= 1) return 0;
  int result = 0, ix = 0;
  double dmax = fabs(x[0]);
  ix += incx;
  for (int i = 1; i != n; ++i) {
    if (fabs(x[ix]) > dmax) {
      result = i;
      dmax = fabs(x[ix]);
    }
    ix += incx;
  }
  return result;
}

void dgetrf(int m, int n, double *a, int lda, int *ipiv) {
  if (!m || !n) return;
  if (m == 1) {
    ipiv[0] = 0;
    assert(a[0] != 0);
  } else if (n == 1) {
    int i = idamax(m, a, 1);
    ipiv[0] = 0;
    assert(a[i] != 0);
    if (i != 0) {
      double tmp = a[0];
      a[0] = a[i];
      a[i] = tmp;
    }
    cblas_dscal(m - 1, 1 / a[0], a + 1, 1);
  } else {
    int n1 = MIN(m, n) / 2;
    int n2 = n - n1;
    dgetrf(m, n1, a, lda, ipiv);
    dlaswp(n2, a + n1 * lda, lda, 1, n1, ipiv, 1);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n1, n2, 1.0, a, lda, a + n1 * lda, lda);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m - n1, n2, n1, -1.0, a + n1, lda, a + n1 * lda, lda, 1.0, a + n1 * lda + n1, lda);
    dgetrf(m - n1, n2, a + n1 * lda + n1, lda, ipiv + n1);
  }
}

void dgetrs(CBLAS_ORDER order, CBLAS_TRANSPOSE trans, int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) {
  if (trans == CblasNoTrans) {
    dlaswp(nrhs, b, ldb, 1, n, ipiv, 1);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, nrhs, 1.0, a, lda, b, ldb);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, nrhs, 1.0, a, lda, b, ldb);
  } else {
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, n, nrhs, 1.0, a, lda, b, ldb);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit, n, nrhs, 1.0, a, lda, b, ldb);
    dlaswp(nrhs, b, ldb, 1, n, ipiv, -1);
  }
}

void dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) {
  dgetrf(n, n, a, lda, ipiv);
  dgetrs(CblasColMajor, CblasNoTrans, n, nrhs, a, lda, ipiv, b, ldb);
}

double wrapper(int n, double *a, double *b) {
  int *ipiv = (int *) malloc(sizeof(int) * n);
  dgesv(n, 1, a, n, ipiv, b, n);
  return sum(b, n);
}

extern void __enzyme_autodiff(void *, int, double *, double *, double *, double *);

int main() {
  const int n = 10;
  double a[n * n], _a[n * n] = {0.0}, b[n], _b[n] = {0.0};
  randinit(a, n * n);
  randinit(b, n);
  __enzyme_autodiff((void *) wrapper, n, a, _a, b, _b);
}