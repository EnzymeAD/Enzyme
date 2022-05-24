#include <assert.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

double r2() {
  return (double) rand() / (double) RAND_MAX ;
}

void randinit(double *v, int n) {
  for (int i = 0; i != n; ++i) {
    v[i] = r2();
  }
}

double sum(double *v, int n) {
  double s = 0;
  for (int i = 0; i != n; ++i) {
    s += v[i];
  }
  return s;
}

double sqnm(double *v, int n) {
  double s = 0;
  for (int i = 0; i != n; ++i) {
    s += v[i] * v[i];
  }
  return s;
}

void dlaswp(int n, double *a, int lda, int k1, int k2, int *ipiv, int incx) {
  int ix0, i1, i2, inc = 0;
  if (incx > 0) {
    ix0 = k1;
    i1 = k1;
    i2 = k2;
    inc = 1;
  } else if (incx < 0) {
    ix0 = k1 + (k1 - k2) * incx;
    i1 = k2;
    i2 = k1;
    inc = -1;
  } else {
    return;
  }
  int ix = ix0;
  for (int i = i1; i != i2; ++i) {
    int ip = ipiv[ix];
    for (int j = 0; j != n; ++j) {
      int tmp = a[j * lda + i];
      a[j * lda + i] = a[j * lda + ip];
      a[j * lda + ip] = tmp;
    }
    ix += incx;
  }
}


int iladlc(int m, int n, double *a, int lda) {
  for (int j = n - 1; j != -1; --j) {
    for (int i = 0; i != m; ++i) {
      if (a[i + j * lda] != 0) return j;
    }
  }
  return 0;
}

int iladlr(int m, int n, double *a, int lda) {
  int result = 0;
  int i;
  for (int j = 0; j != n; ++j) {
    i = m - 1;
    while (i >= 0 && a[i + j * lda] == 0) {
      i -= 1;
    }
    result = MAX(result, i);
  }
  return 0;
}

void dlarf(CBLAS_SIDE side, int m, int n, double *v, int incv, double tau, double *c, int ldc, double *work) {
  int isLeft = (side == CblasLeft);
  int lastv = (isLeft ? m : n);
  int i = (incv > 0) ? (lastv - 1) * incv : 0;
  while (lastv > 0 && v[i] == 0) {
    lastv -= 1;
    i -= incv;
  }
  int lastc = isLeft ? iladlc(lastv, n, c, ldc) : iladlr(m, lastv, c, ldc);
  if (isLeft) {
    if (lastv > 0) {
      cblas_dgemv(CblasColMajor, CblasTrans, lastv, lastc, 1.0, c, ldc, v, incv, 0.0, work, 1);
      cblas_dger(CblasColMajor, lastv, lastc, -tau, v, incv, work, 1, c, ldc);
    }
  } else {
    if (lastv > 0) {
      cblas_dgemv(CblasColMajor, CblasNoTrans, lastc, lastv, 1.0, c, ldc, v, incv, 0, work, 1);
      cblas_dger(CblasColMajor, lastc, lastv, -tau, work, 1, v, incv, c, ldc);
    }
  }
}

void dlarfg(int n, double *alpha, double *x, int incx, double *tau) {
  if (n <= 1) {
    *tau = 0;
    return;
  }
  double xnorm = cblas_dnrm2(n - 1, x, incx);
  if (xnorm == 0) {
    *tau = 0;
    return;
  }
  double av = *alpha;
  double lapy = sqrt(av * av + xnorm * xnorm);
  double beta = av >= 0 ? lapy : -lapy;
  *tau = (beta - av) / beta;
  cblas_dscal(n - 1, 1.0 / (av - beta), x, incx);
  *alpha = beta;
}
