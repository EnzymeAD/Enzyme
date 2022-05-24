#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

double r2() {
  return (double) rand() / (double) RAND_MAX ;
}

void randinit(double *v, int n) {
  for (int i = 0; i != n; ++i) {
    v[i] = r2();
  }
}

double sqnm(double *v, int n) {
  double s = 0;
  for (int i = 0; i != n; ++i) {
    s += v[i] * v[i];
  }
  return s;
}

void rodrigues(double *out, double *r, double *x, int n) {
  double n2 = sqnm(r, 3);
  cblas_dscal(3, 1 / n2, r, 1);
  for (int i = 0; i != n; ++i) {
    out[3 * i] = x[3 * i] * cos(n2) + (r[1] * x[3 * i + 2] - r[2] * x[3 * i + 1]) * sin(n2);
    out[3 * i + 1] = x[3 * i + 1] * cos(n2) + (r[2] * x[3 * i] - r[0] * x[3 * i + 2]) * sin(n2);
    out[3 * i + 2] = x[3 * i + 2] * cos(n2) + (r[0] * x[3 * i + 1] - r[1] * x[3 * i]) * sin(n2);
  }
  double *mat = (double *) malloc(sizeof(double) * 9);
  for (int i = 0; i != 3; ++i) {
    for (int j = 0; j != 3; ++j) {
      mat[3 * i + j] = r[i] * r[j] * (1 - cos(n2));
    }
  }
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, 3, n, 1.0, mat, 3, x, 3, 0.0, out, 3);
  printf("Hello world!\n");
}

void p2e(double *x, int n) {
  for (int i = 0; i != n; ++i) {
    cblas_dscal(2, 1 / x[3 * i + 2], x + 3 * i, 1);
  }
}

void distort(double *kappa, double *x, int n) {
  for (int i = 0; i != n; ++i) {
    double n2 = x[3 * i] * x[3 * i] + x[3 * i + 1] * x[3 * i + 1] + x[3 * i + 2] * x[3 * i + 2];
    double scale = 1 + kappa[0] * n2 + kappa[1] * n2 * n2;
    cblas_dscal(2, scale, x + 3 * i, 1);
  }
}

double output(double w, double *m, double *r, double *c, double *x0, double *kappa, int n) {
  double *x = (double *) malloc(sizeof(double) * 3 * n);
  randinit(x, 3 * n);
  double *rod = (double *) malloc(sizeof(double) * 3 * n);
  for (int i = 0; i != n; ++i) {
    cblas_daxpy(2, -1.0, c, 1, x + (3 * i), 1);
  }
  double f = 0.1;
  rodrigues(rod, r, x, n);
  p2e(rod, n);
  distort(kappa, rod, n);
  for (int i = 0; i != n; ++i) {
    cblas_daxpy(2, f, x0, 1, rod + (3 * i), 1);
    cblas_dscal(2, -1.0, rod + (3 * i), 1);
    cblas_daxpy(2, 1.0, m, 1, rod + (3 * i), 1);
    cblas_dscal(2, w, rod + (3 * i), 1);
    rod[3 * i + 2] = 1 - w * w;
  }
  return cblas_ddot(3 * n, rod, 1, rod, 1);
}

extern double __enzyme_autodiff(void *, double, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, int);

int main() {
  const int n = 100;
  double w = 0.5, m[2] = {0.457, -0.668}, r[3] = {0.574, 0.323, 0.819}, c[3] = {1.0, 1.0, 1.0}, x0[2] = {-0.422, 0.213}, kappa[2] = {2.0, 4.0};
  double _m[2] = {0.0}, _r[3] = {0.0}, _c[3] = {0.0}, _x0[2] = {0.0}, _kappa[2] = {0.0};
  double _w = __enzyme_autodiff(output, w, m, _m, r, _r, c, _c, x0, _x0, kappa, _kappa, n);
  printf("%f\n", _m[0]);
  printf("Hello world!\n");
  return 0;
}
