#include <stdio.h>

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
                const double alpha, const double *X, const int incX,
                const double *Y, const int incY, double *A, const int lda);

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);

void f(double *restrict x, double *restrict y, double *restrict v, double *restrict w, double *restrict C) {
    double A[] = {0.00, 0.00, 0.00, 0.00,
                 0.00, 0.00, 0.00, 0.00,
                 0.00};
    double B[] = {0.00, 0.00, 0.00, 0.00,
                 0.00, 0.00, 0.00, 0.00,
                 0.00};
    int lda = 3, ldb = 3, ldc = 3;
    int m = 3, n = 3, p = 3;
    double alpha = 3.14, beta = 4.31;
    int incx = 1, incy = 1, incv = 1, incw = 1;

    cblas_dger(CblasRowMajor, m, n, alpha, x, incx, y, incy, A, lda);
    cblas_dger(CblasRowMajor, n, p, beta, v, incv, w, incw, B, ldb);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, A, lda, B, ldb, beta, C, ldc);
}

int main() {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {2.0, 3.0, 4.0};
    double v[] = {0.0, 2.5, 3.5};
    double w[] = {1.0, 1.0, 1.0};
    double C[] = {0.00, 0.00, 0.00, 0.00,
                 0.00, 0.00, 0.00, 0.00,
                 0.00};
    f(x,y,v,w,C);
    for (int i = 0; i < 9; i++)
      printf("%f\n", C[i]);
}
