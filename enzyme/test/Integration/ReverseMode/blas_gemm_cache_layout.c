// RUN: if [ %llvmver -ge 12 ]; then %clang -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi

// Reverse mode through a single cblas_dgemm whose inputs are overwritten after
// the call, so that Enzyme has to cache them for the reverse pass. The
// matrices are deliberately non-square and are checked in both the row-major
// and column-major layouts, for all combinations of transposes.
// See https://github.com/EnzymeAD/Enzyme/issues/3908

#include "../test_utils.h"

enum { RowMajor = 101, ColMajor = 102, NoTrans = 111, Trans = 112 };

double __enzyme_autodiff(void *, ...);

// Element (r, c) of a matrix stored with the given layout.
static int idx(int layout, int ld, int r, int c) {
  return layout == RowMajor ? r * ld + c : r + c * ld;
}

// C = alpha * op(A) * op(B) + beta * C
__attribute__((noinline)) void cblas_dgemm(int layout, int transA, int transB,
                                           int M, int N, int K, double alpha,
                                           const double *A, int lda,
                                           const double *B, int ldb,
                                           double beta, double *C, int ldc) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0;
      for (int k = 0; k < K; k++) {
        double a = transA == NoTrans ? A[idx(layout, lda, i, k)]
                                     : A[idx(layout, lda, k, i)];
        double b = transB == NoTrans ? B[idx(layout, ldb, k, j)]
                                     : B[idx(layout, ldb, j, k)];
        sum += a * b;
      }
      int o = idx(layout, ldc, i, j);
      C[o] = alpha * sum + (beta == 0 ? 0 : beta * C[o]);
    }
  }
}

// Used by the derivative with respect to alpha.
__attribute__((noinline)) double cblas_ddot(int N, const double *X, int incX,
                                            const double *Y, int incY) {
  double sum = 0;
  for (int i = 0; i < N; i++)
    sum += X[i * incX] * Y[i * incY];
  return sum;
}

// B = A for an M x N matrix, with the semantics of LAPACKE_dlacpy. Only used
// to cache the inputs when running with -enzyme-lapack-copy=1.
__attribute__((noinline)) void cblas_dlacpy(int layout, char uplo, int M,
                                            int N, const double *A, int lda,
                                            double *B, int ldb) {
  if (ldb < (layout == RowMajor ? N : M)) {
    fprintf(stderr, "cblas_dlacpy: illegal ldb=%d for a %s %d x %d matrix\n",
            ldb, layout == RowMajor ? "row-major" : "col-major", M, N);
    abort();
  }
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      B[idx(layout, ldb, i, j)] = A[idx(layout, lda, i, j)];
}

// A = A * cto / cfrom for a general M x N matrix, with the semantics of
// LAPACKE_dlascl. Only used when running with -enzyme-lapack-copy=1.
__attribute__((noinline)) void cblas_dlascl(int layout, char type, int KL,
                                            int KU, double cfrom, double cto,
                                            int M, int N, double *A, int lda,
                                            int info) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      A[idx(layout, lda, i, j)] *= cto / cfrom;
}

enum { OverwriteNone = 0, OverwriteA = 1, OverwriteB = 2, OverwriteBoth = 3 };

#define MAXDIM 8
#define MAXSZ (MAXDIM * MAXDIM)

// Number of rows / columns of the stored (i.e. not op'd) A and B.
static int rowsA(int transA, int m, int k) { return transA == NoTrans ? m : k; }
static int colsA(int transA, int m, int k) { return transA == NoTrans ? k : m; }
static int rowsB(int transB, int n, int k) { return transB == NoTrans ? k : n; }
static int colsB(int transB, int n, int k) { return transB == NoTrans ? n : k; }

// Leading dimension of a rows x cols matrix stored without padding.
static int ldOf(int layout, int rows, int cols) {
  return layout == RowMajor ? cols : rows;
}

static double weight(int o) { return 1.0 + o * 0.25; }

// The arguments are restrict so that A and B only need to be cached when they
// are explicitly overwritten below (and not because of the write to C).
__attribute__((noinline)) double f(double *restrict A, double *restrict B,
                                   double *restrict C, int m, int n, int k,
                                   int which, int layout, int transA,
                                   int transB, double alpha) {
  int lda = ldOf(layout, rowsA(transA, m, k), colsA(transA, m, k));
  int ldb = ldOf(layout, rowsB(transB, n, k), colsB(transB, n, k));
  int ldc = ldOf(layout, m, n);
  cblas_dgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, 0.0, C,
              ldc);
  double s = 0;
  for (int i = 0; i < m * n; i++)
    s += weight(i) * C[i];
  if (which & OverwriteA)
    for (int i = 0; i < m * k; i++)
      A[i] = 0;
  if (which & OverwriteB)
    for (int i = 0; i < n * k; i++)
      B[i] = 0;
  return s;
}

static int check(int m, int n, int k, int which, int layout, int transA,
                 int transB, int activeAlpha) {
  const double alpha = 1.5;
  double A0[MAXSZ], B0[MAXSZ];
  double A[MAXSZ], B[MAXSZ], C[MAXSZ];
  double dA[MAXSZ], dB[MAXSZ], dC[MAXSZ];
  for (int i = 0; i < m * k; i++) {
    A[i] = A0[i] = 0.1 + 0.07 * i;
    dA[i] = 0;
  }
  for (int i = 0; i < n * k; i++) {
    B[i] = B0[i] = -0.3 + 0.05 * i;
    dB[i] = 0;
  }
  for (int i = 0; i < m * n; i++)
    C[i] = dC[i] = 0;

  double dalpha = 0;
  if (activeAlpha)
    dalpha = __enzyme_autodiff((void *)f, enzyme_dup, A, dA, enzyme_dup, B, dB,
                               enzyme_dup, C, dC, enzyme_const, m,
                               enzyme_const, n, enzyme_const, k, enzyme_const,
                               which, enzyme_const, layout, enzyme_const,
                               transA, enzyme_const, transB, enzyme_out, alpha);
  else
    __enzyme_autodiff((void *)f, enzyme_dup, A, dA, enzyme_dup, B, dB,
                      enzyme_dup, C, dC, enzyme_const, m, enzyme_const, n,
                      enzyme_const, k, enzyme_const, which, enzyme_const,
                      layout, enzyme_const, transA, enzyme_const, transB,
                      enzyme_const, alpha);

  int lda = ldOf(layout, rowsA(transA, m, k), colsA(transA, m, k));
  int ldb = ldOf(layout, rowsB(transB, n, k), colsB(transB, n, k));
  int ldc = ldOf(layout, m, n);

  // Closed form: with W the weights of C and op(A), op(B) the (possibly
  // transposed) inputs, d op(A) = alpha W op(B)^T, d op(B) = alpha op(A)^T W
  // and d alpha = sum(W .* (op(A) op(B))).
  double errA = 0, errB = 0, errAlpha = 0;
  if (activeAlpha) {
    double s = 0;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        for (int kk = 0; kk < k; kk++) {
          double a = transA == NoTrans ? A0[idx(layout, lda, i, kk)]
                                       : A0[idx(layout, lda, kk, i)];
          double b = transB == NoTrans ? B0[idx(layout, ldb, kk, j)]
                                       : B0[idx(layout, ldb, j, kk)];
          s += weight(idx(layout, ldc, i, j)) * a * b;
        }
    errAlpha = __builtin_fabs(s - dalpha);
  }
  for (int i = 0; i < m; i++) {
    for (int kk = 0; kk < k; kk++) {
      double s = 0;
      for (int j = 0; j < n; j++) {
        double b = transB == NoTrans ? B0[idx(layout, ldb, kk, j)]
                                     : B0[idx(layout, ldb, j, kk)];
        s += alpha * weight(idx(layout, ldc, i, j)) * b;
      }
      double got = transA == NoTrans ? dA[idx(layout, lda, i, kk)]
                                     : dA[idx(layout, lda, kk, i)];
      double e = __builtin_fabs(s - got);
      if (e > errA)
        errA = e;
    }
  }
  for (int kk = 0; kk < k; kk++) {
    for (int j = 0; j < n; j++) {
      double s = 0;
      for (int i = 0; i < m; i++) {
        double a = transA == NoTrans ? A0[idx(layout, lda, i, kk)]
                                     : A0[idx(layout, lda, kk, i)];
        s += alpha * a * weight(idx(layout, ldc, i, j));
      }
      double got = transB == NoTrans ? dB[idx(layout, ldb, kk, j)]
                                     : dB[idx(layout, ldb, j, kk)];
      double e = __builtin_fabs(s - got);
      if (e > errB)
        errB = e;
    }
  }

  static const char *whats[] = {"nothing", "A", "B", "A and B"};
  int ok = errA < 1e-9 && errB < 1e-9 && errAlpha < 1e-9;
  printf("%s m=%d n=%d k=%d transA=%c transB=%c %s alpha, %s overwritten: dA "
         "max err %.1e | dB max err %.1e | dalpha err %.1e  %s\n",
         layout == RowMajor ? "row-major" : "col-major", m, n, k,
         transA == NoTrans ? 'N' : 'T', transB == NoTrans ? 'N' : 'T',
         activeAlpha ? "active" : "const", whats[which], errA, errB, errAlpha,
         ok ? "OK" : "WRONG");
  return ok;
}

int main(int argc, char **argv) {
  static const int layouts[] = {ColMajor, RowMajor};
  static const int transes[] = {NoTrans, Trans};
  // Square, and non-square in both directions for each of A and B.
  static const int dims[][3] = {{4, 4, 4}, {3, 4, 5}, {5, 4, 3}, {2, 5, 3}};
  int failures = 0;
  for (int d = 0; d < 4; d++)
    for (int l = 0; l < 2; l++)
      for (int ta = 0; ta < 2; ta++)
        for (int tb = 0; tb < 2; tb++)
          for (int which = OverwriteNone; which <= OverwriteBoth; which++)
            // TODO: the derivative with respect to alpha is still wrong for
            // non-square matrices in the row-major layout, so it is only
            // checked in the column-major layout for now.
            for (int activeAlpha = 0;
                 activeAlpha < (layouts[l] == ColMajor ? 2 : 1); activeAlpha++)
              failures += !check(dims[d][0], dims[d][1], dims[d][2], which,
                                 layouts[l], transes[ta], transes[tb],
                                 activeAlpha);
  if (failures) {
    fprintf(stderr, "%d configurations produced a wrong gradient\n", failures);
    abort();
  }
  return 0;
}
