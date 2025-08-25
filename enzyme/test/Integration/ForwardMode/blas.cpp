// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1  | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-lapack-copy=1 | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -mllvm -enzyme-lapack-copy=1 -S | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -mllvm -enzyme-lapack-copy=1 -S | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -mllvm -enzyme-lapack-copy=1  -S | %lli -; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// TODO: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi


#include "../blasinfra.h"
#include "../test_utils.h"
extern "C" double sqrt(double);

int enzyme_dup;
int enzyme_out;
int enzyme_const;
template <typename RT, typename... T> RT __enzyme_fwddiff(void *, T...);

void my_dgemv(char layout, char trans, int M, int N, double alpha,
              double *__restrict__ A, int lda, double *__restrict__ X, int incx,
              double beta, double *__restrict__ Y, int incy) {
  cblas_dgemv(layout, trans, M, N, alpha, A, lda, X, incx, beta, Y, incy);
}

void my_dsymv(char layout, char uplo, int N, double alpha,
              double *__restrict__ A, int lda, double *__restrict__ X, int incx,
              double beta, double *__restrict__ Y, int incy) {
  cblas_dsymv(layout, uplo, N, alpha, A, lda, X, incx, beta, Y, incy);
}

double my_ddot(int N, double *__restrict__ X, int incx, double *__restrict__ Y,
               int incy) {
  double res = cblas_ddot(N, X, incx, Y, incy);
  return res;
}

double my_dnrm2(int N, double *__restrict__ X, int incx) {
  double res = cblas_dnrm2(N, X, incx);
  return res;
}

void my_dgemm(char layout, char transA, char transB, int M, int N, int K,
              double alpha, double *__restrict__ A, int lda,
              double *__restrict__ B, int ldb, double beta,
              double *__restrict__ C, int ldc) {
  cblas_dgemm(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

void ow_dgemm(char layout, char transA, char transB, int M, int N, int K,
              double alpha, double *A, int lda, double *B, int ldb, double beta,
              double *C, int ldc) {
  cblas_dgemm(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

void my_dsyrk(char layout, char uplo, char trans,
                                           int N, int K, double alpha,
                                           double *__restrict__ A, int lda, double beta,
                                           double *__restrict__ C, int ldc) {

    cblas_dsyrk(layout, uplo, trans, N, K, alpha, A, lda, beta,
                                           C, ldc);
}

void my_potrf(char layout, char uplo, int N, double *__restrict__ A, int lda) {
    int info;
    cblas_dpotrf(layout, uplo, N, A, lda, nullptr); //&info);
}

static void dotTests() {
  {
    std::string Test = "DOT active both ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, N, incA),
        /*B*/ BlasInfo(B, N, incB),
        /*C*/ BlasInfo(C, M, incC), BlasInfo(), BlasInfo(), BlasInfo(),
    };
    init();
    my_ddot(N, A, incA, B, incB);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    __enzyme_fwddiff<double>((void *)my_ddot, enzyme_const, N, enzyme_dup, A,
                             dA, enzyme_const, incA, enzyme_dup, B, dB,
                             enzyme_const, incB);
    foundCalls = calls;
    init();


    cblas_ddot(N, dA, incA, B, incB);
    cblas_ddot(N, A, incA, dB, incB);
    
    my_ddot(N, A, incA, B, incB);

    checkTest(Test);

    // Check memory of primal of expected derivative
    checkMemoryTrace(inputs, "Expected " + Test, calls);

    // Check memory of primal of our derivative (if equal above, it
    // should be the same).
    checkMemoryTrace(inputs, "Found " + Test, foundCalls);
  }
  {
    std::string Test = "DOT active A ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, N, incA),
        /*B*/ BlasInfo(B, N, incB),
        /*C*/ BlasInfo(C, M, incC), BlasInfo(), BlasInfo(), BlasInfo(),
    };
    init();
    my_ddot(N, A, incA, B, incB);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    __enzyme_fwddiff<double>((void *)my_ddot, enzyme_const, N, enzyme_dup, A,
                             dA, enzyme_const, incA, enzyme_const, B,
                             enzyme_const, incB);
    foundCalls = calls;
    init();

    cblas_ddot(N, dA, incA, B, incB);
    
    my_ddot(N, A, incA, B, incB);

    checkTest(Test);

    // Check memory of primal of expected derivative
    checkMemoryTrace(inputs, "Expected " + Test, calls);

    // Check memory of primal of our derivative (if equal above, it
    // should be the same).
    checkMemoryTrace(inputs, "Found " + Test, foundCalls);
  }
  {
    std::string Test = "DOT active B ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, N, incA),
        /*B*/ BlasInfo(B, N, incB),
        /*C*/ BlasInfo(C, M, incC), BlasInfo(), BlasInfo(), BlasInfo(),
    };
    init();
    my_ddot(N, A, incA, B, incB);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    __enzyme_fwddiff<double>((void *)my_ddot, enzyme_const, N, enzyme_const, A,
                             enzyme_const, incA, enzyme_dup, B, dB,
                             enzyme_const, incB);
    foundCalls = calls;
    init();


    cblas_ddot(N, A, incA, dB, incB);
    
    my_ddot(N, A, incA, B, incB);

    checkTest(Test);

    // Check memory of primal of expected derivative
    checkMemoryTrace(inputs, "Expected " + Test, calls);

    // Check memory of primal of our derivative (if equal above, it
    // should be the same).
    checkMemoryTrace(inputs, "Found " + Test, foundCalls);
  }
  {
    std::string Test = "DOT const both";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, N, incA),
        /*B*/ BlasInfo(B, N, incB),
        /*C*/ BlasInfo(C, M, incC), BlasInfo(), BlasInfo(), BlasInfo(),
    };
    init();
    my_ddot(N, A, incA, B, incB);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    double dres = __enzyme_fwddiff<double>((void *)my_ddot, enzyme_const, N,
                                           enzyme_const, A, enzyme_const, incA,
                                           enzyme_const, B, enzyme_const, incB);
    foundCalls = calls;
    init();

    my_ddot(N, A, incA, B, incB);

    checkTest(Test);

    // Check memory of primal of expected derivative
    checkMemoryTrace(inputs, "Expected " + Test, calls);

    // Check memory of primal of our derivative (if equal above, it
    // should be the same).
    checkMemoryTrace(inputs, "Found " + Test, foundCalls);
    APPROX_EQ(dres, 0.0, 1e-10);
  }
}

static void nrm2Tests() {
  {
    std::string Test = "NRM2 active";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, N, incA),
        /*B*/ BlasInfo(B, N, incB),
        /*C*/ BlasInfo(C, M, incC), BlasInfo(), BlasInfo(), BlasInfo(),
    };
    init();
    my_dnrm2(N, A, incA);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    double ADres =
        __enzyme_fwddiff<double>((void *)my_dnrm2, enzyme_const, N, enzyme_dup,
                                 A, dA, enzyme_const, incA);
    foundCalls = calls;
    init();

    double dres = cblas_ddot(N, A, incA, dA, incA);
    double nres = my_dnrm2(N, A, incA);
    double trueRes = dres / nres;
    my_dnrm2(N, A, incA);

    checkTest(Test);

    // Check memory of primal of expected derivative
    checkMemoryTrace(inputs, "Expected " + Test, calls);

    // Check memory of primal of our derivative (if equal above, it
    // should be the same).
    checkMemoryTrace(inputs, "Found " + Test, foundCalls);

    APPROX_EQ(ADres, trueRes, 1e-10);

    // TODO: Check for nres == 0.0
  }
  {
    std::string Test = "NRM2 const";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, N, incA),
        /*B*/ BlasInfo(B, N, incB),
        /*C*/ BlasInfo(C, M, incC), BlasInfo(), BlasInfo(), BlasInfo(),
    };
    init();
    my_dnrm2(N, A, incA);
    my_ddot(N, A, incA, B, incB);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    double dres = __enzyme_fwddiff<double>((void *)my_dnrm2, enzyme_const, N,
                                           enzyme_const, A, enzyme_const, incA);
    foundCalls = calls;
    init();

    my_dnrm2(N, A, incA);

    checkTest(Test);

    // Check memory of primal of expected derivative
    checkMemoryTrace(inputs, "Expected " + Test, calls);

    // Check memory of primal of our derivative (if equal above, it
    // should be the same).
    checkMemoryTrace(inputs, "Found " + Test, foundCalls);
    APPROX_EQ(dres, 0.0, 1e-10);
  }
}

static void gemvTests() {
  // N means normal matrix, T means transposed
  for (char layout : {CblasRowMajor, CblasColMajor}) {
    for (auto transA :
         {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans}) {
      // todo in fortran blas consider 'N', 'n', 'T', 't'}

      {

        bool trans = !is_normal(transA);
        std::string Test = "GEMV active A, C ";
        BlasInfo inputs[6] = {/*A*/ BlasInfo(A, layout, M, N, lda),
                              /*B*/ BlasInfo(B, trans ? M : N, incB),
                              /*C*/ BlasInfo(C, trans ? N : M, incC),
                              BlasInfo(),
                              BlasInfo(),
                              BlasInfo()};
        init();
        my_dgemv(layout, (char)transA, M, N, alpha, A, lda, B, incB, beta, C,
                 incC);

        assert(calls.size() == 1);
        assert(calls[0].inDerivative == false);
        assert(calls[0].type == CallType::GEMV);
        assert(calls[0].pout_arg1 == C);
        assert(calls[0].pin_arg1 == A);
        assert(calls[0].pin_arg2 == B);
        assert(calls[0].farg1 == alpha);
        assert(calls[0].farg2 == beta);
        assert(calls[0].layout == layout);
        assert(calls[0].targ1 == (char)transA);
        assert(calls[0].targ2 == UNUSED_TRANS);
        assert(calls[0].iarg1 == M);
        assert(calls[0].iarg2 == N);
        assert(calls[0].iarg3 == UNUSED_INT);
        assert(calls[0].iarg4 == lda);
        assert(calls[0].iarg5 == incB);
        assert(calls[0].iarg6 == incC);

        // Check memory of primal on own.
        checkMemoryTrace(inputs, "Primal " + Test, calls);

        init();
        __enzyme_fwddiff<void>(
            (void *)my_dgemv, enzyme_const, layout, enzyme_const, transA,
            enzyme_const, M, enzyme_const, N, enzyme_const, alpha, enzyme_dup,
            A, dA, enzyme_const, lda, enzyme_const, B, enzyme_const, incB,
            enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
        foundCalls = calls;
        init();

        cblas_dgemv(layout, (char)transA, M, N, alpha, dA, lda, B, incB, beta,
                    dC, incC);

        my_dgemv(layout, (char)transA, M, N, alpha, A, lda, B, incB, beta, C,
                 incC);

        // cblas_dscal(trans ? N : M, beta, dC, incC);

        checkTest(Test);

        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);

        Test = "GEMV active A, B, C ";

        init();
        __enzyme_fwddiff<void>(
            (void *)my_dgemv, enzyme_const, layout, enzyme_const, transA,
            enzyme_const, M, enzyme_const, N, enzyme_const, alpha, enzyme_dup,
            A, dA, enzyme_const, lda, enzyme_dup, B, dB, enzyme_const, incB,
            enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
        foundCalls = calls;
        init();

        cblas_dgemv(layout, (char)transA, M, N, alpha, A, lda, dB, incB, beta,
                    dC, incC);

        cblas_dgemv(layout, (char)transA, M, N, alpha, dA, lda, B, incB, 1.0, dC, incC);

        // cblas_dscal(trans ? N : M, beta, dC, incC);

        my_dgemv(layout, (char)transA, M, N, alpha, A, lda, B, incB, beta, C,
                 incC);

        // NOT ACTIVE: cblas_dgemv(layout, trans, M, N, dalpha, A, lda, B,
        // incB, 1.0, C, incC);

        checkTest(Test);

        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);

        Test = "GEMV active B, C ";

        init();
        __enzyme_fwddiff<void>(
            (void *)my_dgemv, enzyme_const, layout, enzyme_const, transA,
            enzyme_const, M, enzyme_const, N, enzyme_const, alpha, enzyme_const,
            A, enzyme_const, lda, enzyme_dup, B, dB, enzyme_const, incB,
            enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
        foundCalls = calls;
        init();

        cblas_dgemv(layout, (char)transA, M, N, alpha, A, lda, dB, incB, beta,
                    dC, incC);

        // cblas_dscal(trans ? N : M, beta, dC, incC);

        my_dgemv(layout, (char)transA, M, N, alpha, A, lda, B, incB, beta, C,
                 incC);

        // NOT ACTIVE: cblas_dgemv(layout, trans, M, N, dalpha, A, lda, B,
        // incB, 1.0, C, incC);

        checkTest(Test);

        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
      }
    }
  }
}


static void symvTests() {
  // N means normal matrix, T means transposed
  for (char layout : {CblasRowMajor, CblasColMajor}) {
    for (auto uplo : {'U', 'u', 'L', 'l'}) {
      {

        std::string Test = "SYMV active A, C ";
        BlasInfo inputs[6] = {/*A*/ BlasInfo(A, layout, N, N, lda),
                              /*B*/ BlasInfo(B, N, incB),
                              /*C*/ BlasInfo(C, N, incC),
                              BlasInfo(),
                              BlasInfo(),
                              BlasInfo()};
        init();
        my_dsymv(layout, uplo, N, alpha, A, lda, B, incB, beta, C,
                 incC);

        assert(calls.size() == 1);
        assert(calls[0].inDerivative == false);
        assert(calls[0].type == CallType::SYMV);
        assert(calls[0].pout_arg1 == C);
        assert(calls[0].pin_arg1 == A);
        assert(calls[0].pin_arg2 == B);
        assert(calls[0].farg1 == alpha);
        assert(calls[0].farg2 == beta);
        assert(calls[0].layout == layout);
        assert(calls[0].uplo == uplo);
        assert(calls[0].targ1 == UNUSED_TRANS);
        assert(calls[0].targ2 == UNUSED_TRANS);
        assert(calls[0].iarg1 == N);
        assert(calls[0].iarg3 == UNUSED_INT);
        assert(calls[0].iarg4 == lda);
        assert(calls[0].iarg5 == incB);
        assert(calls[0].iarg6 == incC);

        // Check memory of primal on own.
        checkMemoryTrace(inputs, "Primal " + Test, calls);

        init();
        __enzyme_fwddiff<void>(
            (void *)my_dsymv, enzyme_const, layout, enzyme_const, uplo,
            enzyme_const, N, enzyme_const, alpha, enzyme_dup,
            A, dA, enzyme_const, lda, enzyme_const, B, enzyme_const, incB,
            enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
        foundCalls = calls;
        init();

        cblas_dsymv(layout, uplo, N, alpha, dA, lda, B, incB, beta,
                    dC, incC);

        my_dsymv(layout, uplo, N, alpha, A, lda, B, incB, beta, C,
                 incC);

        // cblas_dscal(trans ? N : M, beta, dC, incC);

        checkTest(Test);

        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);

        Test = "SYMV active A, B, C ";

        init();
        __enzyme_fwddiff<void>(
            (void *)my_dsymv, enzyme_const, layout, enzyme_const, uplo,
            enzyme_const, N, enzyme_const, alpha, enzyme_dup,
            A, dA, enzyme_const, lda, enzyme_dup, B, dB, enzyme_const, incB,
            enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
        foundCalls = calls;
        init();

        cblas_dsymv(layout, uplo, N, alpha, A, lda, dB, incB, beta,
                    dC, incC);

        cblas_dsymv(layout, uplo, N, alpha, dA, lda, B, incB, 1.0, dC, incC);

        // cblas_dscal(trans ? N : M, beta, dC, incC);

        my_dsymv(layout, uplo, N, alpha, A, lda, B, incB, beta, C,
                 incC);

        // NOT ACTIVE: cblas_dgemv(layout, trans, M, N, dalpha, A, lda, B,
        // incB, 1.0, C, incC);

        checkTest(Test);

        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);

        Test = "SYMV active B, C ";

        init();
        __enzyme_fwddiff<void>(
            (void *)my_dsymv, enzyme_const, layout, enzyme_const, uplo,
            enzyme_const, N, enzyme_const, alpha, enzyme_const,
            A, enzyme_const, lda, enzyme_dup, B, dB, enzyme_const, incB,
            enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
        foundCalls = calls;
        init();

        cblas_dsymv(layout, uplo, N, alpha, A, lda, dB, incB, beta,
                    dC, incC);

        // cblas_dscal(trans ? N : M, beta, dC, incC);

        my_dsymv(layout, uplo, N, alpha, A, lda, B, incB, beta, C,
                 incC);

        // NOT ACTIVE: cblas_dgemv(layout, trans, M, N, dalpha, A, lda, B,
        // incB, 1.0, C, incC);

        checkTest(Test);

        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
      }
    }
  }
}

static void gemmTests() {
  // N means normal matrix, T means transposed
  for (char layout : {CblasRowMajor, CblasColMajor}) {
    for (auto transA :
         {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans}) {
      for (auto transB :
           {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans}) {
        // todo fortran blas {'N', 'n', 'T', 't'}

        {

          bool transA_bool = !is_normal(transA);
          bool transB_bool = !is_normal(transB);
          std::string Test = "GEMM Active A, B, C";
          BlasInfo inputs[6] = {/*A*/ BlasInfo(A, layout, transA_bool ? K : M,
                                               transA_bool ? M : K, lda),
                                /*B*/
                                BlasInfo(B, layout, transB_bool ? N : K,
                                         transB_bool ? K : N, incB),
                                /*C*/ BlasInfo(C, layout, M, N, incC),
                                BlasInfo(),
                                BlasInfo(),
                                BlasInfo()};
          init();
          my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda,
                   B, incB, beta, C, incC);

          assert(calls.size() == 1);
          assert(calls[0].inDerivative == false);
          assert(calls[0].type == CallType::GEMM);
          assert(calls[0].pout_arg1 == C);
          assert(calls[0].pin_arg1 == A);
          assert(calls[0].pin_arg2 == B);
          assert(calls[0].farg1 == alpha);
          assert(calls[0].farg2 == beta);
          assert(calls[0].layout == layout);
          assert(calls[0].targ1 == (char)transA);
          assert(calls[0].targ2 == (char)transB);
          assert(calls[0].iarg1 == M);
          assert(calls[0].iarg2 == N);
          assert(calls[0].iarg3 == K);
          assert(calls[0].iarg4 == lda);
          assert(calls[0].iarg5 == incB);
          assert(calls[0].iarg6 == incC);

          // Check memory of primal on own.
          checkMemoryTrace(inputs, "Primal " + Test, calls);

          init();
          __enzyme_fwddiff<void>(
              (void *)my_dgemm, enzyme_const, layout, enzyme_const, transA,
              enzyme_const, transB, enzyme_const, M, enzyme_const, N,
              enzyme_const, K, enzyme_const, alpha, enzyme_dup, A, dA,
              enzyme_const, lda, enzyme_dup, B, dB, enzyme_const, incB,
              enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
          foundCalls = calls;
          init();

          my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda,
                   dB, incB, beta, dC, incC);

          my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, dA, lda,
                   B, incB, 1.0, dC, incC);

          // NOT ACTIVE: my_dgemm(layout, (char)transA, (char)transB, M, N, K,
          // dalpha, A, lda, B, incB, 1.0, C, incC);

          // cblas_dlascl(layout, 'G', 0, 0, 1.0, beta, M, N, dC, incC, 0);

          my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda,
                   B, incB, beta, C, incC);

          checkTest(Test);

          // Check memory of primal of expected derivative
          checkMemoryTrace(inputs, "Expected " + Test, calls);

          // Check memory of primal of our derivative (if equal above, it
          // should be the same).
          checkMemoryTrace(inputs, "Found " + Test, foundCalls);

          Test = "GEMM Active A, C";

          init();
          __enzyme_fwddiff<void>(
              (void *)my_dgemm, enzyme_const, layout, enzyme_const, transA,
              enzyme_const, transB, enzyme_const, M, enzyme_const, N,
              enzyme_const, K, enzyme_const, alpha, enzyme_dup, A, dA,
              enzyme_const, lda, enzyme_const, B, enzyme_const, incB,
              enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
          foundCalls = calls;
          init();

          my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, dA, lda,
                   B, incB, beta, dC, incC);

          // NOT ACTIVE: my_dgemm(layout, (char)transA, (char)transB, M, N, K,
          // dalpha, A, lda, B, incB, 1.0, C, incC);

          // cblas_dlascl(layout, 'G', 0, 0, 1.0, beta, M, N, dC, incC, 0);

          my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda,
                   B, incB, beta, C, incC);

          checkTest(Test);

          // Check memory of primal of expected derivative
          checkMemoryTrace(inputs, "Expected " + Test, calls);

          // Check memory of primal of our derivative (if equal above, it
          // should be the same).
          checkMemoryTrace(inputs, "Found " + Test, foundCalls);

          Test = "GEMM Active B, C";

          init();
          __enzyme_fwddiff<void>(
              (void *)my_dgemm, enzyme_const, layout, enzyme_const, transA,
              enzyme_const, transB, enzyme_const, M, enzyme_const, N,
              enzyme_const, K, enzyme_const, alpha, enzyme_const, A,
              enzyme_const, lda, enzyme_dup, B, dB, enzyme_const, incB,
              enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
          foundCalls = calls;
          init();

          my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda,
                   dB, incB, beta, dC, incC);

          // NOT ACTIVE: my_dgemm(layout, (char)transA, (char)transB, M, N, K,
          // dalpha, A, lda, B, incB, 1.0, C, incC);

          // cblas_dlascl(layout, 'G', 0, 0, 1.0, beta, M, N, dC, incC, 0);

          my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda,
                   B, incB, beta, C, incC);

          checkTest(Test);

          // Check memory of primal of expected derivative
          checkMemoryTrace(inputs, "Expected " + Test, calls);

          // Check memory of primal of our derivative (if equal above, it
          // should be the same).
          checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        }
      }
    }
  }
}

static void syrkTests() {
  // N means normal matrix, T means transposed
  for (char layout : {CblasColMajor, CblasRowMajor}) {

    for (auto transA :
         {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans})

      for (auto uplo : {'U', 'u', 'L', 'l'})

      {

          bool trans = !is_normal(transA);
          BlasInfo inputs[6] = {
              /*A*/ BlasInfo(A, layout, trans ? K : N, trans ? N : K, lda),
              /*B*/ BlasInfo(),
              /*C*/ BlasInfo(C, layout, N, N, incC),
              BlasInfo(),
              BlasInfo(),
              BlasInfo(),
          };
        {

          std::string Test = "SYRK active C, A ";
          init();

          my_dsyrk(layout, uplo, (char)transA, N, K, alpha, A, lda, beta, C,
                   incC);

          assert(calls.size() == 1);
          assert(calls[0].inDerivative == false);
          assert(calls[0].type == CallType::SYRK);
          assert(calls[0].pout_arg1 == C);
          assert(calls[0].pin_arg1 == A);
          assert(calls[0].pin_arg2 == UNUSED_POINTER);
          assert(calls[0].farg1 == alpha);
          assert(calls[0].farg2 == beta);
          assert(calls[0].layout == layout);
          assert(calls[0].targ1 == (char)transA);
          assert(calls[0].targ2 == UNUSED_TRANS);
          assert(calls[0].iarg1 == N);
          assert(calls[0].iarg2 == K);
          assert(calls[0].iarg3 == UNUSED_INT);
          assert(calls[0].iarg4 == lda);
          assert(calls[0].iarg5 == incC);
          assert(calls[0].iarg6 == UNUSED_INT);
          assert(calls[0].side == UNUSED_TRANS);
          assert(calls[0].uplo == uplo);
          assert(calls[0].diag == UNUSED_TRANS);

          // Check memory of primal on own.
          checkMemoryTrace(inputs, "Primal " + Test, calls);

          init();
          __enzyme_fwddiff<void>(
              (void *)my_dsyrk, enzyme_const, layout, enzyme_const, uplo,
              enzyme_const, transA, enzyme_const, N, enzyme_const, K,
              enzyme_const, alpha, enzyme_dup, A, dA, enzyme_const, lda,
              enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
          foundCalls = calls;
          init();

          cblas_dsyr2k(layout, uplo, (char)transA, N, K, alpha, A, lda, dA, lda, beta, dC, incC);
          my_dsyrk(layout, uplo, (char)transA, N, K, alpha, A, lda, beta, C,
                   incC);

          checkTest(Test);

          // Check memory of primal of expected derivative
          checkMemoryTrace(inputs, "Expected " + Test, calls);

          // Check memory of primal of our derivative (if equal above, it
          // should be the same).
          checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        }

        {

          std::string Test = "SYRK active C ";

          init();
          __enzyme_fwddiff<void>(
              (void *)my_dsyrk, enzyme_const, layout, enzyme_const, uplo,
              enzyme_const, transA, enzyme_const, N, enzyme_const, K,
              enzyme_const, alpha, enzyme_const, A, enzyme_const, lda,
              enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
          foundCalls = calls;
          init();

          cblas_dlascl(layout, uplo, 0, 0, 1.0, beta, N, N, dC, incC, 0);
          my_dsyrk(layout, uplo, (char)transA, N, K, alpha, A, lda, beta, C,
                   incC);

          checkTest(Test);

          // Check memory of primal of expected derivative
          checkMemoryTrace(inputs, "Expected " + Test, calls);

          // Check memory of primal of our derivative (if equal above, it
          // should be the same).
          checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        }

        {

          std::string Test = "SYRK active C, alpha ";

          init();
          double dalpha = 46.1345;
          __enzyme_fwddiff<void>(
              (void *)my_dsyrk, enzyme_const, layout, enzyme_const, uplo,
              enzyme_const, transA, enzyme_const, N, enzyme_const, K,
              enzyme_dup, alpha, dalpha, enzyme_dup, A, dA, enzyme_const, lda,
              enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
          foundCalls = calls;
          init();

          cblas_dsyr2k(layout, uplo, (char)transA, N, K, alpha, A, lda, dA, lda, beta, dC, incC);
          cblas_dsyrk(layout, uplo, (char)transA, N, K, dalpha, A, lda, 1.0, dC, incC);
          my_dsyrk(layout, uplo, (char)transA, N, K, alpha, A, lda, beta, C,
                   incC);

          checkTest(Test);

          // Check memory of primal of expected derivative
          checkMemoryTrace(inputs, "Expected " + Test, calls);

          // Check memory of primal of our derivative (if equal above, it
          // should be the same).
          checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        }
      }
  }
}

static void potrfTests() {
  int N = 17;
  // N means normal matrix, T means transposed
  for (char layout : {CblasColMajor, CblasRowMajor}) {
      for (auto uplo : {'U', 'u', 'L', 'l'})

      {
          BlasInfo inputs[6] = {
              /*A*/ BlasInfo(A, layout, N, N, lda),
              /*B*/ BlasInfo(),
              /*C*/ BlasInfo(),
              BlasInfo(),
              BlasInfo(),
              BlasInfo(),
          };
        {

          std::string Test = "POTRF active A ";
          init();

          my_potrf(layout, uplo, N, A, lda);

          assert(calls.size() == 1);
          assert(calls[0].inDerivative == false);
          assert(calls[0].type == CallType::POTRF);
          assert(calls[0].pout_arg1 == A);
          assert(calls[0].pin_arg1 == UNUSED_POINTER);
          assert(calls[0].pin_arg2 == UNUSED_POINTER);
          assert(calls[0].farg1 == UNUSED_DOUBLE);
          assert(calls[0].farg2 == UNUSED_DOUBLE);
          assert(calls[0].layout == layout);
          assert(calls[0].targ1 == UNUSED_TRANS);
          assert(calls[0].targ2 == UNUSED_TRANS);
          assert(calls[0].iarg1 == N);
          assert(calls[0].iarg2 == UNUSED_INT);
          assert(calls[0].iarg3 == UNUSED_INT);
          assert(calls[0].iarg4 == lda);
          assert(calls[0].iarg5 == UNUSED_INT);
          assert(calls[0].iarg6 == UNUSED_INT);
          assert(calls[0].side == UNUSED_TRANS);
          assert(calls[0].uplo == uplo);
          assert(calls[0].diag == UNUSED_TRANS);

          // Check memory of primal on own.
          checkMemoryTrace(inputs, "Primal " + Test, calls);

          init();
          __enzyme_fwddiff<void>(
              (void *)my_potrf, enzyme_const, layout, enzyme_const, uplo,
              enzyme_const, N, enzyme_dup, A, dA, enzyme_const, lda);
          foundCalls = calls;
          init();

          my_potrf(layout, uplo, N, A, lda);

          assert(foundCalls.size() >= 2);
          assert(foundCalls[1].type == CallType::LACPY);
          double* tri = (double*)foundCalls[1].pout_arg1;
		  inputs[3] = BlasInfo(tri, layout, N, N, N);
          cblas_dlacpy(layout, flip_uplo(uplo), N, N, dA, lda, tri, N);

#define dAv(r, c)                                                              \
  dA[(r) * (layout == CblasRowMajor ? lda : 1) +                               \
     (c) * (layout == CblasRowMajor ? 1 : lda)]

          int upperinc = (&dAv(0, 1) - &dAv(0, 0));
          int lowerinc = (&dAv(1, 0) - &dAv(0, 0));
          if (layout == CblasColMajor) {
            assert(upperinc == lda);
            assert(lowerinc == 1);
          } else {
            assert(upperinc == 1);
            assert(lowerinc == lda);
          }
          bool is_lower = uplo == 'L' || uplo == 'l';
          for (int i = 0; i < N - 1; i++) {
            cblas_dcopy(N - i - 1,
                        is_lower ? (&dAv(i + 1, i)) : (&dAv(i, i + 1)),
                        is_lower ? lowerinc : upperinc,
                        is_lower ? (&dAv(i, i + 1)) : (&dAv(i + 1, i)),
                        is_lower ? upperinc : lowerinc);
          }

          cblas_dtrsm(layout, 'L', uplo, uplo_to_normal(uplo), 'N', N, N, 1.0, A, lda, dA, lda);
          cblas_dtrsm(layout, 'R', uplo, uplo_to_trans(uplo), 'N', N, N, 1.0, A, lda, dA, lda);
          cblas_dscal(N, 0.5, dA, lda+1);
        
          assert(foundCalls.size() >= 9);
          assert(foundCalls[21].type == CallType::COPY);
          double *tmp = (double *)foundCalls[21].pout_arg1;
          inputs[4] = BlasInfo(tmp, N, 1);

          cblas_dcopy(N, dA, lda+1, tmp, 1);
          cblas_dlascl(layout, flip_uplo(uplo), 0, 0, 1.0, 0.0, N, N, dA, lda, 0);
          cblas_dcopy(N, tmp, 1, dA, lda+1);
          cblas_dtrmm(layout, uplo_to_side(uplo), uplo, 'N', 'N', N, N, 1.0, A, lda, dA, lda);
         
          cblas_dcopy(N, dA, lda+1, tmp, 1);
          cblas_dlacpy(layout, flip_uplo(uplo), N, N, tri, N, dA, lda);
          cblas_dcopy(N, tmp, 1, dA, lda+1);

          checkTest(Test);

          SkipVecIncCheck = true;
          // Check memory of primal of expected derivative
          checkMemoryTrace(inputs, "Expected " + Test, calls);

          // Check memory of primal of our derivative (if equal above, it
          // should be the same).
          checkMemoryTrace(inputs, "Found " + Test, foundCalls);
          SkipVecIncCheck = false;
        }

      }
  }
}

int main() {
  dotTests();

  nrm2Tests();

  gemvTests();

  gemmTests();
  
  syrkTests();

  potrfTests();

  symvTests();
}
