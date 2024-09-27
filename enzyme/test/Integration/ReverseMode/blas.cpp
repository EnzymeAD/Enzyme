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

int enzyme_dup;
int enzyme_out;
int enzyme_const;
template<typename ...T>
void __enzyme_autodiff(void*, T...);

void my_dgemv(char layout, char trans, int M, int N, double alpha, double* __restrict__ A, int lda, double* __restrict__ X, int incx, double beta, double* __restrict__ Y, int incy) {
    cblas_dgemv(layout, trans, M, N, alpha, A, lda, X, incx, beta, Y, incy);
    inDerivative = true;
}

void ow_dgemv(char layout, char trans, int M, int N, double alpha, double* A, int lda, double* X, int incx, double beta, double* Y, int incy) {
    cblas_dgemv(layout, trans, M, N, alpha, A, lda, X, incx, beta, Y, incy);
    inDerivative = true;
}


void my_dsymv(char layout, char uplo, int N, double alpha, double* __restrict__ A, int lda, double* __restrict__ X, int incx, double beta, double* __restrict__ Y, int incy) {
    cblas_dsymv(layout, uplo, N, alpha, A, lda, X, incx, beta, Y, incy);
    inDerivative = true;
}

void ow_dsymv(char layout, char uplo, int N, double alpha, double* A, int lda, double* X, int incx, double beta, double* Y, int incy) {
    cblas_dsymv(layout, uplo, N, alpha, A, lda, X, incx, beta, Y, incy);
    inDerivative = true;
}


double my_ddot(int N, double* __restrict__ X, int incx, double* __restrict__ Y, int incy) {
    double res = cblas_ddot(N, X, incx, Y, incy);
    inDerivative = true;
    return res;
}

double my_dnrm2(int N, double *__restrict__ X, int incx) {
  double res = cblas_dnrm2(N, X, incx);
  inDerivative = true;
  return res;
}

void my_dgemm(char layout, char transA, char transB, int M, int N, int K, double alpha, double* __restrict__ A, int lda, double* __restrict__ B, int ldb, double beta, double* __restrict__ C, int ldc) {
    cblas_dgemm(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    inDerivative = true;
}

void ow_dgemm(char layout, char transA, char transB, int M, int N, int K, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc) {
    cblas_dgemm(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    inDerivative = true;
}

void my_dtrmv(char layout, char uplo, char trans,
               char diag, int N, double * __restrict__ A, int lda,
               double *__restrict__ X, int incx) {
    cblas_dtrmv(layout, uplo, trans, diag, N, A, lda, X, incx);
    inDerivative = true;
}

void my_dtrmm(char layout, char side, char uplo,
                                           char trans, char diag, int M, int N,
                                           double alpha, double * __restrict__ A, int lda,
                                           double *__restrict B, int ldb) {
    cblas_dtrmm(layout, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
    inDerivative = true;
}

void ow_dtrmm(char layout, char side, char uplo,
                                           char trans, char diag, int M, int N,
                                           double alpha, double * A, int lda,
                                           double * B, int ldb) {
    cblas_dtrmm(layout, side, uplo, trans, diag, M, N, alpha, A, lda, B, ldb);
    inDerivative = true;
}

void my_dsyrk(char layout, char uplo, char trans,
                                           int N, int K, double alpha,
                                           double *__restrict__ A, int lda, double beta,
                                           double *__restrict__ C, int ldc) {

    cblas_dsyrk(layout, uplo, trans, N, K, alpha, A, lda, beta,
                                           C, ldc);
    inDerivative = true;
}

void my_potrf(char layout, char uplo, int N, double *__restrict__ A, int lda) {
  int info;
  cblas_dpotrf(layout, uplo, N, A, lda, &info);
  inDerivative = true;
}
void ow_potrf(char layout, char uplo, int N, double *__restrict__ A, int lda) {
  int info;
  cblas_dpotrf(layout, uplo, N, A, lda, &info);
  cblas_dscal(1, 0.0, A, lda);
  inDerivative = true;
}

void my_potrs(char layout, char uplo, int N, int Nrhs, double *__restrict__ A, int lda, double *__restrict__ B, int ldb) {
  int info;
  cblas_dpotrs(layout, uplo, N, Nrhs, A, lda, B, ldb, &info);
  inDerivative = true;
}

void my_trtrs(char layout, char uplo, char trans, char diag, int N, int Nrhs,
              double *__restrict__ A, int lda, double *__restrict__ B,
              int ldb) {
  int info;
  cblas_dtrtrs(layout, uplo, trans, diag, N, Nrhs, A, lda, B, ldb, &info);
  inDerivative = true;
}
void ow_trtrs(char layout, char uplo, char trans, char diag, int N, int Nrhs,
              double *A, int lda, double *B, int ldb) {
  int info;
  cblas_dtrtrs(layout, uplo, trans, diag, N, Nrhs, A, lda, B, ldb, &info);
  cblas_dscal(1, 0.0, A, lda);
  inDerivative = true;
}

void my_symm(char layout, char side, char uplo,
                                           int M, int N, double alpha,
                                           double * __restrict__ A, int lda, double * __restrict__ B,
                                           int ldb, double beta, double * __restrict__ C,
                                           int ldc) {
  cblas_dsymm(layout, side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
  inDerivative = true;
}

void ow_symm(char layout, char side, char uplo,
                                           int M, int N, double alpha,
                                           double * A, int lda, double * B,
                                           int ldb, double beta, double * C,
                                           int ldc) {
  cblas_dsymm(layout, side, uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc);
  inDerivative = true;
}

static void dotTests() {

    std::string Test = "DOT active both ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, N, incA),
        /*B*/ BlasInfo(B, N, incB),
        /*C*/ BlasInfo(C, M, incC),
		BlasInfo(),
		BlasInfo(),
		BlasInfo(),
    };
    init();
    my_ddot(N, A, incA, B, incB);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    __enzyme_autodiff((void*) my_ddot,
                            enzyme_const, N,
                            enzyme_dup, A, dA,
                            enzyme_const, incA,
                            enzyme_dup, B, dB,
                            enzyme_const, incB);
        foundCalls = calls;
        init();

        my_ddot(N, A, incA, B, incB);

        inDerivative = true;

        cblas_daxpy(N, 1.0, B, incB, dA, incA);
        cblas_daxpy(N, 1.0, A, incA, dB, incB);

	checkTest(Test);
    
        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);
        
        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
}

static void nrm2Tests() {

  std::string Test = "DNRM2 active both ";
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
  __enzyme_autodiff((void *)my_dnrm2, enzyme_const, N, enzyme_dup, A, dA,
                    enzyme_const, incA);
  foundCalls = calls;
  init();

  my_dnrm2(N, A, incA);

  inDerivative = true;

  double tmp = cblas_dnrm2(N, A, incA);
  cblas_daxpy(N, 1.0 / tmp, A, incA, dA, incA);

  checkTest(Test);

  // Check memory of primal of expected derivative
  checkMemoryTrace(inputs, "Expected " + Test, calls);

  // Check memory of primal of our derivative (if equal above, it
  // should be the same).
  checkMemoryTrace(inputs, "Found " + Test, foundCalls);
}

static void gemvTests() {
  // N means normal matrix, T means transposed
  for (char layout : { CblasRowMajor, CblasColMajor }) {
  for (auto transA : {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans}) {
      // todo in fortran blas consider 'N', 'n', 'T', 't'}
  
    {

        bool trans = !is_normal(transA);
        std::string Test = "GEMV active A, C ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, layout, M, N, lda),
        /*B*/ BlasInfo(B, trans ? M : N, incB),
        /*C*/ BlasInfo(C, trans ? N : M, incC),
		BlasInfo(),
		BlasInfo(),
		BlasInfo()
    };
    init();
    my_dgemv(layout, (char)transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

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
    __enzyme_autodiff((void*) my_dgemv,
                            enzyme_const, layout,
                            enzyme_const, transA,
                            enzyme_const, M,
                            enzyme_const, N,
                            enzyme_const, alpha,
                            enzyme_dup, A, dA,
                            enzyme_const, lda,
                            enzyme_const, B,
                            enzyme_const, incB,
                            enzyme_const, beta,
                            enzyme_dup, C, dC,
                            enzyme_const, incC);
        foundCalls = calls;
        init();

        my_dgemv(layout, (char)transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

        inDerivative = true;
        // dC = alpha * X * transpose(Y) + A
        cblas_dger(layout, M, N, alpha, trans ? B : dC, trans ? incB : incC, trans ? dC : B, trans ? incC : incB, dA, lda);
        // dY = beta * dY
        cblas_dscal(trans ? N : M, beta, dC, incC);

		checkTest(Test);
    
        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);
        
        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        
        Test = "GEMV active A, B, C ";
    
        init();
        __enzyme_autodiff((void*) my_dgemv,
                                enzyme_const, layout,
                                enzyme_const, transA,
                                enzyme_const, M,
                                enzyme_const, N,
                                enzyme_const, alpha,
                                enzyme_dup, A, dA,
                                enzyme_const, lda,
                                enzyme_dup, B, dB,
                                enzyme_const, incB,
                                enzyme_const, beta,
                                enzyme_dup, C, dC,
                                enzyme_const, incC);
            foundCalls = calls;
            init();

            my_dgemv(layout, (char)transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

            inDerivative = true;
            // dC = alpha * X * transpose(Y) + A
            cblas_dger(layout, M, N, alpha, trans ? B : dC, trans ? incB : incC, trans ? dC : B, trans ? incC : incB, dA, lda);

            // dB = alpha * trans(A) * dC + dB
            cblas_dgemv(layout, (char)transpose(transA), M, N, alpha, A, lda, dC, incC, 1.0, dB, incB); 

            // dY = beta * dY
            cblas_dscal(trans ? N : M, beta, dC, incC);

            checkTest(Test);
        
            // Check memory of primal of expected derivative
            checkMemoryTrace(inputs, "Expected " + Test, calls);
            
            // Check memory of primal of our derivative (if equal above, it
            // should be the same).
            checkMemoryTrace(inputs, "Found " + Test, foundCalls);



        Test = "GEMV active/overwrite";
    
        init();
        __enzyme_autodiff((void*) ow_dgemv,
                                enzyme_const, layout,
                                enzyme_const, transA,
                                enzyme_const, M,
                                enzyme_const, N,
                                enzyme_const, alpha,
                                enzyme_dup, A, dA,
                                enzyme_const, lda,
                                enzyme_dup, B, dB,
                                enzyme_const, incB,
                                enzyme_const, beta,
                                enzyme_dup, C, dC,
                                enzyme_const, incC);
            foundCalls = calls;
            init();

			assert(foundCalls.size() > 2);
			auto A_cache = (double*)foundCalls[0].pout_arg1;
			cblas_dlacpy(layout, '\0', M, N, A, lda, A_cache, M);
			inputs[4] = BlasInfo(A_cache, layout, M, N, M);
			auto B_cache = (double*)foundCalls[1].pout_arg1;
			cblas_dcopy(trans ? M : N, B, incB, B_cache, 1);
			inputs[5] = BlasInfo(B_cache, trans ? M : N, 1);

            ow_dgemv(layout, (char)transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

            inDerivative = true;
            // dC = alpha * X * transpose(Y) + A
            cblas_dger(layout, M, N, alpha,
							trans ? B_cache : dC, 
							trans ? 1 : incC, 
							trans ? dC : B_cache,
							trans ? incC : 1, dA,
							lda);

            // dB = alpha * trans(A) * dC + dB
            cblas_dgemv(layout, (char)transpose(transA), M, N, alpha, A_cache, M, dC, incC, 1.0, dB, incB); 

            // dY = beta * dY
            cblas_dscal(trans ? N : M, beta, dC, incC);

            checkTest(Test);
        
            // Check memory of primal of expected derivative
            checkMemoryTrace(inputs, "Expected " + Test, calls);
            
            // Check memory of primal of our derivative (if equal above, it
            // should be the same).
            checkMemoryTrace(inputs, "Found " + Test, foundCalls);

			inputs[4] = BlasInfo();
			inputs[5] = BlasInfo();
    }


  }
  }
}


static void symvTests() {
  int N = 17;
  // N means normal matrix, T means transposed
  for (char layout : { CblasRowMajor, CblasColMajor }) {
  for (auto uplo : {'U', 'u', 'L', 'l'})
    {

        std::string Test = "SYMV active A, C ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, layout, N, N, lda),
        /*B*/ BlasInfo(B, N, incB),
        /*C*/ BlasInfo(C, N, incC),
        BlasInfo(),
        BlasInfo(),
        BlasInfo()
    };

    {
    init();
    my_dsymv(layout, uplo, N, alpha, A, lda, B, incB, beta, C, incC);

    assert(calls.size() == 1);
    assert(calls[0].inDerivative == false);
    assert(calls[0].type == CallType::SYMV);
    assert(calls[0].pout_arg1 == C);
    assert(calls[0].pin_arg1 == A);
    assert(calls[0].pin_arg2 == B);
    assert(calls[0].farg1 == alpha);
    assert(calls[0].farg2 == beta);
    assert(calls[0].layout == layout);
    assert(calls[0].targ1 == UNUSED_TRANS);
    assert(calls[0].targ2 == UNUSED_TRANS);
    assert(calls[0].iarg1 == N);
    assert(calls[0].iarg3 == UNUSED_INT);
    assert(calls[0].iarg4 == lda);
    assert(calls[0].iarg5 == incB);
    assert(calls[0].iarg6 == incC);
    assert(calls[0].uplo == uplo);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    __enzyme_autodiff((void*) my_dsymv,
                            enzyme_const, layout,
                            enzyme_const, uplo,
                            enzyme_const, N,
                            enzyme_const, alpha,
                            enzyme_dup, A, dA,
                            enzyme_const, lda,
                            enzyme_const, B,
                            enzyme_const, incB,
                            enzyme_const, beta,
                            enzyme_dup, C, dC,
                            enzyme_const, incC);
        foundCalls = calls;
        init();

        my_dsymv(layout, uplo, N, alpha, A, lda, B, incB, beta, C, incC);

        inDerivative = true;

        double *tmp = (double *)foundCalls[1].pout_arg1;
        inputs[4] = BlasInfo(tmp, N, 1);
        cblas_dcopy(N, dA, lda + 1, tmp, 1);
        cblas_dsyr2(layout, uplo, N, alpha, B, incB, dC, incC, dA, lda);
        cblas_dcopy(N, tmp, 1, dA, lda + 1);

        // dY = beta * dY
        cblas_dscal(N, beta, dC, incC);

        checkTest(Test);
    
        SkipVecIncCheck = true;
        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);
        
        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        SkipVecIncCheck = false;
    }
   
        {
        Test = "SYMV active A, B, C ";
    
        init();
        __enzyme_autodiff((void*) my_dsymv,
                                enzyme_const, layout,
                                enzyme_const, uplo,
                                enzyme_const, N,
                                enzyme_const, alpha,
                                enzyme_dup, A, dA,
                                enzyme_const, lda,
                                enzyme_dup, B, dB,
                                enzyme_const, incB,
                                enzyme_const, beta,
                                enzyme_dup, C, dC,
                                enzyme_const, incC);
            foundCalls = calls;
            init();

            my_dsymv(layout, uplo, N, alpha, A, lda, B, incB, beta, C, incC);
    
            inDerivative = true;

            double *tmp = (double *)foundCalls[1].pout_arg1;
            inputs[4] = BlasInfo(tmp, N, 1);
            cblas_dcopy(N, dA, lda + 1, tmp, 1);
            cblas_dsyr2(layout, uplo, N, alpha, B, incB, dC, incC, dA, lda);
            cblas_dcopy(N, tmp, 1, dA, lda + 1);

            cblas_dsymv(layout, uplo, N, alpha, A, lda, dC, incC, 1.0, dB, incB);
    
            // dY = beta * dY
            cblas_dscal(N, beta, dC, incC);


            checkTest(Test);
        
            SkipVecIncCheck = true;
            // Check memory of primal of expected derivative
            checkMemoryTrace(inputs, "Expected " + Test, calls);
            
            // Check memory of primal of our derivative (if equal above, it
            // should be the same).
            checkMemoryTrace(inputs, "Found " + Test, foundCalls);
            SkipVecIncCheck = false;
        }

    {

        Test = "SYMV active/overwrite";
    
        init();
        __enzyme_autodiff((void*) ow_dsymv,
                                enzyme_const, layout,
                                enzyme_const, uplo,
                                enzyme_const, N,
                                enzyme_const, alpha,
                                enzyme_dup, A, dA,
                                enzyme_const, lda,
                                enzyme_dup, B, dB,
                                enzyme_const, incB,
                                enzyme_const, beta,
                                enzyme_dup, C, dC,
                                enzyme_const, incC);
            foundCalls = calls;
            init();

            assert(foundCalls.size() > 2);
            auto A_cache = (double*)foundCalls[0].pout_arg1;
            cblas_dlacpy(layout, uplo, N, N, A, lda, A_cache, N);
            inputs[4] = BlasInfo(A_cache, layout, N, N, N);
            auto B_cache = (double*)foundCalls[1].pout_arg1;
            cblas_dcopy(N, B, incB, B_cache, 1);
            inputs[5] = BlasInfo(B_cache, N, 1);

            ow_dsymv(layout, uplo, N, alpha, A, lda, B, incB, beta, C, incC);

            inDerivative = true;

            double *tmp = (double *)foundCalls[3].pout_arg1;
            inputs[3] = BlasInfo(tmp, N, 1);
            cblas_dcopy(N, dA, lda + 1, tmp, 1);
            cblas_dsyr2(layout, uplo, N, alpha, B_cache, 1, dC, incC, dA, lda);
            cblas_dcopy(N, tmp, 1, dA, lda + 1);

            cblas_dsymv(layout, uplo, N, alpha, A_cache, N, dC, incC, 1.0, dB, incB);
    
            // dY = beta * dY
            cblas_dscal(N, beta, dC, incC);


            checkTest(Test);
        
            SkipVecIncCheck = true;
            // Check memory of primal of expected derivative
            checkMemoryTrace(inputs, "Expected " + Test, calls);
            
            // Check memory of primal of our derivative (if equal above, it
            // should be the same).
            checkMemoryTrace(inputs, "Found " + Test, foundCalls);
            SkipVecIncCheck = false;

            inputs[4] = BlasInfo();
            inputs[5] = BlasInfo();
    }


  }
  }
}

static void gemmTests() {
  // N means normal matrix, T means transposed
  for (char layout : { CblasRowMajor, CblasColMajor }) {
  for (auto transA : {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans}) {
  for (auto transB : {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans}) {
    // todo fortran blas {'N', 'n', 'T', 't'}
  
    {

        bool transA_bool = !is_normal(transA);
        bool transB_bool = !is_normal(transB);
        std::string Test = "GEMM";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, layout, transA_bool ? K : M, transA_bool ? M : K, lda),
        /*B*/ BlasInfo(B, layout, transB_bool ? N : K , transB_bool ? K : N, incB),
        /*C*/ BlasInfo(C, layout, M, N, incC),
		BlasInfo(),
		BlasInfo(),
		BlasInfo()
    };
    init();
    my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda, B, incB, beta, C, incC);

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
    __enzyme_autodiff((void*) my_dgemm,
                            enzyme_const, layout,
                            enzyme_const, transA,
                            enzyme_const, transB,
                            enzyme_const, M,
                            enzyme_const, N,
                            enzyme_const, K,
                            enzyme_const, alpha,
                            enzyme_dup, A, dA,
                            enzyme_const, lda,
                            enzyme_dup, B, dB,
                            enzyme_const, incB,
                            enzyme_const, beta,
                            enzyme_dup, C, dC,
                            enzyme_const, incC);
        foundCalls = calls;
        init();

    
        my_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda, B, incB, beta, C, incC);

        inDerivative = true;

        // dA = 
        my_dgemm(layout,
                    transA_bool ? (char)transB : (char)CBLAS_TRANSPOSE::CblasNoTrans,
                    transA_bool ? (char)CBLAS_TRANSPOSE::CblasTrans : (char)transpose(transB),
                    transA_bool ? K : M,
                    transA_bool ? M : K,
                    N,
                    alpha,
                    transA_bool ? B : dC,
                    transA_bool ? incB : incC,
                    transA_bool ? dC : B,
                    transA_bool ? incC : incB,
                    1.0, dA, lda);
        
        // dB = 
        my_dgemm(layout,
                    transB_bool ? (char)CBLAS_TRANSPOSE::CblasTrans : (char)transpose(transA),
                    transB_bool ? (char)transA : (char)CBLAS_TRANSPOSE::CblasNoTrans, //transB,
                    transB_bool ? N : K,
                    transB_bool ? K : N,
                    M,
                    alpha,
                    transB_bool ? dC : A,
                    transB_bool ? incC : lda,
                    transB_bool ? A : dC,
                    transB_bool ? lda : incC,
                    1.0, dB, incB);
       
        cblas_dlascl(layout, 'G', 0, 0, 1.0, beta, M, N, dC, incC, 0 );
		
        checkTest(Test);
    
        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);
        
        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        

        Test = "GEMM overwrite";

    init();
    __enzyme_autodiff((void*) ow_dgemm,
                            enzyme_const, layout,
                            enzyme_const, transA,
                            enzyme_const, transB,
                            enzyme_const, M,
                            enzyme_const, N,
                            enzyme_const, K,
                            enzyme_const, alpha,
                            enzyme_dup, A, dA,
                            enzyme_const, lda,
                            enzyme_dup, B, dB,
                            enzyme_const, incB,
                            enzyme_const, beta,
                            enzyme_dup, C, dC,
                            enzyme_const, incC);
        foundCalls = calls;
        init();

			assert(foundCalls.size() > 2);
			auto A_cache = (double*)foundCalls[0].pout_arg1;
			cblas_dlacpy(layout, '\0', (!transA_bool) ? M : K, (!transA_bool) ? K : M, A, lda, A_cache, (!transA_bool) ? M : K);
			inputs[4] = BlasInfo(A_cache, layout, (!transA_bool) ? M : K, (!transA_bool) ? K : M, (!transA_bool) ? M : K);
			auto B_cache = (double*)foundCalls[1].pout_arg1;
			cblas_dlacpy(layout, '\0', (!transB_bool) ? K : N, (!transB_bool) ? N : K, B, incB, B_cache, (!transB_bool) ? K : N);
			inputs[5] = BlasInfo(B_cache, layout, (!transB_bool) ? K : N, (!transB_bool) ? N : K, (!transB_bool) ? K : N);

        ow_dgemm(layout, (char)transA, (char)transB, M, N, K, alpha, A, lda, B, incB, beta, C, incC);

        inDerivative = true;

        // dA = 
        my_dgemm(layout,
                    transA_bool ? (char)transB : (char)CBLAS_TRANSPOSE::CblasNoTrans,
                    transA_bool ? (char)CBLAS_TRANSPOSE::CblasTrans : (char)transpose(transB),
                    transA_bool ? K : M,
                    transA_bool ? M : K,
                    N,
                    alpha,
                    transA_bool ? B_cache : dC,
                    transA_bool ? ( (!transB_bool) ? K : N )  : incC,
                    transA_bool ? dC : B_cache,
                    transA_bool ? incC : ( (!transB_bool) ? K : N),
                    1.0, dA, lda);
        
        // dB = 
        my_dgemm(layout,
                    transB_bool ? (char)CBLAS_TRANSPOSE::CblasTrans : (char)transpose(transA),
                    transB_bool ? (char)transA : (char)CBLAS_TRANSPOSE::CblasNoTrans, //transB,
                    transB_bool ? N : K,
                    transB_bool ? K : N,
                    M,
                    alpha,
                    transB_bool ? dC : A_cache,
                    transB_bool ? incC : ( (!transA_bool) ? M : K),
                    transB_bool ? A_cache : dC,
                    transB_bool ? ( (!transA_bool) ? M : K) : incC,
                    1.0, dB, incB);
       
        cblas_dlascl(layout, 'G', 0, 0, 1.0, beta, M, N, dC, incC, 0 );
		
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

static void trmvTests() {
  REALCOPY = true;
  // N means normal matrix, T means transposed
  int N = 7;
  double* B = (double*)malloc(sizeof(double*)*incB*N);
  double* dB = (double*)malloc(sizeof(double*)*incB*N);
  // TODO row major
  for (char layout : { CblasColMajor, /*CblasRowMajor */}) {
  
  for (auto uplo : {'U', 'u', 'L', 'l'})
  
  for (auto diag : {'U', 'u', 'N', 'n'})
  
  for (auto transA : {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans})

  {
      // todo in fortran blas consider 'N', 'n', 'T', 't'}
  
    {

        bool trans = !is_normal(transA);
        std::string Test = "TRMV active A, C ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, layout, N, N, lda),
        /*B*/ /*BlasInfo(B, N, incB),*/BlasInfo(),
		BlasInfo(),
		BlasInfo(),
        BlasInfo(B, N, incB),
        BlasInfo(dB, N, incB)
    };
    init();

  for (int i=0; i<N*incB; i++) {
        B[i] = i *1e-4;
        dB[i] = -i *1e-4;
    }
  for (size_t i=0; i<N; i++) {
    B[incB*i] = 7 + i;
    dB[incB*i] = 300 + i;
  }
    my_dtrmv(layout, uplo, (char)transA, diag, N, A, lda, B, incB);

    assert(calls.size() == 1);
    assert(calls[0].inDerivative == false);
    assert(calls[0].type == CallType::TRMV);
    assert(calls[0].pout_arg1 == B);
    assert(calls[0].pin_arg1 == A);
    assert(calls[0].pin_arg2 == UNUSED_POINTER);
    assert(calls[0].farg1 == UNUSED_DOUBLE);
    assert(calls[0].farg2 == UNUSED_DOUBLE);
    assert(calls[0].layout == layout);
    assert(calls[0].targ1 == (char)transA);
    assert(calls[0].targ2 == UNUSED_TRANS);
    assert(calls[0].iarg1 == N);
    assert(calls[0].iarg2 == UNUSED_INT);
    assert(calls[0].iarg3 == UNUSED_INT);
    assert(calls[0].iarg4 == lda);
    assert(calls[0].iarg5 == incB);
    assert(calls[0].iarg6 == UNUSED_INT);
    assert(calls[0].uplo == uplo);
    assert(calls[0].diag == diag);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
  for (int i=0; i<N*incB; i++) {
        B[i] = i *1e-4;
        dB[i] = -i *1e-4;
    }
  for (size_t i=0; i<N; i++) {
    B[incB*i] = 7 + i;
    dB[incB*i] = 300 + i;
  }
    __enzyme_autodiff((void*) my_dtrmv,
                            enzyme_const, layout,
                            enzyme_const, uplo,
                            enzyme_const, transA,
                            enzyme_const, diag,
                            enzyme_const, N,
                            enzyme_dup, A, dA,
                            enzyme_const, lda,
                            enzyme_dup, B, dB,
                            enzyme_const, incB);
        foundCalls = calls;
        init();

  for (int i=0; i<N*incB; i++) {
        B[i] = i *1e-4;
        dB[i] = -i *1e-4;
    }
  for (size_t i=0; i<N; i++) {
    B[incB*i] = 7 + i;
    dB[incB*i] = 300 + i;
  }
        assert(foundCalls.size() >= 2);
        assert(foundCalls[0].type == CallType::COPY);
        double* cacheB = (double*)foundCalls[0].pout_arg1;
			
        cblas_dcopy(N, B, incB, cacheB, 1);
		inputs[3] = BlasInfo(cacheB, N, 1);
        auto B0 = cacheB;

        my_dtrmv(layout, uplo, (char)transA, diag, N, A, lda, B, incB);

        inDerivative = true;

        auto d = (diag == 'n' || diag == 'N') ? 0 : 1;

        #define Aa(r,c) dA[(r-1)*(layout == CblasRowMajor ? lda : 1)  + (c-1)*(layout == CblasRowMajor ? 1 : lda) ]

        if (is_normal(transA)) {
          if (uplo == 'u' || uplo == 'U') {
            for (int i=1; i<=N; i++) {
              cblas_daxpy(i-d, B0[i-1], dB, incB, &Aa(1, i), 1);
            }
          } else {
            // A is lower triangular
            for (int i=1; i<=N-d; i++)
              cblas_daxpy(N-i+1-d, B0[i-1], &dB[(i+d-1)*incB], incB, &Aa(i+d,i), 1);
          }
        } else {
          // BLAS operation
          //   x := A'*x where A is triangular
          // RMD operation
          //   Aa += x*xa'
          if( uplo == 'u' || uplo == 'U') {
            // A is upper triangular
            for (int i=1; i<=N; i++)
              cblas_daxpy(i-d, dB[(i-1)*incB], B0, 1, &Aa(1, i), 1);
          } else {
            // A is lower triangular
            for (int i=1; i<=N-d; i++)
              cblas_daxpy(N-i+1-d, dB[(i-1)*incB], &B0[i+d-1], 1, &Aa(i+d,i), 1);
          }
        }

        cblas_dtrmv(layout, uplo, (char)transpose(transA), diag, N, A, lda, dB, incB);

		checkTest(Test);
    
        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);
        
        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        
    }


  }
  }
  REALCOPY = false;
}

static void trmmTests() {
  // N means normal matrix, T means transposed
  // TODO: row major is presently an exepcted failure. We should re-enable.
  for (char layout : { CblasColMajor, /*CblasRowMajor*/ }) {
  
  for (auto side : {'L', 'l', 'R', 'r'})
  
  for (auto transA : {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans})
  
  for (auto uplo : {'U', 'u', 'L', 'l'})
  
  for (auto diag : {'U', 'u', 'N', 'n'})

  {
      // todo in fortran blas consider 'N', 'n', 'T', 't'}
  
    int N = 7;
    int M = 13;
    {

        bool trans = !is_normal(transA);
        std::string Test = "TRMM active A, B ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, layout, (side == 'L' || side == 'l') ? M : N, (side == 'L' || side == 'l') ? M : N, lda),
        /*B*/ BlasInfo(B, layout, M, N, incB),
		BlasInfo(),
		BlasInfo(),
		BlasInfo(),
		BlasInfo()
    };
    init();

    my_dtrmm(layout, side, uplo, (char)transA, diag, M, N, alpha, A, lda, B, incB);

    assert(calls.size() == 1);
    assert(calls[0].inDerivative == false);
    assert(calls[0].type == CallType::TRMM);
    assert(calls[0].pout_arg1 == B);
    assert(calls[0].pin_arg1 == A);
    assert(calls[0].pin_arg2 == UNUSED_POINTER);
    assert(calls[0].farg1 == alpha);
    assert(calls[0].farg2 == UNUSED_DOUBLE);
    assert(calls[0].layout == layout);
    assert(calls[0].targ1 == (char)transA);
    assert(calls[0].targ2 == UNUSED_TRANS);
    assert(calls[0].iarg1 == M);
    assert(calls[0].iarg2 == N);
    assert(calls[0].iarg3 == UNUSED_INT);
    assert(calls[0].iarg4 == lda);
    assert(calls[0].iarg5 == incB);
    assert(calls[0].iarg6 == UNUSED_INT);
    assert(calls[0].side == side);
    assert(calls[0].uplo == uplo);
    assert(calls[0].diag == diag);

    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    __enzyme_autodiff((void*) my_dtrmm,
                            enzyme_const, layout,
                            enzyme_const, side,
                            enzyme_const, uplo,
                            enzyme_const, transA,
                            enzyme_const, diag,
                            enzyme_const, M,
                            enzyme_const, N,
                            enzyme_const, alpha,
                            enzyme_dup, A, dA,
                            enzyme_const, lda,
                            enzyme_dup, B, dB,
                            enzyme_const, incB);
        foundCalls = calls;
        init();


        double* cacheB = (double*)foundCalls[0].pout_arg1;
			
		cblas_dlacpy(layout, '\0', M, N,
                B,
                incB, cacheB, M);
		inputs[4] = BlasInfo(cacheB, layout, M, N, M);
        my_dtrmm(layout, side, uplo, (char)transA, diag, M, N, alpha, A, lda, B, incB);

        assert(foundCalls.size() >= 2);
        assert(foundCalls[0].type == CallType::LACPY);
        inDerivative = true;

        auto d = (diag == 'n' || diag == 'N') ? 0 : 1;

    #define B0(r,c) cacheB[(r-1)*(layout == CblasRowMajor ? M : 1) + (c-1)*(layout == CblasRowMajor ? 1 : M) ]
    #define Ba(r,c) dB[(r-1)*(layout == CblasRowMajor ? incB : 1)  + (c-1)*(layout == CblasRowMajor ? 1 : incB) ]
    #define Aa(r,c) dA[(r-1)*(layout == CblasRowMajor ? lda : 1)  + (c-1)*(layout == CblasRowMajor ? 1 : lda) ]

    auto ldb = incB;

    char toTrans;
    if (side == 'l')
        toTrans = 'n';
    else if (side == 'L')
        toTrans = 'N';
    else if (side == 'r')
        toTrans = 't';
    else if (side == 'R')
        toTrans = 'T';

    if (side == 'l' || side == 'L') {
      if (is_normal(transA)) {
        // BLAS operation
        // B = alpha*A*B0
        // RMD operation
        // Aa += alpha*Ba*B0'
        if(uplo == 'u' || uplo == 'U') {
          // A is upper triangular
          for (int i=1; i<=M; i++)
            cblas_dgemv(layout, toTrans,i-d,N, alpha,dB,incB,&B0(i, 1),M,1.0,&Aa(1, i),1);
        } else {
          // A is lower triangular
          for (int i=1; i<=M-d; i++)
            cblas_dgemv(layout, toTrans,M-i+1-d,N,alpha,&Ba(i+d,1),ldb,&B0(i,1),M,1.0, &Aa(i+d,i),1);
        }
      } else {
        // BLAS operation
        // B = alpha*A'*B0
        // RMD operation
        // Aa += alpha*B*Ba'
        if(uplo == 'u' || uplo == 'U') {
          // A is upper triangular
          for (int i=1; i<=M; i++)
            cblas_dgemv(layout, toTrans,i-d,N, alpha,&B0(1,1),M,&Ba(i,1),ldb,1.0,&Aa(1,i),1);
        } else {
          // A is lower triangular
          for (int i=1; i<=M-d; i++)
            cblas_dgemv(layout, toTrans,M-i+1-d,N,alpha,&B0(i+d,1),M,&Ba(i,1),ldb,1.0, &Aa(i+d,i),1);
        }
      }
    } else {
      if (is_normal(transA)) {
        // BLAS operation
        // B = alpha*B0*A
        // RMD operation
        // Aa += alpha*B0'*Ba
        if(uplo == 'u' || uplo == 'U') {
          // A is upper triangular
          for (int i=1; i<=N; i++)
            cblas_dgemv(layout, toTrans,M,i-d,alpha,&B0(1,1),M,&Ba(1,i),1, 1.0,&Aa(1,i),1);
        } else {
          // A is lower triangular
          for (int i=1; i<=N-d; i++)
            cblas_dgemv(layout, toTrans,M,N-i+1-d,alpha,&B0(1,i+d),M,&Ba(1,i),1, 1.0, &
                Aa(i+d,i),1);
        }
      } else {
        // BLAS operation
        // B = alpha*B0*A'
        // RMD operation
        // Aa += alpha*Ba'*B0
        if(uplo == 'u' || uplo == 'U') {
          // A is upper triangular
          for (int i=1; i<=N; i++)
            cblas_dgemv(layout, toTrans,M,i-d,alpha,&Ba(1,1),ldb,&B0(1,i),1, 1.0,&Aa(1,i),1);
        } else {
          // A is lower triangular
          for (int i=1; i<=N-d; i++)
            cblas_dgemv(layout, toTrans,M,N-i+1-d,alpha,&Ba(1,i+d),ldb,&B0(1,i),1, 1.0, &Aa(i+d,i),1);
        }
      }
    }

        cblas_dtrmm(layout, side, uplo, (char)transpose(transA), diag, M, N, alpha, A, lda, dB, incB);

		checkTest(Test);
    
        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);
        
        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        
    }
    
    {

        bool trans = !is_normal(transA);
        std::string Test = "TRMM overwrite active A, B ";
    BlasInfo inputs[6] = {
        /*A*/ BlasInfo(A, layout, (side == 'L' || side == 'l') ? M : N, (side == 'L' || side == 'l') ? M : N, lda),
        /*B*/ BlasInfo(B, layout, M, N, incB),
		BlasInfo(),
		BlasInfo(),
		BlasInfo(),
		BlasInfo()
    };
    init();

    ow_dtrmm(layout, side, uplo, (char)transA, diag, M, N, alpha, A, lda, B, incB);
    
    // Check memory of primal on own.
    checkMemoryTrace(inputs, "Primal " + Test, calls);

    init();
    __enzyme_autodiff((void*) ow_dtrmm,
                            enzyme_const, layout,
                            enzyme_const, side,
                            enzyme_const, uplo,
                            enzyme_const, transA,
                            enzyme_const, diag,
                            enzyme_const, M,
                            enzyme_const, N,
                            enzyme_const, alpha,
                            enzyme_dup, A, dA,
                            enzyme_const, lda,
                            enzyme_dup, B, dB,
                            enzyme_const, incB);
        foundCalls = calls;
        init();

        
        double* cacheA = (double*)foundCalls[0].pout_arg1;
			
		cblas_dlacpy(layout, '\0', is_left(side) ? M : N, is_left(side) ? M : N,
                A,
                lda, cacheA, is_left(side) ? M : N);
		inputs[5] = BlasInfo(cacheA, layout, is_left(side) ? M : N, is_left(side) ? M : N, is_left(side) ? M : N);

        double* cacheB = (double*)foundCalls[1].pout_arg1;
			
		cblas_dlacpy(layout, '\0', M, N,
                B,
                incB, cacheB, M);
		inputs[4] = BlasInfo(cacheB, layout, M, N, M);

        ow_dtrmm(layout, side, uplo, (char)transA, diag, M, N, alpha, A, lda, B, incB);

        assert(foundCalls.size() >= 2);
        assert(foundCalls[0].type == CallType::LACPY);
        inDerivative = true;

        auto d = (diag == 'n' || diag == 'N') ? 0 : 1;

    #define B0(r,c) cacheB[(r-1)*(layout == CblasRowMajor ? M : 1) + (c-1)*(layout == CblasRowMajor ? 1 : M) ]
    #define Ba(r,c) dB[(r-1)*(layout == CblasRowMajor ? incB : 1)  + (c-1)*(layout == CblasRowMajor ? 1 : incB) ]
    #define Aa(r,c) dA[(r-1)*(layout == CblasRowMajor ? lda : 1)  + (c-1)*(layout == CblasRowMajor ? 1 : lda) ]

    auto ldb = incB;

    char toTrans;
    if (side == 'l')
        toTrans = 'n';
    else if (side == 'L')
        toTrans = 'N';
    else if (side == 'r')
        toTrans = 't';
    else if (side == 'R')
        toTrans = 'T';

    if (side == 'l' || side == 'L') {
      if (is_normal(transA)) {
        // BLAS operation
        // B = alpha*A*B0
        // RMD operation
        // Aa += alpha*Ba*B0'
        if(uplo == 'u' || uplo == 'U') {
          // A is upper triangular
          for (int i=1; i<=M; i++)
            cblas_dgemv(layout, toTrans,i-d,N, alpha,dB,incB,&B0(i, 1),M,1.0,&Aa(1, i),1);
        } else {
          // A is lower triangular
          for (int i=1; i<=M-d; i++)
            cblas_dgemv(layout, toTrans,M-i+1-d,N,alpha,&Ba(i+d,1),ldb,&B0(i,1),M,1.0, &Aa(i+d,i),1);
        }
      } else {
        // BLAS operation
        // B = alpha*A'*B0
        // RMD operation
        // Aa += alpha*B*Ba'
        if(uplo == 'u' || uplo == 'U') {
          // A is upper triangular
          for (int i=1; i<=M; i++)
            cblas_dgemv(layout, toTrans,i-d,N, alpha,&B0(1,1),M,&Ba(i,1),ldb,1.0,&Aa(1,i),1);
        } else {
          // A is lower triangular
          for (int i=1; i<=M-d; i++)
            cblas_dgemv(layout, toTrans,M-i+1-d,N,alpha,&B0(i+d,1),M,&Ba(i,1),ldb,1.0, &Aa(i+d,i),1);
        }
      }
    } else {
      if (is_normal(transA)) {
        // BLAS operation
        // B = alpha*B0*A
        // RMD operation
        // Aa += alpha*B0'*Ba
        if(uplo == 'u' || uplo == 'U') {
          // A is upper triangular
          for (int i=1; i<=N; i++)
            cblas_dgemv(layout, toTrans,M,i-d,alpha,&B0(1,1),M,&Ba(1,i),1, 1.0,&Aa(1,i),1);
        } else {
          // A is lower triangular
          for (int i=1; i<=N-d; i++)
            cblas_dgemv(layout, toTrans,M,N-i+1-d,alpha,&B0(1,i+d),M,&Ba(1,i),1, 1.0, &
                Aa(i+d,i),1);
        }
      } else {
        // BLAS operation
        // B = alpha*B0*A'
        // RMD operation
        // Aa += alpha*Ba'*B0
        if(uplo == 'u' || uplo == 'U') {
          // A is upper triangular
          for (int i=1; i<=N; i++)
            cblas_dgemv(layout, toTrans,M,i-d,alpha,&Ba(1,1),ldb,&B0(1,i),1, 1.0,&Aa(1,i),1);
        } else {
          // A is lower triangular
          for (int i=1; i<=N-d; i++)
            cblas_dgemv(layout, toTrans,M,N-i+1-d,alpha,&Ba(1,i+d),ldb,&B0(1,i),1, 1.0, &Aa(i+d,i),1);
        }
      }
    }

        cblas_dtrmm(layout, side, uplo, (char)transpose(transA), diag, M, N, alpha, cacheA, is_left(side) ? M : N, dB, incB);

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

static void syrkTests() {
  int N = 13;
  int K = 7;
  double *C = (double *)malloc(sizeof(double *) * incC * N * N);
  double *dC = (double *)malloc(sizeof(double *) * incC * N * N);
  // N means normal matrix, T means transposed
  // TODO: row major is presently an exepcted failure. We should re-enable.
  for (char layout : {CblasColMajor, /*CblasRowMajor*/}) {

    for (auto transA :
         {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans})

      for (auto uplo : {'U', 'u', 'L', 'l'})

      {

        {

          bool trans = !is_normal(transA);
          std::string Test = "SYRK active C, B ";
          BlasInfo inputs[6] = {
              /*A*/ BlasInfo(A, layout, trans ? K : N, trans ? N : K, lda),
              /*B*/ BlasInfo(),
              /*C*/ BlasInfo(),
              BlasInfo(),
              /*C*/ BlasInfo(C, layout, N, N, incC),
              /*C*/ BlasInfo(dC, layout, N, N, incC),
          };
          init();

          for (int i = 0; i < N * N * incC; i++) {
            C[i] = i * 1e-4;
            dC[i] = -i * 1e-4;
          }
          for (size_t i = 0; i < N * N; i++) {
            C[incC * i] = 7 + i;
            dC[incC * i] = 300 + i;
          }
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
          for (int i = 0; i < N * N * incC; i++) {
            C[i] = i * 1e-4;
            dC[i] = -i * 1e-4;
          }
          for (size_t i = 0; i < N * N; i++) {
            C[incC * i] = 7 + i;
            dC[incC * i] = 300 + i;
          }
          __enzyme_autodiff(
              (void *)my_dsyrk, enzyme_const, layout, enzyme_const, uplo,
              enzyme_const, transA, enzyme_const, N, enzyme_const, K,
              enzyme_const, alpha, enzyme_dup, A, dA, enzyme_const, lda,
              enzyme_const, beta, enzyme_dup, C, dC, enzyme_const, incC);
          foundCalls = calls;
          init();

          for (int i = 0; i < N * N * incC; i++) {
            C[i] = i * 1e-4;
            dC[i] = -i * 1e-4;
          }
          for (size_t i = 0; i < N * N; i++) {
            C[incC * i] = 7 + i;
            dC[incC * i] = 300 + i;
          }

          my_dsyrk(layout, uplo, (char)transA, N, K, alpha, A, lda, beta, C,
                   incC);

          inDerivative = true;

#define Av(r, c)                                                               \
  A[(r - 1) * (layout == CblasRowMajor ? lda : 1) +                            \
    (c - 1) * (layout == CblasRowMajor ? 1 : lda)]
#define Aa(r, c)                                                               \
  dA[(r - 1) * (layout == CblasRowMajor ? lda : 1) +                           \
     (c - 1) * (layout == CblasRowMajor ? 1 : lda)]

#define Ca(r, c)                                                               \
  dC[(r - 1) * (layout == CblasRowMajor ? incC : 1) +                          \
     (c - 1) * (layout == CblasRowMajor ? 1 : incC)]

          if (is_normal(transA)) {
            // BLAS operation
            //   C = alpha*A*A' + beta*C
            // RMD op
            //   Aa += alpha*(Ca+diag(Ca))*A
            cblas_dsymm(layout, 'l', uplo, N, K, alpha, dC, incC, A, lda, 1.0,
                        dA, lda);
            for (int i = 1; i <= N; i++)
              cblas_daxpy(K, alpha * Ca(i, i), &Av(i, 1), lda, &Aa(i, 1), lda);
          } else {
            // BLAS operation
            //   C = alpha*A'*A + beta*C
            // RMD operation
            //   Aa += alpha*A*(Ca+diag(Ca))
            cblas_dsymm(layout, 'r', uplo, K, N, alpha, dC, incC, A, lda, 1.0,
                        dA, lda);
            for (int i = 1; i <= N; i++)
              cblas_daxpy(K, alpha * Ca(i, i), &Av(1, i), 1, &Aa(1, i), 1);
          }
          cblas_dlascl(layout, uplo, 0, 0, 1.0, beta, N, N, dC, incC, 0);

          checkTest(Test);

          // Check memory of primal of expected derivative
          checkMemoryTrace(inputs, "Expected " + Test, calls);

          // Check memory of primal of our derivative (if equal above, it
          // should be the same).
          checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        }
      }
  }
  free(C);
  free(dC);
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
        __enzyme_autodiff((void *)my_potrf, enzyme_const, layout, enzyme_const,
                          uplo, enzyme_const, N, enzyme_dup, A, dA,
                          enzyme_const, lda);
        foundCalls = calls;
        init();

        my_potrf(layout, uplo, N, A, lda);

        inDerivative = true;

        assert(foundCalls.size() >= 2);
        assert(foundCalls[1].type == CallType::LACPY);
        double *tri = (double *)foundCalls[1].pout_arg1;
        inputs[3] = BlasInfo(tri, layout, N, N, N);

        cblas_dlacpy(layout, uplo, N, N, dA, lda, tri, N);

        cblas_dtrmm(layout, uplo_to_side(uplo), uplo, 'T', 'N', N, N, 1.0,
                    A, lda, tri, N);

        assert(foundCalls.size() >= 5);
        assert(foundCalls[3].type == CallType::COPY);
        double *tmp = (double *)foundCalls[3].pout_arg1;
        inputs[4] = BlasInfo(tmp, N, 1);

        cblas_dcopy(N, tri, N + 1, tmp, 1);
        cblas_dscal(N, 0.5, tmp, 1);
        cblas_dlascl(layout, flip_uplo(uplo), 0, 0, 1.0, 0.0, N, N, tri, N, 0);
        cblas_dcopy(N, tmp, 1, tri, N + 1);

        cblas_dtrsm(layout, uplo_to_rside(uplo), uplo, 'N', 'N', N, N, 1.0,
                    A, lda, tri, N);
        cblas_dtrsm(layout, uplo_to_side(uplo), uplo, 'T', 'N', N, N, 1.0,
                    A, lda, tri, N);
#define triv(r, c)                                                               \
  tri[(r) * (layout == CblasRowMajor ? N : 1) +                            \
    (c) * (layout == CblasRowMajor ? 1 : N)]

        int upperinc = (&triv(0, 1) - &triv(0,0));
        int lowerinc = (&triv(1, 0) - &triv(0,0));
        if (layout == CblasColMajor) {
            assert(upperinc == N);
            assert(lowerinc == 1);
        } else {
          assert(upperinc == 1);
          assert(lowerinc == N);
        }
        bool is_lower = uplo == 'L' || uplo == 'l';
        for (int i = 0; i < N - 1; i++) {
          cblas_daxpy(N - i - 1, 1.0,
                      is_lower ? &triv(i, i + 1) : &triv(i + 1, i),
                      is_lower ? upperinc : lowerinc,
                      is_lower ? &triv(i + 1, i) : &triv(i, i + 1),
                      is_lower ? lowerinc : upperinc);
        }

        cblas_dlacpy(layout, uplo, N, N, tri, N, dA, lda);

        checkTest(Test);

        SkipVecIncCheck = true;
        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        SkipVecIncCheck = false;
      }
      {

        std::string Test = "POTRF overwrite A ";
        init();

        ow_potrf(layout, uplo, N, A, lda);

        // Check memory of primal on own.
        checkMemoryTrace(inputs, "Primal " + Test, calls);

        init();
        __enzyme_autodiff((void *)ow_potrf, enzyme_const, layout, enzyme_const,
                          uplo, enzyme_const, N, enzyme_dup, A, dA,
                          enzyme_const, lda);
        foundCalls = calls;
        init();

        cblas_dpotrf(layout, uplo, N, A, lda, nullptr);
        double *cacheA = (double *)foundCalls[1].pout_arg1;
        inputs[5] = BlasInfo(cacheA, (char)layout, N, N, N);
        assert(inputs[5].ty == ValueType::Matrix);
        cblas_dlacpy(layout, uplo, N, N, A, lda, cacheA, N);
        cblas_dscal(1, 0.0, A, lda);

        inDerivative = true;
        cblas_dscal(1, 0.0, dA, lda);

        assert(foundCalls.size() >= 2);
        assert(foundCalls[4].type == CallType::LACPY);
        double *tri = (double *)foundCalls[4].pout_arg1;
        inputs[3] = BlasInfo(tri, (char)layout, N, N, N);

        cblas_dlacpy(layout, uplo, N, N, dA, lda, tri, N);

        cblas_dtrmm(layout, uplo_to_side(uplo), uplo, 'T', 'N', N, N, 1.0,
                    cacheA, N, tri, N);

        assert(foundCalls.size() >= 5);
        assert(foundCalls[6].type == CallType::COPY);
        double *tmp = (double *)foundCalls[6].pout_arg1;
        inputs[4] = BlasInfo(tmp, N, 1);

        cblas_dcopy(N, tri, N + 1, tmp, 1);
        cblas_dscal(N, 0.5, tmp, 1);
        cblas_dlascl(layout, flip_uplo(uplo), 0, 0, 1.0, 0.0, N, N, tri, N, 0);
        cblas_dcopy(N, tmp, 1, tri, N + 1);

        cblas_dtrsm(layout, uplo_to_rside(uplo), uplo, 'N', 'N', N, N, 1.0,
                    cacheA, N, tri, N);
        cblas_dtrsm(layout, uplo_to_side(uplo), uplo, 'T', 'N', N, N, 1.0,
                    cacheA, N, tri, N);
#define triv(r, c)                                                             \
  tri[(r) * (layout == CblasRowMajor ? N : 1) +                                \
      (c) * (layout == CblasRowMajor ? 1 : N)]

        int upperinc = (&triv(0, 1) - &triv(0, 0));
        int lowerinc = (&triv(1, 0) - &triv(0, 0));
        if (layout == CblasColMajor) {
          assert(upperinc == N);
          assert(lowerinc == 1);
        } else {
          assert(upperinc == 1);
          assert(lowerinc == N);
        }
        bool is_lower = uplo == 'L' || uplo == 'l';
        for (int i = 0; i < N - 1; i++) {
          cblas_daxpy(N - i - 1, 1.0,
                      is_lower ? &triv(i, i + 1) : &triv(i + 1, i),
                      is_lower ? upperinc : lowerinc,
                      is_lower ? &triv(i + 1, i) : &triv(i, i + 1),
                      is_lower ? lowerinc : upperinc);
        }

        cblas_dlacpy(layout, uplo, N, N, tri, N, dA, lda);

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

static void potrsTests() {
  int N = 17;
  int Nrhs = M;
  // N means normal matrix, T means transposed
  for (char layout : {CblasColMajor, CblasRowMajor}) {
    for (auto uplo : {'U', 'u', 'L', 'l'})

    {
      BlasInfo inputs[6] = {
          /*A*/ BlasInfo(A, layout, N, N, lda),
          /*B*/ BlasInfo(B, layout, N, Nrhs, incB),
          /*C*/ BlasInfo(),
          BlasInfo(),
          BlasInfo(),
          BlasInfo(),
      };
      {

        std::string Test = "POTRS active A, B";
        init();

        my_potrs(layout, uplo, N, Nrhs, A, lda, B, incB);

        assert(calls.size() == 1);
        assert(calls[0].inDerivative == false);
        assert(calls[0].type == CallType::POTRS);
        assert(calls[0].pout_arg1 == B);
        assert(calls[0].pin_arg1 == A);
        assert(calls[0].pin_arg2 == UNUSED_POINTER);
        assert(calls[0].farg1 == UNUSED_DOUBLE);
        assert(calls[0].farg2 == UNUSED_DOUBLE);
        assert(calls[0].layout == layout);
        assert(calls[0].targ1 == UNUSED_TRANS);
        assert(calls[0].targ2 == UNUSED_TRANS);
        assert(calls[0].iarg1 == N);
        assert(calls[0].iarg2 == Nrhs);
        assert(calls[0].iarg3 == UNUSED_INT);
        assert(calls[0].iarg4 == lda);
        assert(calls[0].iarg5 == incB);
        assert(calls[0].iarg6 == UNUSED_INT);
        assert(calls[0].side == UNUSED_TRANS);
        assert(calls[0].uplo == uplo);
        assert(calls[0].diag == UNUSED_TRANS);

        // Check memory of primal on own.
        checkMemoryTrace(inputs, "Primal " + Test, calls);

        init();
        __enzyme_autodiff((void *)my_potrs, enzyme_const, layout, enzyme_const,
                          uplo, enzyme_const, N, enzyme_const, Nrhs, enzyme_dup, A, dA,
                          enzyme_const, lda, enzyme_dup, B, dB, enzyme_const, incB);
        foundCalls = calls;
        init();

        assert(foundCalls[0].type == CallType::LACPY);
        double *inpB = (double *)foundCalls[0].pout_arg1;
        inputs[3] = BlasInfo(inpB, layout, N, Nrhs, N);
        cblas_dlacpy(layout, '\0', N, Nrhs, B, incB, inpB, N);

        my_potrs(layout, uplo, N, Nrhs, A, lda, B, incB);

        inDerivative = true;

        assert(foundCalls[2].type == CallType::SYR2K);
        double *tri = (double *)foundCalls[2].pout_arg1;
        inputs[4] = BlasInfo(tri, layout, N, N, N);
        cblas_dsyr2k(layout, 'U', 'N', N, Nrhs, 1.0, inpB, N, dB, incB, 0.0,
                     tri, N);

#define triv(r, c)                                                               \
  tri[(r) * (layout == CblasRowMajor ? N : 1) +                            \
    (c) * (layout == CblasRowMajor ? 1 : N)]

        bool is_lower = uplo == 'L' || uplo == 'l';
        int upperinc = (&triv(0, 1) - &triv(0,0));
        int lowerinc = (&triv(1, 0) - &triv(0,0));
        if (layout == CblasColMajor) {
            assert(upperinc == N);
            assert(lowerinc == 1);
        } else {
          assert(upperinc == 1);
          assert(lowerinc == N);
        }
          for (int i = 0; i < N - 1; i++) {
            cblas_dcopy(N - i - 1, &triv(i, i + 1), upperinc, &triv(i + 1, i),
                        lowerinc);
          }

          cblas_dtrsm(layout, uplo_to_rside(uplo), uplo, 'T', 'N', N, N, 1.0, A,
                      lda, tri, N);
          
          cblas_dtrsm(layout, uplo_to_side(uplo), uplo, 'N', 'N', N, N, 1.0, A,
                      lda, tri, N);
          
          cblas_dtrsm(layout, uplo_to_side(uplo), uplo, 'T', 'N', N, N, 1.0, A,
                      lda, tri, N);


#define Av(r, c)                                                               \
  dA[(r) * (layout == CblasRowMajor ? lda : 1) +                            \
    (c) * (layout == CblasRowMajor ? 1 : lda)]
        
        int Aupperinc = (&Av(0, 1) - &Av(0,0));
        int Alowerinc = (&Av(1, 0) - &Av(0,0));
        if (layout == CblasColMajor) {
            assert(Aupperinc == lda);
            assert(Alowerinc == 1);
        } else {
          assert(Aupperinc == 1);
          assert(Alowerinc == lda);
        }

        for (int i = 0; i < N; i++) {
          cblas_daxpy(N - i, -1.0, &triv(i, i), is_lower ? lowerinc : upperinc,
                      &Av(i, i), is_lower ? Alowerinc : Aupperinc);
        }

        cblas_dpotrs(layout, uplo, N, Nrhs, A, lda, dB, incB, nullptr);

        checkTest(Test);

        SkipVecIncCheck = true;
        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, calls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCalls);
        SkipVecIncCheck = false;
      }
      {

        std::string Test = "POTRS active B";

        init();
        __enzyme_autodiff((void *)my_potrs, enzyme_const, layout, enzyme_const,
                          uplo, enzyme_const, N, enzyme_const, Nrhs, enzyme_const, A,
                          enzyme_const, lda, enzyme_dup, B, dB, enzyme_const, incB);
        foundCalls = calls;
        init();

        my_potrs(layout, uplo, N, Nrhs, A, lda, B, incB);

        inDerivative = true;


        cblas_dpotrs(layout, uplo, N, Nrhs, A, lda, dB, incB, nullptr);

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

static void trtrsTests() {
  int N = 17;
  int Nrhs = M;
  // N means normal matrix, T means transposed
  for (char layout : {CblasColMajor, CblasRowMajor}) {
    for (auto uplo : {'U', 'u', 'L', 'l'})
      for (auto diag : {'U', 'u', 'N', 'n'})
        for (auto transA :
             {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans}) {
          BlasInfo inputs[6] = {
              /*A*/ BlasInfo(A, layout, N, N, lda),
              /*B*/ BlasInfo(B, layout, N, Nrhs, incB),
              /*C*/ BlasInfo(),
              BlasInfo(),
              BlasInfo(),
              BlasInfo(),
          };
          {

            std::string Test = "TRTRS active A, B";
            init();

            my_trtrs(layout, uplo, (char)transA, diag, N, Nrhs, A, lda, B,
                     incB);

            assert(calls.size() == 1);
            assert(calls[0].inDerivative == false);
            assert(calls[0].type == CallType::TRTRS);
            assert(calls[0].pout_arg1 == B);
            assert(calls[0].pin_arg1 == A);
            assert(calls[0].pin_arg2 == UNUSED_POINTER);
            assert(calls[0].farg1 == UNUSED_DOUBLE);
            assert(calls[0].farg2 == UNUSED_DOUBLE);
            assert(calls[0].layout == layout);
            assert(calls[0].targ1 == (char)transA);
            assert(calls[0].targ2 == UNUSED_TRANS);
            assert(calls[0].iarg1 == N);
            assert(calls[0].iarg2 == Nrhs);
            assert(calls[0].iarg3 == UNUSED_INT);
            assert(calls[0].iarg4 == lda);
            assert(calls[0].iarg5 == incB);
            assert(calls[0].iarg6 == UNUSED_INT);
            assert(calls[0].side == UNUSED_TRANS);
            assert(calls[0].uplo == uplo);
            assert(calls[0].diag == diag);

            // Check memory of primal on own.
            checkMemoryTrace(inputs, "Primal " + Test, calls);

            init();
            __enzyme_autodiff((void *)my_trtrs, enzyme_const, layout,
                              enzyme_const, uplo, enzyme_const, (char)transA,
                              enzyme_const, diag, enzyme_const, N, enzyme_const,
                              Nrhs, enzyme_dup, A, dA, enzyme_const, lda,
                              enzyme_dup, B, dB, enzyme_const, incB);
            foundCalls = calls;
            init();

            my_trtrs(layout, uplo, (char)transA, diag, N, Nrhs, A, lda, B,
                     incB);

            inDerivative = true;

            cblas_dtrtrs(layout, uplo, (char)transpose(transA), diag, N, Nrhs,
                         A, lda, dB, incB, nullptr);

            assert(foundCalls[2].type == CallType::LACPY);
            double *tri = (double *)foundCalls[2].pout_arg1;
            inputs[3] = BlasInfo(tri, layout, N, N, N);

            cblas_dlacpy(layout, uplo, N, N, dA, lda, tri, N);

            cblas_dgemm(
                layout, 'N', 'T', N, N, Nrhs, -1.0, is_normal(transA) ? dB : B,
                is_normal(transA) ? incB : incB, is_normal(transA) ? B : dB,
                is_normal(transA) ? incB : incB, 1.0, tri, N);

            cblas_dcopy((diag == 'U' || diag == 'u') ? N : 0, dA, lda + 1, tri,
                        N + 1);

            cblas_dlacpy(layout, uplo, N, N, tri, N, dA, lda);

            checkTest(Test);

            SkipVecIncCheck = true;
            // Check memory of primal of expected derivative
            checkMemoryTrace(inputs, "Expected " + Test, calls);

            // Check memory of primal of our derivative (if equal above, it
            // should be the same).
            checkMemoryTrace(inputs, "Found " + Test, foundCalls);
            SkipVecIncCheck = false;
          }
          {

            std::string Test = "TRTRS active B";

            init();
            __enzyme_autodiff((void *)my_trtrs, enzyme_const, layout,
                              enzyme_const, uplo, enzyme_const, (char)transA,
                              enzyme_const, diag, enzyme_const, N, enzyme_const,
                              Nrhs, enzyme_const, A, enzyme_const, lda,
                              enzyme_dup, B, dB, enzyme_const, incB);
            foundCalls = calls;
            init();

            my_trtrs(layout, uplo, (char)transA, diag, N, Nrhs, A, lda, B,
                     incB);

            inDerivative = true;

            cblas_dtrtrs(layout, uplo, (char)transpose(transA), diag, N, Nrhs,
                         A, lda, dB, incB, nullptr);

            // Check memory of primal of expected derivative
            checkMemoryTrace(inputs, "Expected " + Test, calls);

            // Check memory of primal of our derivative (if equal above, it
            // should be the same).
            checkMemoryTrace(inputs, "Found " + Test, foundCalls);
          }
          {

            std::string Test = "TRTRS active A";

            init();
            __enzyme_autodiff((void *)my_trtrs, enzyme_const, layout,
                              enzyme_const, uplo, enzyme_const, (char)transA,
                              enzyme_const, diag, enzyme_const, N, enzyme_const,
                              Nrhs, enzyme_dup, A, dA, enzyme_const, lda,
                              enzyme_const, B, enzyme_const, incB);
            foundCalls = calls;
            init();

            my_trtrs(layout, uplo, (char)transA, diag, N, Nrhs, A, lda, B,
                     incB);

            inDerivative = true;

            // Check memory of primal of expected derivative
            checkMemoryTrace(inputs, "Expected " + Test, calls);

            // Check memory of primal of our derivative (if equal above, it
            // should be the same).
            checkMemoryTrace(inputs, "Found " + Test, foundCalls);
          }
          {

            std::string Test = "TRTRS OW active A, B";

            init();
            __enzyme_autodiff((void *)ow_trtrs, enzyme_const, layout,
                              enzyme_const, uplo, enzyme_const, (char)transA,
                              enzyme_const, diag, enzyme_const, N, enzyme_const,
                              Nrhs, enzyme_dup, A, dA, enzyme_const, lda,
                              enzyme_dup, B, dB, enzyme_const, incB);
            foundCalls = calls;
            init();

            cblas_dtrtrs(layout, uplo, (char)transA, diag, N, Nrhs, A, lda, B,
                         incB, nullptr);
            assert(foundCalls[1].type == CallType::LACPY);
            double *cacheA = (double *)foundCalls[1].pout_arg1;
            inputs[4] = BlasInfo(cacheA, (char)layout, N, N, N);
            assert(inputs[4].ty == ValueType::Matrix);
            cblas_dlacpy(layout, uplo, N, N, A, lda, cacheA, N);

            assert(foundCalls[2].type == CallType::LACPY);
            double *cacheB = (double *)foundCalls[2].pout_arg1;
            inputs[5] = BlasInfo(cacheB, (char)layout, N, Nrhs, N);
            assert(inputs[5].ty == ValueType::Matrix);
            cblas_dlacpy(layout, '\0', N, Nrhs, B, incB, cacheB, N);
            cblas_dscal(1, 0.0, A, lda);

            inDerivative = true;

            cblas_dscal(1, 0.0, dA, lda);

            cblas_dtrtrs(layout, uplo, (char)transpose(transA), diag, N, Nrhs,
                         cacheA, N, dB, incB, nullptr);

            assert(foundCalls[6].type == CallType::LACPY);
            double *tri = (double *)foundCalls[6].pout_arg1;
            inputs[3] = BlasInfo(tri, layout, N, N, N);

            cblas_dlacpy(layout, uplo, N, N, dA, lda, tri, N);

            cblas_dgemm(layout, 'N', 'T', N, N, Nrhs, -1.0,
                        is_normal(transA) ? dB : cacheB,
                        is_normal(transA) ? incB : N,
                        is_normal(transA) ? cacheB : dB,
                        is_normal(transA) ? N : incB, 1.0, tri, N);

            cblas_dcopy((diag == 'U' || diag == 'u') ? N : 0, dA, lda + 1, tri,
                        N + 1);

            cblas_dlacpy(layout, uplo, N, N, tri, N, dA, lda);

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

static void symmTests() {
  int N = 17;
  int M = 9;
  // N means normal matrix, T means transposed
  for (char layout : {CblasColMajor, CblasRowMajor}) {
    for (auto uplo : {'U', 'u', 'L', 'l'})
      for (auto side : {'L', 'l', 'R', 'r'}) {
          BlasInfo inputs[6] = {
              /*A*/ BlasInfo(A, layout, is_left(side) ? M : N, is_left(side) ? M : N, lda),
              /*B*/ BlasInfo(B, layout, M, N, incB),
              /*C*/ BlasInfo(C, layout, M, N, incC),
              BlasInfo(),
              BlasInfo(),
              BlasInfo(),
          };
          {

            std::string Test = "SYMM active A, B, C";
            init();

            my_symm(layout, side, uplo, M, N, alpha, A, lda, B, incB, beta, C, incC);

            assert(calls.size() == 1);
            assert(calls[0].inDerivative == false);
            assert(calls[0].type == CallType::SYMM);
            assert(calls[0].pout_arg1 == C);
            assert(calls[0].pin_arg1 == A);
            assert(calls[0].pin_arg2 == B);
            assert(calls[0].farg1 == alpha);
            assert(calls[0].farg2 == beta);
            assert(calls[0].layout == layout);
            assert(calls[0].targ1 == UNUSED_TRANS);
            assert(calls[0].targ2 == UNUSED_TRANS);
            assert(calls[0].iarg1 == M);
            assert(calls[0].iarg2 == N);
            assert(calls[0].iarg3 == UNUSED_INT);
            assert(calls[0].iarg4 == lda);
            assert(calls[0].iarg5 == incB);
            assert(calls[0].iarg6 == incC);
            assert(calls[0].side == side);
            assert(calls[0].uplo == uplo);
            assert(calls[0].diag == UNUSED_TRANS);

            // Check memory of primal on own.
            checkMemoryTrace(inputs, "Primal " + Test, calls);

            init();
            __enzyme_autodiff((void *)my_symm,
                                enzyme_const, layout,
                                enzyme_const, side,
                                enzyme_const, uplo,
                                enzyme_const, M,
                                enzyme_const, N,
                                enzyme_const, alpha,
                                enzyme_dup, A, dA,
                                enzyme_const, lda,
                                enzyme_dup, B, dB,
                                enzyme_const, incB,
                                enzyme_const, beta,
                                enzyme_dup, C, dC,
                                enzyme_const, incC);
            foundCalls = calls;
            init();

            my_symm(layout, side, uplo, M, N, alpha, A, lda, B, incB, beta, C, incC);

            inDerivative = true;


            assert(foundCalls[1].type == CallType::COPY);
            double *tmp = (double *)foundCalls[1].pout_arg1;
			cblas_dcopy(is_left(side) ? M : N, dA, lda+1, tmp, 1);
			inputs[3] = BlasInfo(tmp, is_left(side) ? M : N, 1);

            //  ssyr2k(uplo, 'n', m, n, alpha,B,ldb,Ca,ldc, 1.0,Aa,lda)
            //  ssyr2k(uplo,'t', n,m, alpha,B,ldb,Ca,ldc, 1.0,Aa,lda)
            cblas_dsyr2k(layout,
                          uplo,
                          side_to_trans(side), 
                          is_left(side) ? M : N,
                          is_left(side) ? N : M,
                          alpha,
                          B,
                          incB,
                          dC,
                          incC,
                          1.0,
                          dA,
                          lda);

            cblas_daxpy(is_left(side) ? M : N, -1, dA, lda+1, tmp, 1);
            cblas_daxpy(is_left(side) ? M : N, 0.5, tmp, 1, dA, lda+1);
            
            cblas_dsymm(layout, side, uplo, M, N, alpha, A, lda, dC, incC, 1.0, dB, incB);
        
            cblas_dlascl(layout, 'G', 0, 0, 1.0, beta, M, N, dC, incC, 0 );

            checkTest(Test);

            SkipVecIncCheck = true;
            // Check memory of primal of expected derivative
            checkMemoryTrace(inputs, "Expected " + Test, calls);

            // Check memory of primal of our derivative (if equal above, it
            // should be the same).
            checkMemoryTrace(inputs, "Found " + Test, foundCalls);
            SkipVecIncCheck = false;
          }
          {

            std::string Test = "SYMM overwriten active A, B, C";
            init();

            ow_symm(layout, side, uplo, M, N, alpha, A, lda, B, incB, beta, C, incC);

            // Check memory of primal on own.
            checkMemoryTrace(inputs, "Primal " + Test, calls);

            init();
            __enzyme_autodiff((void *)ow_symm,
                                enzyme_const, layout,
                                enzyme_const, side,
                                enzyme_const, uplo,
                                enzyme_const, M,
                                enzyme_const, N,
                                enzyme_const, alpha,
                                enzyme_dup, A, dA,
                                enzyme_const, lda,
                                enzyme_dup, B, dB,
                                enzyme_const, incB,
                                enzyme_const, beta,
                                enzyme_dup, C, dC,
                                enzyme_const, incC);
            foundCalls = calls;
            init();

            double *cacheA = (double *)foundCalls[0].pout_arg1;
            inputs[4] = BlasInfo(cacheA, layout, is_left(side) ? M : N, is_left(side) ? M : N, is_left(side) ? M : N);
            assert(inputs[4].ty == ValueType::Matrix);
            cblas_dlacpy(layout, '\0', is_left(side) ? M : N, is_left(side) ? M : N, A, lda, cacheA, is_left(side) ? M : N);
            
            double *cacheB = (double *)foundCalls[1].pout_arg1;
            inputs[5] = BlasInfo(cacheB, layout, M, N, M);
            assert(inputs[5].ty == ValueType::Matrix);
            cblas_dlacpy(layout, '\0', M, N, B, incB, cacheB, M);

            ow_symm(layout, side, uplo, M, N, alpha, A, lda, B, incB, beta, C, incC);

            inDerivative = true;

            //cblas_dscal(1, 0.0, dA, lda);


            //assert(foundCalls[1].type == CallType::COPY);
            double *tmp = (double *)foundCalls[3].pout_arg1;
			cblas_dcopy(is_left(side) ? M : N, dA, lda+1, tmp, 1);
			inputs[3] = BlasInfo(tmp, is_left(side) ? M : N, 1);

            //  ssyr2k(uplo, 'n', m, n, alpha,B,ldb,Ca,ldc, 1.0,Aa,lda)
            //  ssyr2k(uplo,'t', n,m, alpha,B,ldb,Ca,ldc, 1.0,Aa,lda)
            cblas_dsyr2k(layout,
                          uplo,
                          side_to_trans(side), 
                          is_left(side) ? M : N,
                          is_left(side) ? N : M,
                          alpha,
                          cacheB,
                          M,
                          dC,
                          incC,
                          1.0,
                          dA,
                          lda);

            cblas_daxpy(is_left(side) ? M : N, -1, dA, lda+1, tmp, 1);
            cblas_daxpy(is_left(side) ? M : N, 0.5, tmp, 1, dA, lda+1);
            
            cblas_dsymm(layout, side, uplo, M, N, alpha, cacheA, is_left(side) ? M : N, dC, incC, 1.0, dB, incB);
        
            cblas_dlascl(layout, 'G', 0, 0, 1.0, beta, M, N, dC, incC, 0 );

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
    /*
  dotTests();

  nrm2Tests();

  gemvTests();

  gemmTests();

  trmvTests();

  trmmTests();

  syrkTests();

  potrfTests();

  potrsTests();

  trtrsTests();
  
  symmTests();
  */

  symvTests();
}
