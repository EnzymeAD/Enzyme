// This should work on LLVM 7, 8, 9, however in CI the version of clang
// installed on Ubuntu 18.04 cannot load a clang plugin properly without
// segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1  | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme  -mllvm -enzyme-lapack-copy=1 | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -mllvm -enzyme-lapack-copy=1 -S | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -mllvm -enzyme-lapack-copy=1 -S | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -mllvm -enzyme-lapack-copy=1  -S | %lli -
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
template <typename... T> void __enzyme_autodiff(void *, T...);

void my_dscal_v2(cublasHandle_t *handle, int N, double alpha,
                 double *__restrict__ X, int incx) {
  cublasDscal_v2(handle, N, &alpha, X, incx);
  inDerivative = true;
}

void my_dgemv(cublasHandle_t *handle, cublasOperation_t trans, int M, int N,
              double alpha, double *__restrict__ A, int lda,
              double *__restrict__ X, int incx, double beta,
              double *__restrict__ Y, int incy) {
  cublasDgemv(handle, trans, M, N, &alpha, A, lda, X, incx, &beta, Y, incy);
  inDerivative = true;
}

void ow_dgemv(cublasHandle_t *handle, cublasOperation_t trans, int M, int N,
              double alpha, double *A, int lda, double *X, int incx,
              double beta, double *Y, int incy) {
  cublasDgemv(handle, trans, M, N, &alpha, A, lda, X, incx, &beta, Y, incy);
  inDerivative = true;
}

double my_ddot(cublasHandle_t *handle, int N, double *__restrict__ X, int incx,
               double *__restrict__ Y, int incy) {
  double res = cublasDdot(handle, N, X, incx, Y, incy);
  inDerivative = true;
  return res;
}

double my_ddot2(cublasHandle_t *handle, int N, double *__restrict__ X, int incx,
                double *__restrict__ Y, int incy) {
  double res = 0.0;
  cublasDdot_v2(handle, N, X, incx, Y, incy, &res);
  inDerivative = true;
  return res;
}

void my_dgemm(cublasHandle_t *handle, cublasOperation_t transA,
              cublasOperation_t transB, int M, int N, int K, double alpha,
              double *__restrict__ A, int lda, double *__restrict__ B, int ldb,
              double beta, double *__restrict__ C, int ldc) {
  cublasDgemm(handle, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C,
              ldc);
  inDerivative = true;
}

static void scal2Tests() {

  std::string Test = "SCAL2 active both ";
  cublasHandle_t *handle = DEFAULT_CUBLAS_HANDLE;
  BlasInfo inputs[6] = {
      /*A*/ BlasInfo(A, N, incA),
      BlasInfo(),
      BlasInfo(),
      BlasInfo(),
      BlasInfo(),
      BlasInfo(),
  };
  init();

  double alpha = 3.14;
  // cublasHandle_t handle;
  my_dscal_v2(handle, N, alpha, A, incA);

  // Check memory of primal on own.
  checkMemoryTrace(inputs, "Primal " + Test, calls);

  init();
  __enzyme_autodiff((void *)my_dscal_v2, enzyme_const, handle, enzyme_const, N,
                    enzyme_out, alpha, enzyme_dup, A, dA, enzyme_const, incA);
  foundCalls = calls;

  init();

  my_dscal_v2(handle, N, alpha, A, incA);

  inDerivative = true;

  double *dalpha = (double *)foundCalls[1].pout_arg1;
  inputs[3] = BlasInfo(dalpha, 1, 1);

  cublasDdot_v2(handle, N, A, incA, dA, incA, dalpha);
  cublasDscal_v2(handle, N, &alpha, dA, incA);

  checkTest(Test);

  // Check memory of primal of expected derivative
  checkMemoryTrace(inputs, "Expected " + Test, calls);

  // Check memory of primal of our derivative (if equal above, it
  // should be the same).
  checkMemoryTrace(inputs, "Found " + Test, foundCalls);
}

static void dotTests() {

  std::string Test = "DOT active both ";
  cublasHandle_t *handle = DEFAULT_CUBLAS_HANDLE;
  BlasInfo inputs[6] = {
      /*A*/ BlasInfo(A, N, incA),
      /*B*/ BlasInfo(B, N, incB),
      /*C*/ BlasInfo(C, M, incC),
      BlasInfo(),
      BlasInfo(),
      BlasInfo(),
  };
  init();
  // cublasHandle_t handle;
  my_ddot(handle, N, A, incA, B, incB);

  // Check memory of primal on own.
  checkMemoryTrace(inputs, "Primal " + Test, calls);

  init();
  __enzyme_autodiff((void *)my_ddot, enzyme_const, handle, enzyme_const, N,
                    enzyme_dup, A, dA, enzyme_const, incA, enzyme_dup, B, dB,
                    enzyme_const, incB);
  foundCalls = calls;

  init();

  my_ddot(handle, N, A, incA, B, incB);

  inDerivative = true;

  cublasDaxpy(handle, N, 1.0, B, incB, dA, incA);
  cublasDaxpy(handle, N, 1.0, A, incA, dB, incB);

  checkTest(Test);

  // Check memory of primal of expected derivative
  checkMemoryTrace(inputs, "Expected " + Test, calls);

  // Check memory of primal of our derivative (if equal above, it
  // should be the same).
  checkMemoryTrace(inputs, "Found " + Test, foundCalls);
}

static void dot2Tests() {

  std::string Test = "DOTv2 active both ";
  cublasHandle_t *handle = DEFAULT_CUBLAS_HANDLE;
  BlasInfo inputs[6] = {
      /*A*/ BlasInfo(A, N, incA),
      /*B*/ BlasInfo(B, N, incB),
      /*C*/ BlasInfo(C, M, incC), BlasInfo(), BlasInfo(), BlasInfo(),
  };
  init();
  // cublasHandle_t handle;
  my_ddot2(handle, N, A, incA, B, incB);
  {
    auto primal_stack_ret = (double *)calls[0].pout_arg1;
    inputs[3] = BlasInfo(primal_stack_ret, 1, 1);
  }

  // Check memory of primal on own.
  checkMemoryTrace(inputs, "Primal " + Test, calls);

  init();
  __enzyme_autodiff((void *)my_ddot2, enzyme_const, handle, enzyme_const, N,
                    enzyme_dup, A, dA, enzyme_const, incA, enzyme_dup, B, dB,
                    enzyme_const, incB);
  {
    auto primal_stack_ret = (double *)calls[0].pout_arg1;
    inputs[3] = BlasInfo(primal_stack_ret, 1, 1);
  }
  foundCalls = calls;

  auto stack_ret = (double*)foundCalls[1].pin_arg2;
  inputs[4] = BlasInfo(stack_ret, 1, 1);

  init();

  my_ddot2(handle, N, A, incA, B, incB);

  calls[0].pout_arg1 = (double*)foundCalls[0].pout_arg1;

  inDerivative = true;

  cublasDaxpy_v2(handle, N, stack_ret, B, incB, dA, incA);
  cublasDaxpy_v2(handle, N, stack_ret, A, incA, dB, incB);
  cudaMemset(stack_ret, 0, sizeof(double));

  checkTest(Test);

  // Check memory of primal of expected derivative
  checkMemoryTrace(inputs, "Expected " + Test, calls);

  // Check memory of primal of our derivative (if equal above, it
  // should be the same).
  checkMemoryTrace(inputs, "Found " + Test, foundCalls);
}

static void gemvTests() {
  // N means normal matrix, T means transposed
  for (cublasOperation_t transA :
       {cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T}) {

    {

      bool trans = !is_normal(transA);
      auto handle = DEFAULT_CUBLAS_HANDLE;
      std::string Test = "GEMV active A, C ";
      BlasInfo inputs[6] = {/*A*/ BlasInfo(A, CUBLAS_LAYOUT, M, N, lda),
                            /*B*/ BlasInfo(B, trans ? M : N, incB),
                            /*C*/ BlasInfo(C, trans ? N : M, incC),
                            BlasInfo(),
                            BlasInfo(),
                            BlasInfo()};
      init();
      my_dgemv(handle, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

      assert(calls.size() == 1);
      assert(calls[0].inDerivative == false);
      assert(calls[0].type == CallType::GEMV);
      assert(calls[0].pout_arg1 == C);
      assert(calls[0].pin_arg1 == A);
      assert(calls[0].pin_arg2 == B);
      assert(calls[0].farg1 == alpha);
      assert(calls[0].farg2 == beta);
      assert(calls[0].handle == DEFAULT_CUBLAS_HANDLE);
      assert(calls[0].targ1 == (char)transA);
      assert(calls[0].targ2 == (char)UNUSED_TRANS);
      assert(calls[0].iarg1 == M);
      assert(calls[0].iarg2 == N);
      assert(calls[0].iarg3 == UNUSED_INT);
      assert(calls[0].iarg4 == lda);
      assert(calls[0].iarg5 == incB);
      assert(calls[0].iarg6 == incC);

      // Check memory of primal on own.
      checkMemoryTrace(inputs, "Primal " + Test, calls);

      init();
      __enzyme_autodiff((void *)my_dgemv, enzyme_const, handle, enzyme_const,
                        transA, enzyme_const, M, enzyme_const, N, enzyme_const,
                        alpha, enzyme_dup, A, dA, enzyme_const, lda,
                        enzyme_const, B, enzyme_const, incB, enzyme_const, beta,
                        enzyme_dup, C, dC, enzyme_const, incC);
      foundCalls = calls;
      init();

      my_dgemv(handle, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

      inDerivative = true;
      // dC = alpha * X * transpose(Y) + A
      cublasDger(handle, M, N, &alpha, trans ? B : dC, trans ? incB : incC,
                 trans ? dC : B, trans ? incC : incB, dA, lda);
      // dY = beta * dY
      cublasDscal(handle, trans ? N : M, &beta, dC, incC);

      checkTest(Test);

      // Check memory of primal of expected derivative
      checkMemoryTrace(inputs, "Expected " + Test, calls);

      // Check memory of primal of our derivative (if equal above, it
      // should be the same).
      checkMemoryTrace(inputs, "Found " + Test, foundCalls);

      Test = "GEMV active A, B, C ";

      init();
      __enzyme_autodiff((void *)my_dgemv, enzyme_const, handle, enzyme_const,
                        transA, enzyme_const, M, enzyme_const, N, enzyme_const,
                        alpha, enzyme_dup, A, dA, enzyme_const, lda, enzyme_dup,
                        B, dB, enzyme_const, incB, enzyme_const, beta,
                        enzyme_dup, C, dC, enzyme_const, incC);
      foundCalls = calls;
      init();

      my_dgemv(handle, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

      inDerivative = true;
      // dC = alpha * X * transpose(Y) + A
      cublasDger(handle, M, N, &alpha, trans ? B : dC, trans ? incB : incC,
                 trans ? dC : B, trans ? incC : incB, dA, lda);

      // dB = alpha * trans(A) * dC + dB
      double c1 = 1.0;
      cublasDgemv(handle, transpose(transA), M, N, &alpha, A, lda, dC, incC,
                  &c1, dB, incB);

      // dY = beta * dY
      cublasDscal(handle, trans ? N : M, &beta, dC, incC);

      checkTest(Test);

      // Check memory of primal of expected derivative
      checkMemoryTrace(inputs, "Expected " + Test, calls);

      // Check memory of primal of our derivative (if equal above, it
      // should be the same).
      checkMemoryTrace(inputs, "Found " + Test, foundCalls);

      // Next test fails to compile because the matrixcopy will use lacpy
      // which isn't supported by cublas
      //Test = "GEMV A,B,C active/overwrite";

      //init();
      //__enzyme_autodiff((void *)ow_dgemv, enzyme_const, handle, enzyme_const,
      //                  transA, enzyme_const, M, enzyme_const, N, enzyme_const,
      //                  alpha, enzyme_dup, A, dA, enzyme_const, lda, enzyme_dup,
      //                  B, dB, enzyme_const, incB, enzyme_const, beta,
      //                  enzyme_dup, C, dC, enzyme_const, incC);
      //foundCalls = calls;
      //init();

      //assert(foundCalls.size() > 2);
      //auto A_cache = (double *)foundCalls[0].pout_arg1;
      //// dlacpy is not supported for cublas @wsmoses
      //cublas_dlacpy(handle, '\0', M, N, A, lda, A_cache, M);
      //inputs[4] = BlasInfo(A_cache, handle, M, N, M);
      //auto B_cache = (double *)foundCalls[1].pout_arg1;
      //cublas_dcopy(handle, trans ? M : N, B, incB, B_cache, 1);
      //inputs[5] = BlasInfo(B_cache, handle, trans ? M : N, 1);

      //ow_dgemv(handle, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

      //inDerivative = true;
      //// dC = alpha * X * transpose(Y) + A
      //cublas_dger(handle, M, N, alpha, trans ? B_cache : dC, trans ? 1 : incC,
      //            trans ? dC : B_cache, trans ? incC : 1, dA, lda);

      //// dB = alpha * trans(A) * dC + dB
      //cublas_dgemv(handle, transpose(transA), M, N, alpha, A_cache, M, dC, incC,
      //             1.0, dB, incB);

      //// dY = beta * dY
      //cublas_dscal(handle, trans ? N : M, beta, dC, incC);

      //checkTest(Test);

      //// Check memory of primal of expected derivative
      //checkMemoryTrace(inputs, "Expected " + Test, calls);

      //// Check memory of primal of our derivative (if equal above, it
      //// should be the same).
      //checkMemoryTrace(inputs, "Found " + Test, foundCalls);

      //inputs[4] = BlasInfo();
      //inputs[5] = BlasInfo();
    }
  }
}

static void gemmTests() {
  // N means normal matrix, T means transposed
  auto handle = DEFAULT_CUBLAS_HANDLE;
  for (auto transA :
       {cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T}) {
    for (auto transB :
         {cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T}) {

      {

        bool transA_bool = !is_normal(transA);
        bool transB_bool = !is_normal(transB);
        std::string Test = "GEMM";
        BlasInfo inputs[6] = {
            /*A*/ BlasInfo(A, CUBLAS_LAYOUT, transA_bool ? K : M, transA_bool ? M : K,
                           lda),
            /*B*/
            BlasInfo(B, CUBLAS_LAYOUT, transB_bool ? N : K, transB_bool ? K : N, incB),
            /*C*/ BlasInfo(C, CUBLAS_LAYOUT, M, N, incC),
            BlasInfo(),
            BlasInfo(),
            BlasInfo()};
        init();
        my_dgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, incB, beta,
                 C, incC);

        assert(calls.size() == 1);
        assert(calls[0].inDerivative == false);
        assert(calls[0].type == CallType::GEMM);
        assert(calls[0].pout_arg1 == C);
        assert(calls[0].pin_arg1 == A);
        assert(calls[0].pin_arg2 == B);
        assert(calls[0].farg1 == alpha);
        assert(calls[0].farg2 == beta);
        assert(calls[0].handle == DEFAULT_CUBLAS_HANDLE);
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
        __enzyme_autodiff((void *)my_dgemm, enzyme_const, handle, enzyme_const,
                          transA, enzyme_const, transB, enzyme_const, M,
                          enzyme_const, N, enzyme_const, K, enzyme_const, alpha,
                          enzyme_dup, A, dA, enzyme_const, lda, enzyme_dup, B,
                          dB, enzyme_const, incB, enzyme_const, beta,
                          enzyme_dup, C, dC, enzyme_const, incC);
        foundCalls = calls;
        init();

        my_dgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, incB, beta,
                 C, incC);

        inDerivative = true;

        // dA =
        my_dgemm(handle, transA_bool ? transB : cublasOperation_t::CUBLAS_OP_N,
                 transA_bool ? cublasOperation_t::CUBLAS_OP_T
                             : transpose(transB),
                 transA_bool ? K : M, transA_bool ? M : K, N, alpha,
                 transA_bool ? B : dC, transA_bool ? incB : incC,
                 transA_bool ? dC : B, transA_bool ? incC : incB, 1.0, dA, lda);

        // dB =
        my_dgemm(
            handle,
            transB_bool ? cublasOperation_t::CUBLAS_OP_T : transpose(transA),
            transB_bool ? transA : cublasOperation_t::CUBLAS_OP_N, // transB,
            transB_bool ? N : K, transB_bool ? K : N, M, alpha,
            transB_bool ? dC : A, transB_bool ? incC : lda,
            transB_bool ? A : dC, transB_bool ? lda : incC, 1.0, dB, incB);

        // TODO we are currently faking support here, this needs to be actually implemented
        double c10 = 1.0;
        cublasDlascl(handle, (cublasOperation_t)'G', 0, 0, &c10, &beta, M, N,
                     dC, incC, 0);

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

int main() {
  gemmTests();

  gemvTests();

  dotTests();

  dot2Tests();

  scal2Tests();
}
