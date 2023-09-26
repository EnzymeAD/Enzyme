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

#include "../cublasinfra.h"

int enzyme_dup;
int enzyme_out;
int enzyme_const;
template <typename... T> void __enzyme_autodiff(void *, T...);

void my_dgemv(cublasHandle_t *handle, cublasOperation_t trans, int M, int N,
              double alpha, double *__restrict__ A, int lda,
              double *__restrict__ X, int incx, double beta,
              double *__restrict__ Y, int incy) {
  cublas_dgemv(handle, trans, M, N, alpha, A, lda, X, incx, beta, Y, incy);
  inDerivative = true;
}

void ow_dgemv(cublasHandle_t *handle, cublasOperation_t trans, int M, int N,
              double alpha, double *A, int lda, double *X, int incx,
              double beta, double *Y, int incy) {
  cublas_dgemv(handle, trans, M, N, alpha, A, lda, X, incx, beta, Y, incy);
  inDerivative = true;
}

double my_ddot(cublasHandle_t *handle, int N, double *__restrict__ X, int incx,
               double *__restrict__ Y, int incy) {
  double res = 0.0;
  cublas_ddot(handle, N, X, incx, Y, incy, &res);
  inDerivative = true;
  return res;
}

void my_dgemm(cublasHandle_t *handle, cublasOperation_t transA,
              cublasOperation_t transB, int M, int N, int K, double alpha,
              double *__restrict__ A, int lda, double *__restrict__ B, int ldb,
              double beta, double *__restrict__ C, int ldc) {
  cublas_dgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
               ldc);
  inDerivative = true;
}

static void dotTests() {

  std::string Test = "DOT active both ";
  cublasHandle_t *handle = USED_CUBLAS_HANDLE;
  BlasInfo inputs[6] = {
      /*A*/ BlasInfo(A, handle, N, incA),
      /*B*/ BlasInfo(B, handle, N, incB),
      /*C*/ BlasInfo(C, handle, M, incC),
      BlasInfo(),
      BlasInfo(),
      BlasInfo(),
  };
  init();
  // cublasHandle_t handle;
  my_ddot(handle, N, A, incA, B, incB);

  // Check memory of primal on own.
  checkMemoryTrace(inputs, "Primal " + Test, cucalls);

  init();
  __enzyme_autodiff((void *)my_ddot, enzyme_const, handle, enzyme_const, N,
                    enzyme_dup, A, dA, enzyme_const, incA, enzyme_dup, B, dB,
                    enzyme_const, incB);
  foundCuCalls = cucalls;
  init();

  my_ddot(handle, N, A, incA, B, incB);

  inDerivative = true;

  cublas_daxpy(handle, N, 1.0, B, incB, dA, incA);
  cublas_daxpy(handle, N, 1.0, A, incA, dB, incB);

  checkTest(Test);

  // Check memory of primal of expected derivative
  checkMemoryTrace(inputs, "Expected " + Test, cucalls);

  // Check memory of primal of our derivative (if equal above, it
  // should be the same).
  checkMemoryTrace(inputs, "Found " + Test, foundCuCalls);
}

static void gemvTests() {
  // N means normal matrix, T means transposed
  for (cublasOperation_t transA :
       {cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T}) {

    {

      bool trans = (transA == cublasOperation_t::CUBLAS_OP_T);
      auto handle = USED_CUBLAS_HANDLE;
      std::string Test = "GEMV active A, C ";
      BlasInfo inputs[6] = {/*A*/ BlasInfo(A, handle, M, N, lda),
                            /*B*/ BlasInfo(B, handle, trans ? M : N, incB),
                            /*C*/ BlasInfo(C, handle, trans ? N : M, incC),
                            BlasInfo(),
                            BlasInfo(),
                            BlasInfo()};
      init();
      my_dgemv(handle, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

      assert(cucalls.size() == 1);
      assert(cucalls[0].inDerivative == false);
      assert(cucalls[0].type == CallType::GEMV);
      assert(cucalls[0].pout_arg1 == C);
      assert(cucalls[0].pin_arg1 == A);
      assert(cucalls[0].pin_arg2 == B);
      assert(cucalls[0].farg1 == alpha);
      assert(cucalls[0].farg2 == beta);
      assert(cucalls[0].handle == USED_CUBLAS_HANDLE);
      assert(cucalls[0].op1 == transA);
      assert(cucalls[0].op2 == cublasOperation_t::CUBLAS_OP_UNUSED);
      assert(cucalls[0].iarg1 == M);
      assert(cucalls[0].iarg2 == N);
      assert(cucalls[0].iarg3 == UNUSED_INT);
      assert(cucalls[0].iarg4 == lda);
      assert(cucalls[0].iarg5 == incB);
      assert(cucalls[0].iarg6 == incC);

      // Check memory of primal on own.
      checkMemoryTrace(inputs, "Primal " + Test, cucalls);

      init();
      __enzyme_autodiff((void *)my_dgemv, enzyme_const, handle, enzyme_const,
                        transA, enzyme_const, M, enzyme_const, N, enzyme_const,
                        alpha, enzyme_dup, A, dA, enzyme_const, lda,
                        enzyme_const, B, enzyme_const, incB, enzyme_const, beta,
                        enzyme_dup, C, dC, enzyme_const, incC);
      foundCuCalls = cucalls;
      init();

      my_dgemv(handle, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

      inDerivative = true;
      // dC = alpha * X * transpose(Y) + A
      cublas_dger(handle, M, N, alpha, trans ? B : dC, trans ? incB : incC,
                  trans ? dC : B, trans ? incC : incB, dA, lda);
      // dY = beta * dY
      cublas_dscal(handle, trans ? N : M, beta, dC, incC);

      checkTest(Test);

      // Check memory of primal of expected derivative
      checkMemoryTrace(inputs, "Expected " + Test, cucalls);

      // Check memory of primal of our derivative (if equal above, it
      // should be the same).
      checkMemoryTrace(inputs, "Found " + Test, foundCuCalls);

      Test = "GEMV active A, B, C ";

      init();
      __enzyme_autodiff((void *)my_dgemv, enzyme_const, handle, enzyme_const,
                        transA, enzyme_const, M, enzyme_const, N, enzyme_const,
                        alpha, enzyme_dup, A, dA, enzyme_const, lda, enzyme_dup,
                        B, dB, enzyme_const, incB, enzyme_const, beta,
                        enzyme_dup, C, dC, enzyme_const, incC);
      foundCuCalls = cucalls;
      init();

      my_dgemv(handle, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

      inDerivative = true;
      // dC = alpha * X * transpose(Y) + A
      cublas_dger(handle, M, N, alpha, trans ? B : dC, trans ? incB : incC,
                  trans ? dC : B, trans ? incC : incB, dA, lda);

      // dB = alpha * trans(A) * dC + dB
      cublas_dgemv(handle, transpose(transA), M, N, alpha, A, lda, dC, incC,
                   1.0, dB, incB);

      // dY = beta * dY
      cublas_dscal(handle, trans ? N : M, beta, dC, incC);

      checkTest(Test);

      // Check memory of primal of expected derivative
      checkMemoryTrace(inputs, "Expected " + Test, cucalls);

      // Check memory of primal of our derivative (if equal above, it
      // should be the same).
      checkMemoryTrace(inputs, "Found " + Test, foundCuCalls);

      Test = "GEMV active/overwrite";

      init();
      __enzyme_autodiff((void *)ow_dgemv, enzyme_const, handle, enzyme_const,
                        transA, enzyme_const, M, enzyme_const, N, enzyme_const,
                        alpha, enzyme_dup, A, dA, enzyme_const, lda, enzyme_dup,
                        B, dB, enzyme_const, incB, enzyme_const, beta,
                        enzyme_dup, C, dC, enzyme_const, incC);
      foundCuCalls = cucalls;
      init();

      assert(foundCuCalls.size() > 2);
      auto A_cache = (double *)foundCuCalls[0].pout_arg1;
      // dlacpy is not supported for cublas @wsmoses
      // cublas_dlacpy(handle, '\0', M, N, A, lda, A_cache, M);
      inputs[4] = BlasInfo(A_cache, handle, M, N, M);
      auto B_cache = (double *)foundCuCalls[1].pout_arg1;
      cublas_dcopy(handle, trans ? M : N, B, incB, B_cache, 1);
      inputs[5] = BlasInfo(B_cache, handle, trans ? M : N, 1);

      ow_dgemv(handle, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

      inDerivative = true;
      // dC = alpha * X * transpose(Y) + A
      cublas_dger(handle, M, N, alpha, trans ? B_cache : dC, trans ? 1 : incC,
                  trans ? dC : B_cache, trans ? incC : 1, dA, lda);

      // dB = alpha * trans(A) * dC + dB
      cublas_dgemv(handle, transpose(transA), M, N, alpha, A_cache, M, dC, incC,
                   1.0, dB, incB);

      // dY = beta * dY
      // cublas_dscal(trans ? N : M, beta, dC, incC);

      checkTest(Test);

      // Check memory of primal of expected derivative
      checkMemoryTrace(inputs, "Expected " + Test, cucalls);

      // Check memory of primal of our derivative (if equal above, it
      // should be the same).
      checkMemoryTrace(inputs, "Found " + Test, foundCuCalls);

      inputs[4] = BlasInfo();
      inputs[5] = BlasInfo();
    }
  }
}

static void gemmTests() {
  // N means normal matrix, T means transposed
  auto handle = USED_CUBLAS_HANDLE;
  for (auto transA :
       {cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T}) {
    for (auto transB :
         {cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T}) {

      {

        bool transA_bool = !is_normal(transA);
        bool transB_bool = !is_normal(transB);
        std::string Test = "GEMM";
        BlasInfo inputs[6] = {
            /*A*/ BlasInfo(A, handle, transA_bool ? K : M, transA_bool ? M : K,
                           lda),
            /*B*/
            BlasInfo(B, handle, transB_bool ? N : K, transB_bool ? K : N, incB),
            /*C*/ BlasInfo(C, handle, M, N, incC),
            BlasInfo(),
            BlasInfo(),
            BlasInfo()};
        init();
        my_dgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, incB, beta,
                 C, incC);

        assert(cucalls.size() == 1);
        assert(cucalls[0].inDerivative == false);
        assert(cucalls[0].type == CallType::GEMM);
        assert(cucalls[0].pout_arg1 == C);
        assert(cucalls[0].pin_arg1 == A);
        assert(cucalls[0].pin_arg2 == B);
        assert(cucalls[0].farg1 == alpha);
        assert(cucalls[0].farg2 == beta);
        assert(cucalls[0].handle == USED_CUBLAS_HANDLE);
        assert(cucalls[0].op1 == transA);
        assert(cucalls[0].op2 == transB);
        assert(cucalls[0].iarg1 == M);
        assert(cucalls[0].iarg2 == N);
        assert(cucalls[0].iarg3 == K);
        assert(cucalls[0].iarg4 == lda);
        assert(cucalls[0].iarg5 == incB);
        assert(cucalls[0].iarg6 == incC);

        // Check memory of primal on own.
        checkMemoryTrace(inputs, "Primal " + Test, cucalls);

        init();
        __enzyme_autodiff((void *)my_dgemm, enzyme_const, handle, enzyme_const,
                          transA, enzyme_const, transB, enzyme_const, M,
                          enzyme_const, N, enzyme_const, K, enzyme_const, alpha,
                          enzyme_dup, A, dA, enzyme_const, lda, enzyme_dup, B,
                          dB, enzyme_const, incB, enzyme_const, beta,
                          enzyme_dup, C, dC, enzyme_const, incC);
        foundCuCalls = cucalls;
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

        // not supported yet by cublas @wsmoses
        // cublas_dlascl(handle, 'G', 0, 0, 1.0, beta, M, N, dC, incC /*, extra
        // 0*/ );

        checkTest(Test);

        // Check memory of primal of expected derivative
        checkMemoryTrace(inputs, "Expected " + Test, cucalls);

        // Check memory of primal of our derivative (if equal above, it
        // should be the same).
        checkMemoryTrace(inputs, "Found " + Test, foundCuCalls);
      }
    }
  }
}

int main() {

  dotTests();

  gemvTests();

  gemmTests();
}
