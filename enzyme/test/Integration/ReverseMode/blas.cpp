// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
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

double my_ddot(int N, double* __restrict__ X, int incx, double* __restrict__ Y, int incy) {
    double res = cblas_ddot(N, X, incx, Y, incy);
    inDerivative = true;
    return res;
}

void my_dgemm(char layout, char transA, char transB, int M, int N, int K, double alpha, double* __restrict__ A, int lda, double* __restrict__ B, int ldb, double beta, double* __restrict__ C, int ldc) {
    cblas_dgemm(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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

static void gemvTests() {
  // N means normal matrix, T means transposed
  for (char layout : { CblasRowMajor, CblasColMajor }) {
  for (char transA : {'N', 'n', 'T', 't'}) {
  
    {

        bool trans = !(transA == 'N' || transA == 'n');
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
    my_dgemv(layout, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

    assert(calls.size() == 1);
    assert(calls[0].inDerivative == false);
    assert(calls[0].type == CallType::GEMV);
    assert(calls[0].pout_arg1 == C);
    assert(calls[0].pin_arg1 == A);
    assert(calls[0].pin_arg2 == B);
    assert(calls[0].farg1 == alpha);
    assert(calls[0].farg2 == beta);
    assert(calls[0].layout == layout);
    assert(calls[0].targ1 == transA);
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

        my_dgemv(layout, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

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

            my_dgemv(layout, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

            inDerivative = true;
            // dC = alpha * X * transpose(Y) + A
            cblas_dger(layout, M, N, alpha, trans ? B : dC, trans ? incB : incC, trans ? dC : B, trans ? incC : incB, dA, lda);

            // dB = alpha * trans(A) * dC + dB
            cblas_dgemv(layout, transpose(transA), M, N, alpha, A, lda, dC, incC, 1.0, dB, incB); 

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

            ow_dgemv(layout, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

            inDerivative = true;
            // dC = alpha * X * transpose(Y) + A
            cblas_dger(layout, M, N, alpha,
							trans ? B_cache : dC, 
							trans ? 1 : incC, 
							trans ? dC : B_cache,
							trans ? incC : 1, dA,
							lda);

            // dB = alpha * trans(A) * dC + dB
            cblas_dgemv(layout, transpose(transA), M, N, alpha, A_cache, M, dC, incC, 1.0, dB, incB); 

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

static void gemmTests() {
  // N means normal matrix, T means transposed
  for (char layout : { CblasRowMajor, CblasColMajor }) {
  for (char transA : {'N', 'n', 'T', 't'}) {
  for (char transB : {'N', 'n', 'T', 't'}) {
  
    {

        bool transA_bool = !(transA == 'N' || transA == 'n');
        bool transB_bool = !(transB == 'N' || transB == 'n');
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
    my_dgemm(layout, transA, transB, M, N, K, alpha, A, lda, B, incB, beta, C, incC);

    assert(calls.size() == 1);
    assert(calls[0].inDerivative == false);
    assert(calls[0].type == CallType::GEMM);
    assert(calls[0].pout_arg1 == C);
    assert(calls[0].pin_arg1 == A);
    assert(calls[0].pin_arg2 == B);
    assert(calls[0].farg1 == alpha);
    assert(calls[0].farg2 == beta);
    assert(calls[0].layout == layout);
    assert(calls[0].targ1 == transA);
    assert(calls[0].targ2 == transB);
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

    
        my_dgemm(layout, transA, transB, M, N, K, alpha, A, lda, B, incB, beta, C, incC);

        inDerivative = true;

        // dA = 
        my_dgemm(layout,
                    transA_bool ? transB : 'N',
                    transA_bool ? 'T' : transpose(transB),
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
                    transB_bool ? 'T' : transpose(transA),
                    transB_bool ? transA : 'N', //transB,
                    transB_bool ? N : K,
                    transB_bool ? K : N,
                    M,
                    alpha,
                    transB_bool ? dC : A,
                    transB_bool ? incC : lda,
                    transB_bool ? A : dC,
                    transB_bool ? lda : incC,
                    1.0, dB, incB);
       
        cblas_dlascl(layout, 'G', 0, 0, 1.0, beta, M, N, dC, incC /*, extra 0*/ );
		
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

int main() {
   
  dotTests();

  gemvTests();

  gemmTests();

}
