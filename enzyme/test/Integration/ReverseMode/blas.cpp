// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S
// RUN: %clang++ -std=c++11 -fno-exceptions -ffast-math -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S

#include "test_utils.h"

#include <map>
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

bool inDerivative = false;

__attribute__((noinline))
__attribute__((enzyme_inactive))
void swapToDerivative() {
    inDerivative = true;
}


    char DEFAULT_LAYOUT = 0x72;
    
    double *A = (double*)0x00000100;
    double *dA = (double*)0x00000700;
    int incA = 1234;

    double *B = (double*)0x00010000;
    double *dB = (double*)0x00070000;
    int incB = 5678;

    double *C = (double*)0x01000000;
    double *dC = (double*)0x07000000;
    int incC = 91011;

    double alpha = 2.71828;
    double beta = 47.56;

    int M = 105688;
    int N =  78412;
    int lda = 3416;
char UNUSED_TRANS = 'A';
int UNUSED_INT = -1;

enum class CallType {
    GEMV,
    GEMM,
    SCAL,
    GER
};

struct BlasCall {
    bool inDerivative;
    CallType type;
    void* pout_arg1;
    void* pin_arg1;
    void* pin_arg2;
    double farg1;
    double farg2;
    char layout;
    char targ1;
    char targ2;
    int iarg1;
    int iarg2;
    int iarg3;
    int iarg4;
    int iarg5;
    int iarg6;
};


std::string astr(bool v) {
  if (v) return "Reverse (true)";
  else   return "Primal (false)";
}

std::string astr(CallType v) {
  switch (v) {
      case CallType::GEMV: return "GEMV";
      case CallType::GEMM: return "GEMM";
      case CallType::SCAL: return "SCAL";
      case CallType::GER: return "GER";
      default: return "UNKNOWN_CALL";
    }
}

namespace std {
std::string to_string(void* v) {
        std::ostringstream address;
        address << (void const *)v;
        return address.str();
    }
}

std::string astr(void* v) {
    if (v == A) return "A (" + std::to_string((double*)v) + ")";
    if (v == dA) return "dA (" + std::to_string(v) + ")";
    if (v == B) return "B (" + std::to_string(v) + ")";
    if (v == dB) return "dB (" + std::to_string(v) + ")";
    if (v == C) return "C (" + std::to_string(v) + ")";
    if (v == dC) return "dC (" + std::to_string(v) + ")";
    return "Unknown pointer (" + std::to_string(v) + ")";
}

std::string astr(int v) {
    if (v == incA) return "incA (" + std::to_string(v) + ")";
    if (v == incB) return "incB (" + std::to_string(v) + ")";
    if (v == incC) return "incC (" + std::to_string(v) + ")";
    if (v == M) return "M (" + std::to_string(v) + ")";
    if (v == N) return "N (" + std::to_string(v) + ")";
    if (v == lda) return "lda (" + std::to_string(v) + ")";
    if (v == UNUSED_INT) return "UNUSED_INT (" + std::to_string(v) + ")";
    return "Unknown int (" + std::to_string(v) + ")";
}
std::string astr(double v) {
    if (v == alpha) return "alpha (" + std::to_string(v) + ")";
    if (v == beta) return "beta (" + std::to_string(v) + ")";
    return "Unknown double (" + std::to_string(v) + ")";
}

template<typename T>
void assert_eq(std::string scope, std::string varName, int i, T expected, T real) {
    if (expected == real) return;
    std::cerr << "Failure on test: " << scope << " call: " << i << ", var " << varName << ":\n";
    std::cerr << "  found, " << astr(expected) << " expected " << astr(real) << "\n";
    assert(0);
    exit(1);
}

void check_equiv(std::string scope, int i, BlasCall expected, BlasCall real) {
#define MAKEASSERT(name) assert_eq(scope, #name, i, expected.name, real.name);
    MAKEASSERT(inDerivative)
    MAKEASSERT(type)
    MAKEASSERT(pout_arg1);
    MAKEASSERT(pin_arg1);
    MAKEASSERT(pin_arg2);
    MAKEASSERT(farg1);
    MAKEASSERT(farg2);
    MAKEASSERT(layout);
    MAKEASSERT(targ1);
    MAKEASSERT(targ2);
    MAKEASSERT(iarg1);
    MAKEASSERT(iarg2);
    MAKEASSERT(iarg3);
    MAKEASSERT(iarg4);
    MAKEASSERT(iarg5);
    MAKEASSERT(iarg6);
}

std::vector<BlasCall> calls;

// Y = alpha * A * X + beta * Y
__attribute__((noinline))
void cblas_dgemv(char layout, char trans, int M, int N, double alpha, double* A, int lda, double* X, int incx, double beta, double* Y, int incy) {
    BlasCall call = {inDerivative, CallType::GEMV,
                                Y, A, X,
                                alpha, beta,
                                layout,
                                trans, UNUSED_TRANS,
                                M, N, UNUSED_INT, lda, incx, incy};
    calls.push_back(call);
}

// C = alpha * A^transA * B^transB + beta * C
__attribute__((noinline))
void cblas_dgemm(char layout, char transA, char transB, int M, int N, int K, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc) {
    calls.push_back((BlasCall){inDerivative, CallType::GEMM,
                                C, A, B,
                                alpha, beta,
                                layout,
                                transA, transB,
                                M, N, K, lda, ldb, ldc});
}

// X = alpha * X
__attribute__((noinline))
void cblas_dscal(int N, double alpha, double* X, int incX) {
    calls.push_back((BlasCall){inDerivative, CallType::SCAL,
                                X, 0, 0,
                                alpha, 3.14159,
                                UNUSED_TRANS,
                                UNUSED_TRANS, UNUSED_TRANS,
                                N, UNUSED_INT, UNUSED_INT, incX, UNUSED_INT, UNUSED_INT});
}

// A = alpha * X * transpose(Y) + A
__attribute__((noinline))
void cblas_dger(char layout, int M, int N, double alpha, double* X, int incX, double* Y, int incY, double* A, int lda) {
    calls.push_back((BlasCall){inDerivative, CallType::SCAL,
                                X, 0, 0,
                                alpha, 3.14159,
                                UNUSED_TRANS,
                                UNUSED_TRANS, UNUSED_TRANS,
                                N, UNUSED_INT, UNUSED_INT, incX, UNUSED_INT, UNUSED_INT});
}

int enzyme_dup;
int enzyme_out;
int enzyme_const;
void __enzyme_autodiff(void*, ...);

void my_dgemv(char layout, char trans, int M, int N, double alpha, double* A, int lda, double* X, int incx, double beta, double* Y, int incy) {
    cblas_dgemv(layout, trans, M, N, alpha, A, lda, X, incx, beta, Y, incy);
    swapToDerivative();
}

void init() {
    inDerivative = false;
    calls.clear();
}

int main() {

  // N means normal matrix, T means transposed
  for (char transA : {'N', 'n', 'T', 't'}) {
  
    init();
    my_dgemv(DEFAULT_LAYOUT, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

    {
    assert(calls.size() == 1);
    assert(calls[0].inDerivative = false);
    assert(calls[0].type == CallType::GEMM);
    assert(calls[0].pout_arg1 == C);
    assert(calls[0].pin_arg1 == A);
    assert(calls[0].pin_arg2 == B);
    assert(calls[0].farg1 == alpha);
    assert(calls[0].farg2 == beta);
    assert(calls[0].layout == DEFAULT_LAYOUT);
    assert(calls[0].targ1 == transA);
    assert(calls[0].targ2 == UNUSED_TRANS);
    assert(calls[0].iarg1 == M);
    assert(calls[0].iarg2 == N);
    assert(calls[0].iarg3 == UNUSED_INT);
    assert(calls[0].iarg4 == lda);
    assert(calls[0].iarg5 == incB);
    assert(calls[0].iarg6 == incC);
    }

    init();
    __enzyme_autodiff((void*) my_dgemv,
                            enzyme_const, DEFAULT_LAYOUT,
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
    {
        auto foundCalls = calls;
        init();

        my_dgemv(DEFAULT_LAYOUT, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

        inDerivative = true;
        // dC = alpha * X * transpose(Y) + A
        cblas_dger(DEFAULT_LAYOUT, M, N, alpha, dC, incC, B, incB, dA, lda);
        // dY = beta * dY
        cblas_dscal((transA == 'N' || transA == 'n') ? M : N,
                    beta, dC, incC);

        assert(foundCalls.size() == calls.size());
        for (size_t i=0; i<calls.size(); i++) {
            check_equiv("GEMV active A, C", i, foundCalls[i], calls[i]);
        }
    }


  }


}
