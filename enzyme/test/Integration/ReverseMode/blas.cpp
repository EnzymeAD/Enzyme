// This should work on LLVM 7, 8, 9, however in CI the version of clang installed on Ubuntu 18.04 cannot load
// a clang plugin properly without segfaulting on exit. This is fine on Ubuntu 20.04 or later LLVM versions...
// RUN: %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli -
// RUN: %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli -
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme -mllvm -enzyme-inline=1 -S | %lli - ; fi

#include "test_utils.h"

#include <cassert>
#include <string>
#include <stdlib.h>
#include <string.h>

template <typename T> 
class vector {
    T* data;
    size_t capacity;
	size_t length; 
public:
    vector() : data(nullptr), capacity(0), length(0) {}
    vector(const vector &prev) : data((T*)malloc(sizeof(T)*prev.capacity)), capacity(prev.capacity), length(prev.length) {
		memcpy(data, prev.data, prev.length*sizeof(T));
	}
    void operator=(const vector &prev) {
 		free(data);
		data = (T*)malloc(sizeof(T)*prev.capacity);
		capacity = prev.capacity;
		length = prev.length;
		memcpy(data, prev.data, prev.length*sizeof(T));
    }
	// Don't destruct to avoi dso handle in global
    // ~vector() { free(data); }
 
    void push_back(T v) {
        if (length == capacity) {
			size_t next = capacity == 0 ? 1 : (2 * capacity);
			data = (T*)realloc(data, sizeof(T)*next);
			capacity = next;
        } 
        data[length] = v;
        length++;
    }
 
    T& operator[](size_t index) {
 		assert(index < length);
		return data[index];
    }
    
	const T& operator[] (size_t index) const {
 		assert(index < length);
		return data[index];
    }

	bool operator==(const vector& rhs) const {
		if (length != rhs.length) return false;
		for (size_t i=0; i<length; i++)
			if (data[i] != rhs.data[i]) return false;
		return true;
	}
	bool operator!=(const vector& rhs) const {
		return !(operator==(rhs));
	}
    size_t size() const { return length; }

	void clear() {
		length = 0;
	} 
};


bool inDerivative = false;

    char DEFAULT_LAYOUT = 0x72;
    
    double *UNUSED_POINTER = (double*)0x00000070;

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
double UNUSED_DOUBLE;

enum class CallType {
    GEMV,
    GEMM,
    SCAL,
    GER,
    COPY
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
	bool operator==(const BlasCall &rhs) const {
#define CHECK(A) if (A != rhs.A) return false;
		CHECK(inDerivative)
		CHECK(type)
		CHECK(pout_arg1)
		CHECK(pin_arg1)
		CHECK(pout_arg1)
		CHECK(farg1)
		CHECK(farg2)
		CHECK(layout)
		CHECK(targ1)
		CHECK(targ2)
		CHECK(iarg1)
		CHECK(iarg2)
		CHECK(iarg3)
		CHECK(iarg4)
		CHECK(iarg5)
		CHECK(iarg6)
		return true;
	}
	bool operator!=(const BlasCall& rhs) const {
		return !(operator==(rhs));
	}
};


void printty(bool v) {
  if (v) printf("Reverse (true)");
  else   printf("Primal (false)");
}

void printty(CallType v) {
  switch (v) {
      case CallType::GEMV: printf("GEMV"); return;
      case CallType::GEMM: printf("GEMM"); return;
      case CallType::SCAL: printf("SCAL"); return;
      case CallType::GER: printf("GER"); return;
      case CallType::COPY: printf("COPY"); return;
      default: printf("UNKNOWN CALL (%d)", (int)v);
    }
}

void printty(void* v) {
    if (v == A) printf("A");
    else if (v == dA) printf("dA");
    else if (v == B) printf("B");
    else if (v == dB) printf("dB");
    else if (v == C) printf("C");
    else if (v == dC) printf("dC");
    else if (v == UNUSED_POINTER) printf("UNUSED_POINTER");
    else printf("Unknown pointer");
    printf(" (%p)", v);
}

void printty(int v) {
    if (v == incA) printf("incA");
    else if (v == incB) printf("incB");
    else if (v == incC) printf("incC");
    else if (v == M) printf("M");
    else if (v == N) printf("N");
    else if (v == lda) printf("lda");
    else if (v == UNUSED_INT) printf("UNUSED_INT");
    else printf("Unknown int");
    printf(" (%d)", v);
}
void printty(char v) {
    if (v == DEFAULT_LAYOUT) {
		printf("DEFAULT_LAYOUT (0x%x)", v);
		return;
	}
    if (v == UNUSED_TRANS) {
		printf("UNUSED_TRANS (%c)", v);
		return;
	}
	for (auto t : {'N', 'n', 'T', 't'}) {
		if (v == t) {
			printf("'%c'", v);
			return;
		}
	}
    printf("Unknown char ('%c'=0x%x)", v, v);
}
void printty(double v) {
    if (v == alpha) printf("alpha");
    else if (v == beta) printf("beta");
    else if (v == UNUSED_DOUBLE) printf("UNUSED_DOUBLE");
    else printf("Unknown double");
    printf(" (%f)", v);
}

void printcall(BlasCall rcall) {
  switch (rcall.type) {
      case CallType::GEMV:
		printf("GEMV(layout=");
		printty(rcall.layout);
		printf(", trans=");
		printty(rcall.targ1);
		printf(", M=");
		printty(rcall.iarg1);
		printf(", N=");
		printty(rcall.iarg2);
		printf(", alpha=");
		printty(rcall.farg1);
		printf(", A=");
		printty(rcall.pin_arg1);
		printf(", lda=");
		printty(rcall.iarg4);
		printf(", X=");
		printty(rcall.pin_arg2);
		printf(", incx=");
		printty(rcall.iarg5);
		printf(", beta=");
		printty(rcall.farg2);
		printf(", Y=");
		printty(rcall.pout_arg1);
		printf(", incy=");
		printty(rcall.iarg6);
		printf(")");
		return;
      case CallType::GEMM:
		printf("GEMM(layout=");
		printty(rcall.layout);
		printf(", transA=");
		printty(rcall.targ1);
		printf(", transB=");
		printty(rcall.targ2);
		printf(", M=");
		printty(rcall.iarg1);
		printf(", N=");
		printty(rcall.iarg2);
		printf(", K=");
		printty(rcall.iarg3);
		printf(", alpha=");
		printty(rcall.farg1);
		printf(", A=");
		printty(rcall.pin_arg1);
		printf(", lda=");
		printty(rcall.iarg4);
		printf(", B=");
		printty(rcall.pin_arg2);
		printf(", ldb=");
		printty(rcall.iarg5);
		printf(", beta=");
		printty(rcall.farg2);
		printf(", C=");
		printty(rcall.pout_arg1);
		printf(", ldc=");
		printty(rcall.iarg6);
		printf(")");
		return;
      case CallType::SCAL:
		printf("SCAL(N=");
		printty(rcall.iarg1);
		printf(", alpha=");
		printty(rcall.farg1);
		printf(", X=");
		printty(rcall.pout_arg1);
		printf(", incX=");
		printty(rcall.iarg4);
		printf(")");
		return;
      case CallType::GER:
		printf("GER(layout=");
		printty(rcall.layout);
		printf(", M=");
		printty(rcall.iarg1);
		printf(", N=");
		printty(rcall.iarg2);
		printf(", alpha=");
		printty(rcall.farg1);
		printf(", X=");
		printty(rcall.pin_arg1);
		printf(", incX=");
		printty(rcall.iarg4);
		printf(", Y=");
		printty(rcall.pin_arg2);
		printf(", incY=");
		printty(rcall.iarg5);
		printf(", A=");
		printty(rcall.pout_arg1);
		printf(", lda=");
		printty(rcall.iarg6);
		printf(")");
		return;
      case CallType::COPY:
		printf("COPY(N=");
		printty(rcall.iarg1);
		printf(", X=");
		printty(rcall.pin_arg1);
		printf(", incX=");
		printty(rcall.iarg4);
		printf(", Y=");
		printty(rcall.pout_arg1);
		printf(", incY=");
		printty(rcall.iarg5);
		printf(")");
		return;
      default: printf("UNKNOWN CALL (%d)", (int)rcall.type); return;
    }
}

void printTrace(const vector<BlasCall> &tr, std::string prefix="") {
	printf("%sPrimal:\n", prefix.c_str());
	bool reverse = false;
	for (size_t i=0; i<tr.size(); i++) {
		if (tr[i].inDerivative) {
			if (!reverse) {
				printf("%sReverse:\n", prefix.c_str());
				reverse = true;
			}
		} else
			assert(!reverse);
		printf("%s  %zu:\t", prefix.c_str(), i);
		printcall(tr[i]);
		printf("\n");
	}
}

template<typename T>
void assert_eq(std::string scope, std::string varName, int i, T expected, T real, BlasCall texpected, BlasCall rcall) {
    if (expected == real) return;
    printf("Failure on call %d var %s found ", i, varName.c_str());
    printty(expected);
    printf(" expected ");
    printty(real);
    printf("\n");
    assert(0);
    exit(1);
}

void check_equiv(std::string scope, int i, BlasCall expected, BlasCall real) {
#define MAKEASSERT(name) assert_eq(scope, #name, i, expected.name, real.name, expected, real);
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

vector<BlasCall> calls;
vector<BlasCall> foundCalls;

extern "C" {

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
                                X, UNUSED_POINTER, UNUSED_POINTER,
                                alpha, UNUSED_DOUBLE,
                                UNUSED_TRANS,
                                UNUSED_TRANS, UNUSED_TRANS,
                                N, UNUSED_INT, UNUSED_INT, incX, UNUSED_INT, UNUSED_INT});
}

// A = alpha * X * transpose(Y) + A
__attribute__((noinline))
void cblas_dger(char layout, int M, int N, double alpha, double* X, int incX, double* Y, int incY, double* A, int lda) {
    calls.push_back((BlasCall){inDerivative, CallType::GER,
                                A, X, Y,
                                alpha, UNUSED_DOUBLE,
                                layout,
                                UNUSED_TRANS, UNUSED_TRANS,
                                M, N, UNUSED_INT, incX, incY, lda});
}

__attribute__((noinline))
void cblas_dcopy(int N, double* X, int incX, double* Y, int incY) {

    calls.push_back((BlasCall){inDerivative, CallType::COPY,
                                Y, X, UNUSED_POINTER,
                                alpha, UNUSED_DOUBLE,
                                UNUSED_TRANS,
                                UNUSED_TRANS, UNUSED_TRANS,
                                N, UNUSED_INT, UNUSED_INT, incX, incY, UNUSED_INT});
}
}

int enzyme_dup;
int enzyme_out;
int enzyme_const;
template<typename ...T>
void __enzyme_autodiff(void*, T...);

void my_dgemv(char layout, char trans, int M, int N, double alpha, double* __restrict__ A, int lda, double* __restrict__ X, int incx, double beta, double* __restrict__ Y, int incy) {
    cblas_dgemv(layout, trans, M, N, alpha, A, lda, X, incx, beta, Y, incy);
    inDerivative = true;
}

void init() {
    inDerivative = false;
    calls.clear();
}

void checkTest(std::string name) {
        if (foundCalls.size() != calls.size()) {
            printf("Test %s failed: Expected %zu calls, found %zu\n", name.c_str(), calls.size(), foundCalls.size());
			printf("Expected:\n");
			printTrace(calls, "  ");
			printf("Found:\n");
			printTrace(foundCalls, "  ");
            assert(0 && "non-equal call count");
            exit(1);
        }
		if (foundCalls != calls) {
            printf("Test %s failed\n", name.c_str());
			printf("Expected:\n");
			printTrace(calls, "  ");
			printf("Found:\n");
			printTrace(foundCalls, "  ");
		}
        for (size_t i=0; i<calls.size(); i++) {
            check_equiv(name, i, foundCalls[i], calls[i]);
        }
}

int main() {

  // N means normal matrix, T means transposed
  for (char transA : {'N', 'n', 'T', 't'}) {
  
    init();
    my_dgemv(DEFAULT_LAYOUT, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

    {
    assert(calls.size() == 1);
    assert(calls[0].inDerivative == false);
    assert(calls[0].type == CallType::GEMV);
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
        foundCalls = calls;
        init();

        my_dgemv(DEFAULT_LAYOUT, transA, M, N, alpha, A, lda, B, incB, beta, C, incC);

        inDerivative = true;
        // dC = alpha * X * transpose(Y) + A
        cblas_dger(DEFAULT_LAYOUT, M, N, alpha, dC, incC, B, incB, dA, lda);
        // dY = beta * dY
        cblas_dscal((transA == 'N' || transA == 'n') ? M : N,
                    beta, dC, incC);

		checkTest("GEMV active A, C");
    }


  }


}
