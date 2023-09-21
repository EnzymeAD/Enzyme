
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

char CblasRowMajor = 101;
char CblasColMajor = 102;

bool inDerivative = false;

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

    int M =  105688;
    int N =   78412;
    int K = 5013424;
    int lda = 3416;
char UNUSED_TRANS = 'A';
int UNUSED_INT = -1;
double UNUSED_DOUBLE;

enum class CallType {
    GEMV,
    GEMM,
    SCAL,
    GER,
    DOT,
    AXPY,
    LASCL,
    COPY,
    LACPY,
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
      case CallType::LACPY: printf("LACPY"); return;
      case CallType::DOT: printf("DOT"); return;
      case CallType::AXPY: printf("AXPY"); return;
      case CallType::LASCL: printf("LASCL"); return;
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
    else if (v == K) printf("K");
    else if (v == lda) printf("lda");
    else if (v == UNUSED_INT) printf("UNUSED_INT");
    else printf("Unknown int");
    printf(" (%d)", v);
}
void printty(char v) {
    if (v == CblasRowMajor) {
        printf("RowMajor (%d)", v);
        return;
    }
    if (v == CblasColMajor) {
        printf("RowMajor (%d)", v);
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
      case CallType::LACPY:
		printf("LACPY(layout=");
		printty(rcall.layout);
		printf(", uplo=");
		printty(rcall.targ1);
        printf(", M=");
		printty(rcall.iarg1);
		printf(", N=");
		printty(rcall.iarg2);
		printf(", A=");
		printty(rcall.pin_arg1);
		printf(", lda=");
		printty(rcall.iarg4);
		printf(", B=");
		printty(rcall.pout_arg1);
		printf(", ldb=");
		printty(rcall.iarg5);
		printf(")");
		return;
      case CallType::LASCL:
		printf("LASCL(layout=");
		printty(rcall.layout);
		printf(", type=");
		printty(rcall.targ1);
		printf(", KL=");
		printty(rcall.iarg5);
		printf(", KU=");
		printty(rcall.iarg6);
		printf(", cfrom=");
		printty(rcall.farg1);
		printf(", cto=");
		printty(rcall.farg2);
		
        printf(", M=");
		printty(rcall.iarg1);
		printf(", N=");
		printty(rcall.iarg2);
		printf(", A=");
		printty(rcall.pout_arg1);
		printf(", lda=");
		printty(rcall.iarg4);
		printf(")");
		return;
      case CallType::AXPY:
		printf("DOT(N=");
		printty(rcall.iarg1);
		printf(", alpha=");
		printty(rcall.farg1);
		printf(", X=");
		printty(rcall.pin_arg1);
		printf(", incx=");
		printty(rcall.iarg4);
		printf(", Y=");
		printty(rcall.pout_arg1);
		printf(", incy=");
		printty(rcall.iarg5);
		printf(")");
		return;
      case CallType::DOT:
		printf("DOT(N=");
		printty(rcall.iarg1);
		printf(", X=");
		printty(rcall.pin_arg1);
		printf(", incx=");
		printty(rcall.iarg4);
		printf(", Y=");
		printty(rcall.pin_arg2);
		printf(", incy=");
		printty(rcall.iarg5);
		printf(")");
		return;
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

// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/lascl.html
// technically LAPACKE_dlascl
__attribute__((noinline))
void cblas_dlascl(char layout, char type, int KL, int KU, double cfrom, double cto, int M, int N, double* A, int lda) {
    BlasCall call = {inDerivative, CallType::LASCL,
                                A, UNUSED_POINTER, UNUSED_POINTER,
                                cfrom, cto,
                                layout,
                                type, UNUSED_TRANS,
                                M, N, UNUSED_INT, lda, KL, KU};
    calls.push_back(call);
}

__attribute__((noinline))
double cblas_ddot(int N, double* X, int incx, double* Y, int incy) {
    BlasCall call = {inDerivative, CallType::DOT,
                                UNUSED_POINTER, X, Y,
                                UNUSED_DOUBLE, UNUSED_DOUBLE,
                                UNUSED_TRANS,
                                UNUSED_TRANS, UNUSED_TRANS,
                                N, UNUSED_INT, UNUSED_INT, incx, incy, UNUSED_INT};
    calls.push_back(call);
    return 3.15+N;
}

// Y += alpha * X
__attribute__((noinline))
void cblas_daxpy(int N, double alpha, double* X, int incx, double* Y, int incy) {
    BlasCall call = {inDerivative, CallType::AXPY,
                                Y, X, UNUSED_POINTER,
                                alpha, UNUSED_DOUBLE,
                                UNUSED_TRANS,
                                UNUSED_TRANS, UNUSED_TRANS,
                                N, UNUSED_INT, UNUSED_INT, incx, incy, UNUSED_INT};
    calls.push_back(call);
}

// Y = alpha * op(A) * X + beta * Y
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

__attribute__((noinline))
void cblas_dlacpy(char layout, char uplo, int M, int N, double* A, int lda, double* B, int ldb) {
    calls.push_back((BlasCall){inDerivative, CallType::LACPY,
                                B, A, UNUSED_POINTER,
                                UNUSED_DOUBLE, UNUSED_DOUBLE,
                                layout,
                                uplo, UNUSED_TRANS,
                                M, N, UNUSED_INT, lda, ldb, UNUSED_INT});
}

__attribute__((noinline))
void dlacpy(char *uplo, int *M, int* N, double* A, int *lda, double* B, int* ldb) {
  cblas_dlacpy(CblasColMajor, *uplo, *M, *N, A, *lda, B, *ldb);
}

}

enum class ValueType {
    Matrix,
    Vector
};
struct BlasInfo {
	void* ptr;
    ValueType ty;
    int vec_length;
    int vec_increment;
    char mat_layout;
    int mat_rows;
    int mat_cols;
    int mat_ld;
    BlasInfo (void* v_ptr, int length, int increment) {
		ptr = v_ptr;
        ty = ValueType::Vector;
        vec_length = length;
        vec_increment = increment;
        mat_layout = '@';
        mat_rows = -1;
        mat_cols = -1;
        mat_ld = -1;
    }
    BlasInfo (void* v_ptr, char layout, int rows, int cols, int ld) {
		ptr = v_ptr;
        ty = ValueType::Matrix;
        vec_length = -1;
        vec_increment = -1;
        mat_layout = layout;
        mat_rows = rows;
        mat_cols = cols;
        mat_ld = ld;
    }
	BlasInfo () {
		ptr = (void*)(-1);
        ty = ValueType::Matrix;
        vec_length = -1;
        vec_increment = -1;
        mat_layout = -1;
        mat_rows = -1;
        mat_cols = -1;
        mat_ld = -1;
	}
};

BlasInfo pointer_to_index(void* v, BlasInfo inputs[6]) {
    if (v == A || v == dA) return inputs[0];
    if (v == B || v == dB) return inputs[1];
    if (v == C || v == dC) return inputs[2];
	for (int i=3; i<6; i++)
		if (inputs[i].ptr == v)
			return inputs[i];
    assert(0 && " illegal pointer to invert");
}

void checkVector(BlasInfo info, std::string vecname, int length, int increment, std::string test, BlasCall rcall, const vector<BlasCall> & trace) {
  if (info.ty != ValueType::Vector) {
    printf("Error in test %s, invalid memory\n", test.c_str());
    printTrace(trace);
    printcall(rcall);
    printf(" Input %s is not a vector\n", vecname.c_str());
    exit(1);
  }
  if (info.vec_length != length) {
        printf("Error in test %s, invalid memory\n", test.c_str());
        printTrace(trace);
        printcall(rcall);
        printf(" Input %s length must be ", vecname.c_str());
        printty(info.vec_length);
        printf(" found ");
        printty(length);
        printf("\n");
        exit(1);
  }
  if (info.vec_increment != increment) {
        printf("Error in test %s, invalid memory\n", test.c_str());
        printTrace(trace);
        printcall(rcall);
        printf(" Input %s increment must be ", vecname.c_str());
        printty(info.vec_increment);
        printf(" found ");
        printty(increment);
        printf("\n");
        exit(1);
  }
}

void checkMatrix(BlasInfo info, std::string matname, char layout, int rows, int cols, int ld, std::string test, BlasCall rcall, const vector<BlasCall> & trace) {
  if (info.ty != ValueType::Matrix) {
    printf("Error in test %s, invalid memory\n", test.c_str());
    printTrace(trace);
    printcall(rcall);
    printf(" Input %s is not a matrix\n", matname.c_str());
    exit(1);
  }
  if (info.mat_layout != layout) {
    printf("Error in test %s, invalid memory\n", test.c_str());
    printTrace(trace);
    printcall(rcall);
    printf(" Input %s layout must be ", matname.c_str());
    printty(info.mat_layout);
    printf(" found layout=");
    printty(layout);
    printf("\n");
    exit(1);
  }
  if (info.mat_rows != rows) {
    printf("Error in test %s, invalid memory\n", test.c_str());
    printTrace(trace);
    printcall(rcall);
    printf(" Input %s rows must be ", matname.c_str());
    printty(info.mat_rows);
    printf(" found ");
    printty(rows);
    printf("\n");
    exit(1);
  }
  if (info.mat_cols != cols) {
    printf("Error in test %s, invalid memory\n", test.c_str());
    printTrace(trace);
    printcall(rcall);
    printf(" Input %s cols must be ", matname.c_str());
    printty(info.mat_cols);
    printf(" found ");
    printty(cols);
    printf("\n");
    exit(1);
  }
  if (info.mat_ld != ld) {
    printf("Error in test %s, invalid memory\n", test.c_str());
    printTrace(trace);
    printcall(rcall);
    printf(" Input %s leading dimension rows must be ", test.c_str());
    printty(info.mat_ld);
    printf(" found ");
    printty(ld);
    printf("\n");
    exit(1);
  }
}

void checkMemory(BlasCall rcall, BlasInfo inputs[6], std::string test, const vector<BlasCall> & trace) {
  switch (rcall.type) {
		return;
      case CallType::LASCL: {
        auto A = pointer_to_index(rcall.pout_arg1, inputs);
        
		auto layout = rcall.layout;
		auto type = rcall.targ1;
		auto KL = rcall.iarg5;
		auto KU = rcall.iarg6;
		auto cfrom = rcall.farg1;
        auto cto = rcall.farg2;
		
		auto M = rcall.iarg1;
		auto N = rcall.iarg2;
		auto lda = rcall.iarg4;
	
        // = 'G': A is a full matrix.
        assert(type == 'G');
        
        // A is an m-by-n matrix
        checkMatrix(A, "A", layout, /*rows=*/M, /*cols=*/N, /*ld=*/lda, test, rcall, trace);
		return;
      }
      case CallType::AXPY: {
        auto Y = pointer_to_index(rcall.pout_arg1, inputs);
        
        auto X = pointer_to_index(rcall.pin_arg1, inputs);
		
		auto alpha = rcall.farg1;
        
        auto N = rcall.iarg1;
		auto incX = rcall.iarg4;
		auto incY = rcall.iarg5;
        
        checkVector(X, "X", /*len=*/N, /*inc=*/incX, test, rcall, trace);
        checkVector(Y, "Y", /*len=*/N, /*inc=*/incY, test, rcall, trace);
		return;
      }
      case CallType::DOT: {
        auto X = pointer_to_index(rcall.pin_arg1, inputs);
        auto Y = pointer_to_index(rcall.pin_arg2, inputs);
		
        auto N = rcall.iarg1;
		auto incX = rcall.iarg4;
		auto incY = rcall.iarg5;
        
        checkVector(X, "X", /*len=*/N, /*inc=*/incX, test, rcall, trace);
        checkVector(Y, "Y", /*len=*/N, /*inc=*/incY, test, rcall, trace);
		return;
      }
      case CallType::GEMV:{
        // Y = alpha * op(A) * X + beta * Y
        auto Y = pointer_to_index(rcall.pout_arg1, inputs);
        auto A = pointer_to_index(rcall.pin_arg1, inputs);
        auto X = pointer_to_index(rcall.pin_arg2, inputs);
		
        auto layout = rcall.layout;
		auto trans_char = rcall.targ1;
        auto trans = !(trans_char == 'N' || trans_char == 'n');
		auto M = rcall.iarg1;
		auto N =rcall.iarg2;
		auto alpha = rcall.farg1;
		auto lda = rcall.iarg4;
		auto incX = rcall.iarg5;
		auto beta = rcall.farg2;
		auto incY = rcall.iarg6;

        // A is an m-by-n matrix
        checkMatrix(A, "A", layout, /*rows=*/M, /*cols=*/N, /*ld=*/lda, test, rcall, trace);

        // if no trans, X must be N otherwise must be M
        // From https://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9.html
        //    X is DOUBLE PRECISION array, dimension at least
        //   ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
        //   and at least
        //   ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
        //   Before entry, the incremented array X must contain the
        //   vector x.
        auto Xlen = trans ? M : N;
        checkVector(X, "X", /*len=*/Xlen, /*inc=*/incX, test, rcall, trace);
        
        // if no trans, Y must be M otherwise must be N
        auto Ylen = trans ? N : M;
        checkVector(Y, "Y", /*len=*/Ylen, /*inc=*/incY, test, rcall, trace);

		return;
      }
      case CallType::GEMM:{
        // C = alpha * A^transA * B^transB + beta * C
        auto C = pointer_to_index(rcall.pout_arg1, inputs);
        auto A = pointer_to_index(rcall.pin_arg1, inputs);
        auto B = pointer_to_index(rcall.pin_arg2, inputs);
	
        auto layout = rcall.layout;
		auto transA_char = rcall.targ1;
        auto transA = !(transA_char == 'N' || transA_char == 'n');
		auto transB_char = rcall.targ2;
        auto transB = !(transB_char == 'N' || transB_char == 'n');
		auto M = rcall.iarg1;
		auto N = rcall.iarg2;
		auto K = rcall.iarg3;
		auto alpha = rcall.farg1;
		auto lda = rcall.iarg4;
		auto ldb = rcall.iarg5;
		auto beta = rcall.farg2;
		auto ldc = rcall.iarg6;

        // From https://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
        /*
            M is INTEGER
           On entry,  M  specifies  the number  of rows  of the  matrix
           op( A )  and of the  matrix  C.  M  must  be at least  zero.
          N is INTEGER
           On entry,  N  specifies the number  of columns of the matrix
           op( B ) and the number of columns of the matrix C. N must be
           at least zero.
          K is INTEGER
           On entry,  K  specifies  the number of columns of the matrix
           op( A ) and the number of rows of the matrix op( B ). K must
           be at least  zero.
          LDA is INTEGER
           On entry, LDA specifies the first dimension of A as declared
           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
           LDA must be at least  max( 1, m ), otherwise  LDA must be at
           least  max( 1, k ).
        */
        checkMatrix(A, "A", layout, /*rows=*/(!transA) ? M : K, /*cols=*/(!transA) ? K : M, /*ld=*/lda, test, rcall, trace);
        checkMatrix(B, "B", layout, /*rows=*/(!transB) ? K : N, /*cols=*/(!transB) ? N : K, /*ld=*/ldb, test, rcall, trace);
        checkMatrix(C, "C", layout, /*rows=*/M, /*cols=*/N, /*ld=*/ldc, test, rcall, trace);
		return;
      }

      case CallType::SCAL: {
		auto N = rcall.iarg1;
		auto alpha = rcall.farg1;
        auto X = pointer_to_index(rcall.pout_arg1, inputs);
		auto incX = rcall.iarg4;
        checkVector(X, "X", /*len=*/N, /*inc=*/incX, test, rcall, trace);
		return;
      }
      case CallType::GER: {
        // A = alpha * X * transpose(Y) + A
        auto A = pointer_to_index(rcall.pout_arg1, inputs);
        auto X = pointer_to_index(rcall.pin_arg1, inputs);
        auto Y = pointer_to_index(rcall.pin_arg2, inputs);

		auto layout = rcall.layout;
		auto M = rcall.iarg1;
	    auto N = rcall.iarg2;
		auto alpha = rcall.farg1;
		auto incX = rcall.iarg4;
		auto incY = rcall.iarg5;
		auto incA = rcall.iarg6;

        // From https://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga458222e01b4d348e9b52b9343d52f828.html
        // x is an m element vector, y is an n element
        // vector and A is an m by n matrix.
        checkVector(X, "X", /*len=*/M, /*inc=*/incX, test, rcall, trace);
        checkVector(Y, "Y", /*len=*/N, /*inc=*/incY, test, rcall, trace);
        checkMatrix(A, "A", layout, /*rows=*/M, /*cols=*/N, /*ld=*/incA, test, rcall, trace);
		return;
      }
      case CallType::COPY: {
        auto Y = pointer_to_index(rcall.pout_arg1, inputs);
        auto X = pointer_to_index(rcall.pin_arg1, inputs);
        
		auto N = rcall.iarg1;
		auto incX = rcall.iarg4;
		auto incY = rcall.iarg5;
        checkVector(X, "X", /*len=*/N, /*inc=*/incX, test, rcall, trace);
        checkVector(Y, "Y", /*len=*/N, /*inc=*/incY, test, rcall, trace);
		return;
      }
      case CallType::LACPY: {
        auto B = pointer_to_index(rcall.pout_arg1, inputs);
        auto A = pointer_to_index(rcall.pin_arg1, inputs);
        
        auto layout = rcall.layout;
        auto uplo = rcall.targ1;
        auto M = rcall.iarg1;
		auto N = rcall.iarg2;
		auto lda = rcall.iarg4;
		auto ldb = rcall.iarg5;
        checkMatrix(A, "A", layout, /*rows=*/M, /*cols=*/N, /*ld=*/lda, test, rcall, trace);
        checkMatrix(B, "B", layout, /*rows=*/M, /*cols=*/N, /*ld=*/ldb, test, rcall, trace);
		return;
      }
      default: printf("UNKNOWN CALL (%d)", (int)rcall.type); return;
    }
}

void checkMemoryTrace(BlasInfo inputs[6], std::string test, const vector<BlasCall> & trace) {
    for (size_t i=0; i<trace.size(); i++)
        checkMemory(trace[i], inputs, test, trace);
}

void init() {
    inDerivative = false;
    calls.clear();
}

char transpose(char c) {
    switch (c) {
        case 'N': return 'T';
        case 'n': return 't';
        case 'T': return 'N';
        case 't': return 'n';
        default:
      printf("Illegal transpose of '%c'\n", c);
      exit(1);
    }
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
