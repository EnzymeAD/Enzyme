#include <cassert>
#include <stdlib.h>
#include <string.h>
#include <string>

template <typename T> class vector {
  T *data;
  size_t capacity;
  size_t length;

public:
  vector() : data(nullptr), capacity(0), length(0) {}
  vector(const vector &prev)
      : data((T *)malloc(sizeof(T) * prev.capacity)), capacity(prev.capacity),
        length(prev.length) {
    memcpy(data, prev.data, prev.length * sizeof(T));
  }
  void operator=(const vector &prev) {
    free(data);
    data = (T *)malloc(sizeof(T) * prev.capacity);
    capacity = prev.capacity;
    length = prev.length;
    memcpy(data, prev.data, prev.length * sizeof(T));
  }
  // Don't destruct to avoi dso handle in global
  // ~vector() { free(data); }

  void push_back(T v) {
    if (length == capacity) {
      size_t next = capacity == 0 ? 1 : (2 * capacity);
      data = (T *)realloc(data, sizeof(T) * next);
      capacity = next;
    }
    data[length] = v;
    length++;
  }

  T &operator[](size_t index) {
    assert(index < length);
    return data[index];
  }

  const T &operator[](size_t index) const {
    assert(index < length);
    return data[index];
  }

  bool operator==(const vector &rhs) const {
    if (length != rhs.length)
      return false;
    for (size_t i = 0; i < length; i++)
      if (data[i] != rhs.data[i])
        return false;
    return true;
  }
  bool operator!=(const vector &rhs) const { return !(operator==(rhs)); }
  size_t size() const { return length; }

  void clear() { length = 0; }
};

enum class cublasStatus_t {
  CUBLAS_STATUS_SUCCESS,
  CUBLAS_STATUS_NOT_INITIALIZED,
  CUBLAS_STATUS_ALLOC_FAILED,
  CUBLAS_STATUS_INVALID_VALUE,
  CUBLAS_STATUS_ARCH_MISMATCH,
  CUBLAS_STATUS_MAPPING_ERROR,
  CUBLAS_STATUS_EXECUTION_FAILED,
  CUBLAS_STATUS_INTERNAL_ERROR,
  CUBLAS_STATUS_NOT_SUPPORTED,
  CUBLAS_STATUS_LICENSE_ERROR,
};
enum class cublasOperation_t {
  CUBLAS_OP_N,
  CUBLAS_OP_T,
  CUBLAS_OP_C,
  CUBLAS_OP_UNUSED,
};
bool is_normal(cublasOperation_t op) {
  switch (op) {
  case cublasOperation_t::CUBLAS_OP_N:
    return true;
  case cublasOperation_t::CUBLAS_OP_T:
    return false;
  default:
    printf("Illegal transpose of '%c'\n", op);
    exit(1);
  }
}
cublasOperation_t transpose(cublasOperation_t op) {
  switch (op) {
  case cublasOperation_t::CUBLAS_OP_N:
    return cublasOperation_t::CUBLAS_OP_T;
  case cublasOperation_t::CUBLAS_OP_T:
    return cublasOperation_t::CUBLAS_OP_N;
  default:
    printf("Illegal transpose of '%c'\n", op);
    exit(1);
  }
}
struct cublasHandle_t {};

bool inDerivative = false;

cublasHandle_t *UNUSED_CUBLAS_HANDLE = (cublasHandle_t *)0x00000124;
cublasHandle_t *USED_CUBLAS_HANDLE = (cublasHandle_t *)0x00000126;

double *UNUSED_POINTER = (double *)0x00000070;

double *A = (double *)0x00000100;
double *dA = (double *)0x00000700;
int incA = 1234;

double *B = (double *)0x00010000;
double *dB = (double *)0x00070000;
int incB = 5678;

double *C = (double *)0x01000000;
double *dC = (double *)0x07000000;
int incC = 91011;

double alpha = 2.71828;
double beta = 47.56;

int M = 105688;
int N = 78412;
int K = 5013424;
int lda = 3416;
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

struct CuBlasCall {
  bool inDerivative;
  CallType type;
  void *pout_arg1;
  void *pin_arg1;
  void *pin_arg2;
  double farg1;
  double farg2;
  cublasHandle_t *handle;
  cublasOperation_t op1;
  cublasOperation_t op2;
  int iarg1;
  int iarg2;
  int iarg3;
  int iarg4;
  int iarg5;
  int iarg6;
  double *result;
  bool operator==(const CuBlasCall &rhs) const {
#define CHECK(A)                                                               \
  if (A != rhs.A)                                                              \
    return false;
    CHECK(inDerivative)
    CHECK(type)
    CHECK(pout_arg1)
    CHECK(pin_arg1)
    CHECK(pout_arg1)
    CHECK(farg1)
    CHECK(farg2)
    CHECK(handle)
    CHECK(op1)
    CHECK(op2)
    CHECK(iarg1)
    CHECK(iarg2)
    CHECK(iarg3)
    CHECK(iarg4)
    CHECK(iarg5)
    CHECK(iarg6)
    CHECK(result)
    return true;
  }
  bool operator!=(const CuBlasCall &rhs) const { return !(operator==(rhs)); }
};

void printty(bool v) {
  if (v)
    printf("Reverse (true)");
  else
    printf("Primal (false)");
}

void printty(CallType v) {
  switch (v) {
  case CallType::GEMV:
    printf("GEMV");
    return;
  case CallType::GEMM:
    printf("GEMM");
    return;
  case CallType::SCAL:
    printf("SCAL");
    return;
  case CallType::GER:
    printf("GER");
    return;
  case CallType::COPY:
    printf("COPY");
    return;
  case CallType::LACPY:
    printf("LACPY");
    return;
  case CallType::DOT:
    printf("DOT");
    return;
  case CallType::AXPY:
    printf("AXPY");
    return;
  case CallType::LASCL:
    printf("LASCL");
    return;
  default:
    printf("UNKNOWN CALL (%d)", (int)v);
  }
}

void printty(void *v) {
  if (v == A)
    printf("A");
  else if (v == dA)
    printf("dA");
  else if (v == B)
    printf("B");
  else if (v == dB)
    printf("dB");
  else if (v == C)
    printf("C");
  else if (v == dC)
    printf("dC");
  else if (v == UNUSED_POINTER)
    printf("UNUSED_POINTER");
  else
    printf("Unknown pointer");
  printf(" (%p)", v);
}

void printty(int v) {
  if (v == incA)
    printf("incA");
  else if (v == incB)
    printf("incB");
  else if (v == incC)
    printf("incC");
  else if (v == M)
    printf("M");
  else if (v == N)
    printf("N");
  else if (v == K)
    printf("K");
  else if (v == lda)
    printf("lda");
  else if (v == UNUSED_INT)
    printf("UNUSED_INT");
  else
    printf("Unknown int");
  printf(" (%d)", v);
}
void printty(cublasOperation_t v) {
  if (v == cublasOperation_t::CUBLAS_OP_UNUSED) {
    printf("UNUSED_OP (%c)", v);
    return;
  }
  if (v == cublasOperation_t::CUBLAS_OP_N) {
    printf("CUBLAS_OP_N (%i)", v);
    return;
  }
  if (v == cublasOperation_t::CUBLAS_OP_T) {
    printf("CUBLAS_OP_T (%i)", v);
    return;
  }
  printf("Unknown CuBlas Operation");
}
void printty(double v) {
  if (v == alpha)
    printf("alpha");
  else if (v == beta)
    printf("beta");
  else if (v == UNUSED_DOUBLE)
    printf("UNUSED_DOUBLE");
  else
    printf("Unknown double");
  printf(" (%f)", v);
}

void printcall(CuBlasCall rcall) {
  switch (rcall.type) {
  case CallType::LACPY:
    printf("LACPY(handle=");
    printty(rcall.handle);
    printf(", uplo=");
    printty(rcall.op1);
    printf(", M=");
    printty(rcall.op2);
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
    printf("LASCL(handle=");
    printty(rcall.handle);
    printf(", type=");
    printty(rcall.op1);
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
    printf("GEMV(handle=");
    printty(rcall.handle);
    printf(", trans=");
    printty(rcall.op1);
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
    printf("GEMM(handle=");
    printty(rcall.handle);
    printf(", transA=");
    printty(rcall.op1);
    printf(", transB=");
    printty(rcall.op2);
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
    printf("GER(handle=");
    printty(rcall.handle);
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
  default:
    printf("UNKNOWN CALL (%d)", (int)rcall.type);
    return;
  }
}

void printTrace(const vector<CuBlasCall> &tr, std::string prefix = "") {
  printf("%sPrimal:\n", prefix.c_str());
  bool reverse = false;
  for (size_t i = 0; i < tr.size(); i++) {
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

template <typename T>
void assert_eq(std::string scope, std::string varName, int i, T expected,
               T real, CuBlasCall texpected, CuBlasCall rcall) {
  if (expected == real)
    return;
  printf("Failure on call %d var %s found ", i, varName.c_str());
  printty(expected);
  printf(" expected ");
  printty(real);
  printf("\n");
  exit(1);
}

void check_equiv(std::string scope, int i, CuBlasCall expected,
                 CuBlasCall real) {
#define MAKEASSERT(name)                                                       \
  assert_eq(scope, #name, i, expected.name, real.name, expected, real);
  MAKEASSERT(inDerivative)
  MAKEASSERT(type)
  MAKEASSERT(pout_arg1);
  MAKEASSERT(pin_arg1);
  MAKEASSERT(pin_arg2);
  MAKEASSERT(farg1);
  MAKEASSERT(farg2);
  MAKEASSERT(handle);
  MAKEASSERT(op1);
  MAKEASSERT(op2);
  MAKEASSERT(iarg1);
  MAKEASSERT(iarg2);
  MAKEASSERT(iarg3);
  MAKEASSERT(iarg4);
  MAKEASSERT(iarg5);
  MAKEASSERT(iarg6);
}

vector<CuBlasCall> cucalls;
vector<CuBlasCall> foundCuCalls;

extern "C" {

using cublasOperation_t::CUBLAS_OP_UNUSED;

__attribute__((noinline)) cublasStatus_t
cublas_dlascl(cublasHandle_t *handle, cublasOperation_t type, int KL, int KU,
              double cfrom, double cto, int M, int N, double *A, int lda) {
  cucalls.push_back((CuBlasCall){inDerivative, CallType::LASCL, A,
                                 UNUSED_POINTER, UNUSED_POINTER, cfrom, cto,
                                 handle, type, CUBLAS_OP_UNUSED, M, N,
                                 UNUSED_INT, lda, KL, KU, UNUSED_POINTER});
  return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
}
__attribute__((noinline)) cublasStatus_t cublas_ddot(cublasHandle_t *handle,
                                                     int N, double *X, int incx,
                                                     double *Y, int incy,
                                                     double *result) {
  CuBlasCall call = {inDerivative,
                     CallType::DOT,
                     UNUSED_POINTER,
                     X,
                     Y,
                     UNUSED_DOUBLE,
                     UNUSED_DOUBLE,
                     handle,
                     CUBLAS_OP_UNUSED,
                     CUBLAS_OP_UNUSED,
                     N,
                     UNUSED_INT,
                     UNUSED_INT,
                     incx,
                     incy,
                     UNUSED_INT,
                     result};
  *result = 3.15 + N;
  cucalls.push_back(call);
  return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
}
__attribute__((noinline)) cublasStatus_t cublas_daxpy(cublasHandle_t *handle,
                                                      int N, double alpha,
                                                      double *X, int incx,
                                                      double *Y, int incy) {
  CuBlasCall call = {inDerivative,
                     CallType::AXPY,
                     Y,
                     X,
                     UNUSED_POINTER,
                     alpha,
                     UNUSED_DOUBLE,
                     handle,
                     CUBLAS_OP_UNUSED,
                     CUBLAS_OP_UNUSED,
                     N,
                     UNUSED_INT,
                     UNUSED_INT,
                     incx,
                     incy,
                     UNUSED_INT,
                     UNUSED_POINTER};
  cucalls.push_back(call);
  return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
}
__attribute__((noinline)) cublasStatus_t
cublas_dgemv(cublasHandle_t *handle, cublasOperation_t trans, int M, int N,
             double alpha, double *A, int lda, double *X, int incx, double beta,
             double *Y, int incy) {
  CuBlasCall call = {
      inDerivative,  CallType::GEMV,   Y, A, X,          alpha, beta, handle,
      trans,         CUBLAS_OP_UNUSED, M, N, UNUSED_INT, lda,   incx, incy,
      UNUSED_POINTER};
  cucalls.push_back(call);
  return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
}
__attribute__((noinline)) cublasStatus_t
cublas_dgemm(cublasHandle_t *handle, cublasOperation_t transA,
             cublasOperation_t transB, int M, int N, int K, double alpha,
             double *A, int lda, double *B, int ldb, double beta, double *C,
             int ldc) {
  cucalls.push_back((CuBlasCall){inDerivative, CallType::GEMM, C, A, B, alpha,
                                 beta, handle, transA, transB, M, N, K, lda,
                                 ldb, ldc, UNUSED_POINTER});
  return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
}
__attribute__((noinline)) cublasStatus_t
cublas_dscal(cublasHandle_t *handle, int N, double alpha, double *X, int incX) {
  cucalls.push_back((CuBlasCall){
      inDerivative, CallType::SCAL, X, UNUSED_POINTER, UNUSED_POINTER, alpha,
      UNUSED_DOUBLE, handle, CUBLAS_OP_UNUSED, CUBLAS_OP_UNUSED, N, UNUSED_INT,
      UNUSED_INT, incX, UNUSED_INT, UNUSED_INT, UNUSED_POINTER});
  return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
}

// A = alpha * X * transpose(Y) + A
__attribute__((noinline)) cublasStatus_t
cublas_dger(cublasHandle_t *handle, int M, int N, double alpha, double *X,
            int incX, double *Y, int incY, double *A, int lda) {
  cucalls.push_back((CuBlasCall){inDerivative, CallType::GER, A, X, Y, alpha,
                                 UNUSED_DOUBLE, handle, CUBLAS_OP_UNUSED,
                                 CUBLAS_OP_UNUSED, M, N, UNUSED_INT, incX, incY,
                                 lda, UNUSED_POINTER});
  return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
}

__attribute__((noinline)) cublasStatus_t cublas_dcopy(cublasHandle_t *handle,
                                                      int N, double *X,
                                                      int incX, double *Y,
                                                      int incY) {
  cucalls.push_back((CuBlasCall){
      inDerivative, CallType::COPY, Y, X, UNUSED_POINTER, alpha, UNUSED_DOUBLE,
      handle, CUBLAS_OP_UNUSED, CUBLAS_OP_UNUSED, N, UNUSED_INT, UNUSED_INT,
      incX, incY, UNUSED_INT, UNUSED_POINTER});
  return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
}
}

enum class ValueType { Matrix, Vector };
struct BlasInfo {
  void *ptr;
  ValueType ty;
  cublasHandle_t *cu_handle;
  int vec_length;
  int vec_increment;
  int mat_rows;
  int mat_cols;
  int mat_ld;
  BlasInfo(void *v_ptr, cublasHandle_t *handle, int length, int increment) {
    ptr = v_ptr;
    ty = ValueType::Vector;
    cu_handle = handle;
    vec_length = length;
    vec_increment = increment;
    mat_rows = -1;
    mat_cols = -1;
    mat_ld = -1;
  }
  BlasInfo(void *v_ptr, cublasHandle_t *handle, int rows, int cols, int ld) {
    ptr = v_ptr;
    ty = ValueType::Matrix;
    cu_handle = handle;
    vec_length = -1;
    vec_increment = -1;
    mat_rows = rows;
    mat_cols = cols;
    mat_ld = ld;
  }
  BlasInfo() {
    ptr = (void *)(-1);
    ty = ValueType::Matrix;
    cu_handle = UNUSED_CUBLAS_HANDLE;
    vec_length = -1;
    vec_increment = -1;
    mat_rows = -1;
    mat_cols = -1;
    mat_ld = -1;
  }
};

BlasInfo pointer_to_index(void *v, BlasInfo inputs[6]) {
  if (v == A || v == dA)
    return inputs[0];
  if (v == B || v == dB)
    return inputs[1];
  if (v == C || v == dC)
    return inputs[2];
  for (int i = 3; i < 6; i++)
    if (inputs[i].ptr == v)
      return inputs[i];
  assert(0 && " illegal pointer to invert");
}

void checkVector(BlasInfo info, std::string vecname, int length, int increment,
                 std::string test, CuBlasCall rcall,
                 const vector<CuBlasCall> &trace) {
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

void checkMatrix(BlasInfo info, std::string matname, int rows, int cols, int ld,
                 std::string test, CuBlasCall rcall,
                 const vector<CuBlasCall> &trace) {
  if (info.ty != ValueType::Matrix) {
    printf("Error in test %s, invalid memory\n", test.c_str());
    printTrace(trace);
    printcall(rcall);
    printf(" Input %s is not a matrix\n", matname.c_str());
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

void checkMemory(CuBlasCall rcall, BlasInfo inputs[6], std::string test,
                 const vector<CuBlasCall> &trace) {
  switch (rcall.type) {
    return;
  case CallType::LASCL: {
    printf("unsupported cublas call LASCL");
    exit(1);
  }
  case CallType::LACPY: {
    printf("unsupported cublas call LACPY");
    exit(1);
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
  case CallType::GEMV: {
    // Y = alpha * op(A) * X + beta * Y
    auto Y = pointer_to_index(rcall.pout_arg1, inputs);
    auto A = pointer_to_index(rcall.pin_arg1, inputs);
    auto X = pointer_to_index(rcall.pin_arg2, inputs);

    auto trans_char = rcall.op1;
    auto trans = !is_normal(trans_char);
    auto M = rcall.iarg1;
    auto N = rcall.iarg2;
    auto alpha = rcall.farg1;
    auto lda = rcall.iarg4;
    auto incX = rcall.iarg5;
    auto beta = rcall.farg2;
    auto incY = rcall.iarg6;

    // A is an m-by-n matrix
    checkMatrix(A, "A", /*rows=*/M, /*cols=*/N, /*ld=*/lda, test, rcall, trace);

    // if no trans, X must be N otherwise must be M
    // From
    // https://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9.html
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
  case CallType::GEMM: {
    // C = alpha * A^transA * B^transB + beta * C
    auto C = pointer_to_index(rcall.pout_arg1, inputs);
    auto A = pointer_to_index(rcall.pin_arg1, inputs);
    auto B = pointer_to_index(rcall.pin_arg2, inputs);

    auto transA_char = rcall.op1;
    auto transA = !is_normal(transA_char);
    auto transB_char = rcall.op2;
    auto transB = !is_normal(transB_char);
    auto M = rcall.iarg1;
    auto N = rcall.iarg2;
    auto K = rcall.iarg3;
    auto alpha = rcall.farg1;
    auto lda = rcall.iarg4;
    auto ldb = rcall.iarg5;
    auto beta = rcall.farg2;
    auto ldc = rcall.iarg6;

    // From
    // https://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
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
    checkMatrix(A, "A", /*rows=*/(!transA) ? M : K,
                /*cols=*/(!transA) ? K : M, /*ld=*/lda, test, rcall, trace);
    checkMatrix(B, "B", /*rows=*/(!transB) ? K : N,
                /*cols=*/(!transB) ? N : K, /*ld=*/ldb, test, rcall, trace);
    checkMatrix(C, "C", /*rows=*/M, /*cols=*/N, /*ld=*/ldc, test, rcall, trace);
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

    auto M = rcall.iarg1;
    auto N = rcall.iarg2;
    auto alpha = rcall.farg1;
    auto incX = rcall.iarg4;
    auto incY = rcall.iarg5;
    auto incA = rcall.iarg6;

    // From
    // https://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_ga458222e01b4d348e9b52b9343d52f828.html
    // x is an m element vector, y is an n element
    // vector and A is an m by n matrix.
    checkVector(X, "X", /*len=*/M, /*inc=*/incX, test, rcall, trace);
    checkVector(Y, "Y", /*len=*/N, /*inc=*/incY, test, rcall, trace);
    checkMatrix(A, "A", /*rows=*/M, /*cols=*/N, /*ld=*/incA, test, rcall,
                trace);
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
  default:
    printf("unsupported cublas call LASCL");
    exit(1);
  }
}

void checkMemoryTrace(BlasInfo inputs[6], std::string test,
                      const vector<CuBlasCall> &trace) {
  for (size_t i = 0; i < trace.size(); i++)
    checkMemory(trace[i], inputs, test, trace);
}

void init() {
  inDerivative = false;
  cucalls.clear();
}

void checkTest(std::string name) {
  if (foundCuCalls.size() != cucalls.size()) {
    printf("Test %s failed: Expected %zu calls, found %zu\n", name.c_str(),
           cucalls.size(), foundCuCalls.size());
    printf("Expected:\n");
    printTrace(cucalls, "  ");
    printf("Found:\n");
    printTrace(foundCuCalls, "  ");
    assert(0 && "non-equal call count");
    exit(1);
  }
  if (foundCuCalls != cucalls) {
    printf("Test %s failed\n", name.c_str());
    printf("Expected:\n");
    printTrace(cucalls, "  ");
    printf("Found:\n");
    printTrace(foundCuCalls, "  ");
  }
  for (size_t i = 0; i < cucalls.size(); i++) {
    check_equiv(name, i, foundCuCalls[i], cucalls[i]);
  }
}
