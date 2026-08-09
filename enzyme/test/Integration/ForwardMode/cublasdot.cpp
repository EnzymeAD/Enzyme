// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -; fi

// A cuBLAS _v2 entry point returns its scalar through a trailing pointer
// rather than through the call's return value, and whether that pointer is
// host or device memory depends on the handle's pointer mode -- runtime state
// that a C caller usually leaves at the host default but that CUDA.jl sets to
// device on every handle it creates.
//
// The mock below emulates device memory as an arena and rejects a scalar
// pointer that does not match the current mode, so this test fails if Enzyme
// either drops the tangent or puts it in the wrong address space.
// See https://github.com/EnzymeAD/Enzyme.jl/issues/3442.

#include "../test_utils.h"

#include <stdlib.h>
#include <string.h>

extern "C" {

enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 };

struct cublasHandle_t {
  int mode;
};

// Emulated device memory. A pointer is a device pointer iff it lands here.
static char device_arena[4096];
static size_t device_used = 0;

static double *device_alloc(size_t n) {
  double *p = (double *)(device_arena + device_used);
  device_used += n * sizeof(double);
  return p;
}

static bool is_device_ptr(const void *p) {
  return (const char *)p >= device_arena &&
         (const char *)p < device_arena + sizeof(device_arena);
}

__attribute__((noinline)) int
cublasGetPointerMode_v2(cublasHandle_t *handle, int *mode) {
  *mode = handle->mode;
  return 0;
}

__attribute__((noinline)) int cublasSetPointerMode_v2(cublasHandle_t *handle,
                                                      int mode) {
  handle->mode = mode;
  return 0;
}

__attribute__((noinline)) int cublasSetVector(int n, int elemSize,
                                              const void *x, int incx, void *y,
                                              int incy) {
  // host -> device
  if (is_device_ptr(x) || !is_device_ptr(y))
    abort();
  memcpy(y, x, (size_t)n * elemSize);
  return 0;
}

__attribute__((noinline)) int cublasGetVector(int n, int elemSize,
                                              const void *x, int incx, void *y,
                                              int incy) {
  // device -> host
  if (!is_device_ptr(x) || is_device_ptr(y))
    abort();
  memcpy(y, x, (size_t)n * elemSize);
  return 0;
}

__attribute__((noinline)) int cublasDdot_v2(cublasHandle_t *handle, int n,
                                            const double *x, int incx,
                                            const double *y, int incy,
                                            double *result) {
  // The scalar has to live where the handle's pointer mode says it does.
  if ((handle->mode == CUBLAS_POINTER_MODE_DEVICE) != is_device_ptr(result))
    abort();
  double res = 0;
  for (int i = 0; i < n; i++)
    res += x[i * incx] * y[i * incy];
  *result = res;
  return 0;
}
}

__attribute__((noinline)) void my_ddot(cublasHandle_t *handle, int n,
                                       const double *x, const double *y,
                                       double *result) {
  cublasDdot_v2(handle, n, x, 1, y, 1, result);
}

extern "C" double __enzyme_fwddiff(void *, ...);
int enzyme_const;

int main() {
  const int N = 4;
  double x[N] = {1.0, 2.0, 3.0, 4.0};
  double y[N] = {5.0, 6.0, 7.0, 8.0};
  double dx[N] = {1.0, 0.0, 0.0, 0.0};
  double dy[N] = {0.0, 1.0, 0.0, 0.0};

  // d(x . y) = dx . y + x . dy == y[0] + x[1] == 5 + 2 == 7
  const double expected = 7.0;

  // Device pointer mode, as CUDA.jl configures its handles: the result and its
  // shadow live in device memory.
  {
    cublasHandle_t handle;
    handle.mode = CUBLAS_POINTER_MODE_DEVICE;
    double *res = device_alloc(1);
    double *dres = device_alloc(1);
    *res = 0;
    *dres = 0;
    __enzyme_fwddiff((void *)my_ddot, enzyme_const, &handle, enzyme_const, N, x,
                     dx, y, dy, res, dres);
    APPROX_EQ(*dres, expected, 1e-10);
    // The handle must be left as it was found.
    TEST_EQ(handle.mode, CUBLAS_POINTER_MODE_DEVICE);
  }

  // Host pointer mode, the default a C caller sees.
  {
    cublasHandle_t handle;
    handle.mode = CUBLAS_POINTER_MODE_HOST;
    double res = 0, dres = 0;
    __enzyme_fwddiff((void *)my_ddot, enzyme_const, &handle, enzyme_const, N, x,
                     dx, y, dy, &res, &dres);
    APPROX_EQ(dres, expected, 1e-10);
    APPROX_EQ(res, 70.0, 1e-10);
    TEST_EQ(handle.mode, CUBLAS_POINTER_MODE_HOST);
  }

  return 0;
}
