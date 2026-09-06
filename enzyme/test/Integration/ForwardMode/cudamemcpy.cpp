// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O0 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O1 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O3 %s -S -emit-llvm -o - %loadClangEnzyme | %lli -; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang++ -fno-exceptions -std=c++11 -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-inline=1 | %lli -; fi

// Forward-mode differentiation of code that stages data through the CUDA
// runtime API. The device side is emulated on the host -- what matters is that
// Enzyme repeats every transfer on the shadow buffers by calling back into the
// same CUDA entry points, so the tangent follows the data.

#include "../test_utils.h"

#include <stdlib.h>
#include <string.h>

extern "C" {

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
};

// Count the transfers so the test can tell that the shadow ones really happen.
int num_htod = 0;
int num_dtoh = 0;

__attribute__((noinline)) int cudaMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  return 0;
}

__attribute__((noinline)) int cudaFree(void *ptr) {
  free(ptr);
  return 0;
}

__attribute__((noinline)) int cudaMemset(void *dst, int v, size_t n) {
  memset(dst, v, n);
  return 0;
}

__attribute__((noinline)) int cudaMemcpy(void *dst, const void *src, size_t n,
                                         cudaMemcpyKind kind) {
  if (kind == cudaMemcpyHostToDevice)
    num_htod++;
  if (kind == cudaMemcpyDeviceToHost)
    num_dtoh++;
  memcpy(dst, src, n);
  return 0;
}
}

// x -> device -> back to host, then summed as squares.
__attribute__((noinline)) double staged_sum(const double *x, size_t n) {
  void *dev;
  cudaMalloc(&dev, n * sizeof(double));
  cudaMemcpy(dev, x, n * sizeof(double), cudaMemcpyHostToDevice);

  double *scratch = (double *)malloc(n * sizeof(double));
  cudaMemcpy(scratch, dev, n * sizeof(double), cudaMemcpyDeviceToHost);

  double res = 0;
  for (size_t i = 0; i < n; i++)
    res += scratch[i] * scratch[i];

  free(scratch);
  cudaFree(dev);
  return res;
}

extern double __enzyme_fwddiff(void *, ...);

int main() {
  const size_t N = 4;
  double x[N] = {1.0, 2.0, 3.0, 4.0};
  double dx[N] = {1.0, 0.0, 0.0, 0.0};

  // d/dx sum(x_i^2) contracted with dx == 2 * (x . dx) == 2 * 1.0 == 2.0
  double dres = __enzyme_fwddiff((void *)staged_sum, x, dx, N);
  APPROX_EQ(dres, 2.0, 1e-10);

  // Both the primal and the shadow have to be transferred each way.
  APPROX_EQ((double)num_htod, 2.0, 1e-10);
  APPROX_EQ((double)num_dtoh, 2.0, 1e-10);

  double dx2[N] = {0.0, 1.0, 1.0, 0.0};
  // 2 * (2*1 + 3*1) == 10
  double dres2 = __enzyme_fwddiff((void *)staged_sum, x, dx2, N);
  APPROX_EQ(dres2, 10.0, 1e-10);

  return 0;
}
