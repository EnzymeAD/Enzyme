#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;
__device__ int enzyme_dup;

constexpr int K = 16;

extern "C" __device__ __attribute__((noinline)) void
carried_body(const double *x, double y, double *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < K; ++k) {
    out[i * K + k] = (acc - y) * (acc + y);
    double xi = x[i * K + k];
    acc = acc * 0.5 + xi * xi;
  }
}

__global__ void carried_kernel(const double *x, double y, double *out,
                               double *dout) {
  __poseidon_fp_optimize<void>((void *)carried_body, enzyme_const, x,
                               enzyme_const, y, enzyme_dup, out, dout);
}

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main(void) {
  constexpr int N = 128;
  static double hx[N * K], hout[N * K], hone[N * K];
  for (int i = 0; i < N * K; ++i) {
    hx[i] = 0.25 + (i % 61) / 64.0;
    hone[i] = 1.0;
  }
  double *dx, *dout, *ddout;
  CK(cudaMalloc(&dx, sizeof(hx)));
  CK(cudaMalloc(&dout, sizeof(hout)));
  CK(cudaMalloc(&ddout, sizeof(hone)));
  CK(cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ddout, hone, sizeof(hone), cudaMemcpyHostToDevice));
  carried_kernel<<<2, 64>>>(dx, 0.75, dout, ddout);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hout, dout, sizeof(hout), cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (int i = 0; i < N * K; ++i)
    s += hout[i];
  printf("%.17g\n", s);
  return 0;
}
