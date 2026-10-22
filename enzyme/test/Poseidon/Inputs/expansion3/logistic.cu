#include <cstdio>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

POSEIDON_OPTIMIZE __global__ void logistic_kernel(const double *x0,
                                                  double *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  double x = x0[i];
#pragma unroll
  for (int k = 0; k < 24; ++k)
    x = 3.9 * x * (1.0 - x);
  out[i] = x;
}

int main(void) {
  constexpr int N = 256;
  double hx[N], hout[N];
  for (int i = 0; i < N; ++i)
    hx[i] = 0.2 + 0.6 * i / N;
  double *dx, *dout;
  cudaMalloc(&dx, sizeof(hx));
  cudaMalloc(&dout, sizeof(hout));
  cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
  logistic_kernel<<<4, 64>>>(dx, dout);
  if (cudaMemcpy(hout, dout, sizeof(hout), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;
  double s = 0.0;
  for (int i = 0; i < N; ++i)
    s += hout[i];
  printf("%.17g\n", s);
  return 0;
}
