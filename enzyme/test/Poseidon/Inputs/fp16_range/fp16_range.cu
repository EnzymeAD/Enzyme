#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

POSEIDON_OPTIMIZE __global__ void fms_kernel(const double *x, const double *y,
                                             double *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  double a = x[i], b = y[i];
  double t = a * b + a;
  double u = t * t - b;
  double v = u * b + t;
  out[i] = v * v - a;
}

int main(int argc, char **argv) {
  const double scale = argc > 1 ? std::atof(argv[1]) : 1.0;
  constexpr int N = 256;
  double hx[N], hy[N], hout[N];
  for (int i = 0; i < N; ++i) {
    hx[i] = scale * (0.25 + i / 512.0);
    hy[i] = 0.5 + (i % 17) / 32.0;
  }
  double *dx, *dy, *dout;
  cudaMalloc(&dx, sizeof(hx));
  cudaMalloc(&dy, sizeof(hy));
  cudaMalloc(&dout, sizeof(hout));
  cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice);
  fms_kernel<<<4, 64>>>(dx, dy, dout);
  if (cudaMemcpy(hout, dout, sizeof(hout), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;
  double s = 0.0;
  for (int i = 0; i < N; ++i)
    s += hout[i];
  printf("%.17g\n", s);
  return 0;
}
