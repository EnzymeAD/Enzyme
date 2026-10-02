#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;
__device__ int enzyme_dup;

extern "C" __device__ __attribute__((noinline)) void
fms_body(const double *x, const double *y, double *out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  double a = x[i], b = y[i];
  double t = a * b + a;
  double u = t * t - b;
  double v = u * b + t;
  double w = v * v - a;
  out[i] = w * u + v;
}

__global__ void fms_kernel(const double *x, const double *y, double *out,
                           double *dout) {
  __poseidon_fp_optimize<void>((void *)fms_body, enzyme_const, x, enzyme_const,
                             y, enzyme_dup, out, dout);
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
  constexpr int N = 256;
  double hx[N], hy[N], hout[N], hone[N];
  for (int i = 0; i < N; ++i) {
    hx[i] = 0.25 + i / 512.0;
    hy[i] = 0.5 + (i % 17) / 32.0;
    hone[i] = 1.0;
  }
  double *dx, *dy, *dout, *ddout;
  CK(cudaMalloc(&dx, sizeof(hx)));
  CK(cudaMalloc(&dy, sizeof(hy)));
  CK(cudaMalloc(&dout, sizeof(hout)));
  CK(cudaMalloc(&ddout, sizeof(hone)));
  CK(cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dy, hy, sizeof(hy), cudaMemcpyHostToDevice));
  for (int rep = 0; rep < 3; ++rep) {
    CK(cudaMemcpy(ddout, hone, sizeof(hone), cudaMemcpyHostToDevice));
    fms_kernel<<<4, 64>>>(dx, dy, dout, ddout);
    CK(cudaDeviceSynchronize());
  }
  CK(cudaMemcpy(hout, dout, sizeof(hout), cudaMemcpyDeviceToHost));
  int nbad = 0;
  for (int i = 0; i < N; ++i) {
    double a = hx[i], b = hy[i];
    double t = a * b + a;
    double u = t * t - b;
    double v = u * b + t;
    double w = v * v - a;
    double e = w * u + v;
    if (std::fabs(hout[i] - e) > 1e-12 * std::fabs(e) + 1e-300)
      ++nbad;
  }
  printf(nbad ? "FMS-FAIL %d\n" : "FMS-PASS\n", nbad);
  return nbad != 0;
}
