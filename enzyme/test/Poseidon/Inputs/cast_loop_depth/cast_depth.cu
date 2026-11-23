// Per thread, a sum over j of a squared row entry times a row of a matrix
// summed over k; the trip counts are arguments of at least one.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

POSEIDON_OPTIMIZE __global__ void cast_depth(int n, int m, const double *a,
                                             const double *b, double *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double acc = 0.0;
  int j = 0;
  do {
    const double s = a[i * n + j] * a[i * n + j];
    int k = 0;
    do
      acc += s * b[j * m + k];
    while (++k < m);
  } while (++j < n);
  out[i] = acc;
}

int main(int argc, char **argv) {
  const int n = 8, m = 16, T = 64;
  double *ha = (double *)malloc(sizeof(double) * T * n);
  double *hb = (double *)malloc(sizeof(double) * n * m);
  double *ho = (double *)malloc(sizeof(double) * T);
  for (int i = 0; i < T * n; ++i)
    ha[i] = 0.5 + 0.001 * (i % 17);
  for (int i = 0; i < n * m; ++i)
    hb[i] = 1.0 + 0.01 * (i % 7);
  double *a, *b, *o;
  cudaMalloc(&a, sizeof(double) * T * n);
  cudaMalloc(&b, sizeof(double) * n * m);
  cudaMalloc(&o, sizeof(double) * T);
  cudaMemcpy(a, ha, sizeof(double) * T * n, cudaMemcpyHostToDevice);
  cudaMemcpy(b, hb, sizeof(double) * n * m, cudaMemcpyHostToDevice);
  cast_depth<<<1, T>>>(n, m, a, b, o);
  if (cudaMemcpy(ho, o, sizeof(double) * T, cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;
  double s = 0;
  for (int i = 0; i < T; ++i)
    s += ho[i];
  printf("checksum %.17g\n", s);
  return 0;
}
