#include <cstdio>
#include <cuda_runtime.h>

void run_1(double *x, int n);
void run_2(double *x, int n);

int main() {
  double *x;
  cudaMalloc(&x, 128 * sizeof(double));
  cudaMemset(x, 0, 128 * sizeof(double));
  run_1(x, 128);
  run_2(x, 128);
  printf("status %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  return 0;
}
