__global__ void gemm_kernel(const double *A, const double *B, double *C) {}
__global__ void gemm_tall_kernel(const double *A, const double *B, double *C,
                                 int ncols) {}
void launch(const double *A, const double *B, double *C, int ncols) {
  gemm_kernel<<<dim3(32, 32), dim3(16, 16)>>>(A, B, C);
  gemm_tall_kernel<<<dim3(8, 32), dim3(16, 16)>>>(A, B, C, ncols);
}
