__global__ void matmul_kernel(const double *A, const double *B, double *D,
                              double *dD) {}
void launch(const double *A, const double *B, double *D, double *dD) {
  matmul_kernel<<<1, dim3(16, 16)>>>(A, B, D, dD);
}
