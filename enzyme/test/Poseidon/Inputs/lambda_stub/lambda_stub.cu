// Host IR of lambda_stub.ll: clang -x cuda --cuda-host-only -O0 -S -emit-llvm.
#include <cuda_runtime.h>

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
};

template <typename BODY> __global__ static void CuKernel2D(int N, BODY body) {
  int k = blockIdx.x * blockDim.z + threadIdx.z;
  if (k < N)
    body(k);
}

void reduce(int nq, int ne, int nd, double *Q, double *G, double *y) {
  DeviceTensor<2> QR{nq * ne, Q, {nq, ne}};
  DeviceTensor<2> GR{nq * nd, G, {nq, nd}};
  DeviceTensor<2> Y{nd * ne, y, {nd, ne}};
  auto body = [=] __host__ __device__(int e) {
    for (int i = threadIdx.y; i < nd; i += blockDim.y) {
      double sum = 0.0;
      for (int p = 0; p < nq; ++p)
        sum += QR.data[p + QR.sizes[0] * e] * GR.data[p + GR.sizes[0] * i];
      Y.data[i + Y.sizes[0] * e] += sum;
    }
  };
  CuKernel2D<<<ne, dim3(1, nd, 1)>>>(ne, body);
}
