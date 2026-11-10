// An mfem::forall_2D body carrying POSEIDON_OPTIMIZE_TAU, end to end through
// poseidon-clang++: the stage-3 reduce of MFEM's elasticity apply over
// DeviceTensor captures, which reach the kernel as one by-value lambda. Profile
// generation shadows the three data pointers inside the lambda and seeds the
// one written; the solve recognizes the product with its extents read from the
// lambda's fields, and the stub dispatches it from the launch's arguments.
// The dense reading needs each DeviceTensor's leading extent to equal the loop
// bound captured next to it, which the stub checks at the launch: a padded
// tensor runs the kernel as written.
//
// RUN: rm -rf %t.pcache && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate -poseidon-cache=%t.pcache %s -o %t.exe
// RUN: cat %t.pcache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe profile \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
//
// RUN: rm -rf %t.cache && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t.profile -poseidon-cache=%t.cache \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-herbie=0 -poseidon-enable-pt=0 -poseidon-raise-wmma \
// RUN:     %s -o %t.opt.exe 2>&1 | FileCheck %s
// RUN: %t.opt.exe run | FileCheck --check-prefix=OPT %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  template <typename... I> __host__ __device__ double &operator()(I... i) const {
    const int idx[D] = {i...};
    int r = idx[D - 1];
    for (int k = D - 2; k >= 0; --k)
      r = idx[k] + sizes[k] * r;
    return data[r];
  }
};

template <typename... I> DeviceTensor<sizeof...(I)> Reshape(double *p, I... n) {
  DeviceTensor<sizeof...(I)> t{1, p, {n...}};
  for (int k = 0; k < (int)sizeof...(I); ++k)
    t.capacity *= t.sizes[k];
  return t;
}

template <typename BODY> __global__ static void CuKernel2D(const int N, BODY body) {
  const int k = blockIdx.x * blockDim.z + threadIdx.z;
  if (k >= N)
    return;
  body(k);
}

template <typename BODY> void forall_2D(int N, int X, int Y, BODY &&body) {
  CuKernel2D<<<N, dim3(X, Y, 1)>>>(N, body);
  CK(cudaGetLastError());
}

// y(i,e) += sum_{m<3} sum_{p<np} Q(p,m,0,e) G(p,m,i), with Q stored with
// leading extent ldq >= np.
void reduce(int np, int ldq, int ne, int nd, double *Q, double *Gp, double *y,
            int blockY) {
  constexpr int d = 3, qSize = 1;
  const auto QRead = Reshape(Q, ldq, d, qSize, ne);
  const auto G = Reshape(Gp, np, d, nd);
  auto yDev = Reshape(y, nd, qSize, ne);
  forall_2D(ne, qSize, blockY,
            [=] __host__ __device__ POSEIDON_OPTIMIZE_TAU(1e-12)(int e) {
              for (int i = threadIdx.y; i < nd; i += blockDim.y) {
                double sum = 0.;
                for (int m = 0; m < d; m++)
                  for (int p = 0; p < np; p++)
                    sum += QRead(p, m, 0, e) * G(p, m, i);
                yDev(i, 0, e) += sum;
              }
            });
}

static double check(int np, int ldq, int ne, int nd, const double *hQ,
                    const double *hG, double *dQ, double *dG, int blockY) {
  const size_t szY = (size_t)nd * ne;
  double *dY, *hY = (double *)std::malloc(szY * 8);
  CK(cudaMalloc(&dY, szY * 8));
  CK(cudaMemset(dY, 0, szY * 8));
  reduce(np, ldq, ne, nd, dQ, dG, dY, blockY);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hY, dY, szY * 8, cudaMemcpyDeviceToHost));
  double num = 0.0, den = 0.0;
  for (int e = 0; e < ne; ++e)
    for (int i = 0; i < nd; ++i) {
      double r = 0.0;
      for (int m = 0; m < 3; ++m)
        for (int p = 0; p < np; ++p)
          r += hQ[p + ldq * (m + 3 * e)] * hG[p + np * (m + 3 * i)];
      double dlt = hY[i + nd * e] - r;
      num += dlt * dlt;
      den += r * r;
    }
  CK(cudaFree(dY));
  std::free(hY);
  return std::sqrt(num / den);
}

int main(int argc, char **argv) {
  const bool profile = argc > 1 && !std::strcmp(argv[1], "profile");
  // The reverse pass caches through the device heap and needs more registers
  // than a full block allows.
  CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1ull << 30));
  const int np = 64, ne = 192, nd = 192;
  int failed = 0;
  for (int pad = 0; pad < (profile ? 1 : 2); ++pad) {
    const int ldq = np + pad;
    const size_t szQ = (size_t)ldq * 3 * ne, szG = (size_t)np * 3 * nd;
    double *hQ = (double *)std::malloc(szQ * 8);
    double *hG = (double *)std::malloc(szG * 8);
    for (size_t k = 0; k < szQ; ++k)
      hQ[k] = (k % ldq) < (size_t)np ? std::sin(0.37 * (double)k) : 1e30;
    for (size_t k = 0; k < szG; ++k)
      hG[k] = std::cos(0.11 * (double)k);
    double *dQ, *dG;
    CK(cudaMalloc(&dQ, szQ * 8));
    CK(cudaMalloc(&dG, szG * 8));
    CK(cudaMemcpy(dQ, hQ, szQ * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dG, hG, szG * 8, cudaMemcpyHostToDevice));
    double err = check(np, ldq, ne, nd, hQ, hG, dQ, dG, profile ? 64 : nd);
    // The dispatch is held to the target, the kernel as written to FP64.
    const double bound = pad ? 1e-14 : 1e-11;
    printf("%s: rel err %s\n", pad ? "PADDED" : "DENSE",
           err <= bound ? "ok" : "too large");
    if (!(err <= bound)) {
      fprintf(stderr, "rel err %.3e > %.0e\n", err, bound);
      failed = 1;
    }
    CK(cudaFree(dQ));
    CK(cudaFree(dG));
    std::free(hQ);
    std::free(hG);
  }
  return failed;
}

// The kernel, its two launch arguments, and its three pointers, all fields of
// the lambda (argument 1); only the output's is seeded.
// DESC: CuKernel2D{{.*}} 0 2 3 1 1 1 0 0 8 off {{[0-9]+ [0-9]+ [0-9]+}}"

// PRIMAL: DENSE: rel err ok

// M = nd, Ncols = the kernel's N, K = 3 np, every leading dimension the product
// of a DeviceTensor's sizes, and each DeviceTensor's leading extent assumed to
// be np.
// CHECK: [hostgemm] {{.*}}: recognized GEMM from 3 reduction loop(s)
// CHECK-NEXT: [hostgemm]   C = alpha*A^T*B + beta*C  alpha=1{{.*}} beta=1{{.*}}
// CHECK-NEXT: [hostgemm]   M=arg1[+{{[0-9]+}}]  Ncols=arg0  K=3*arg1[+[[NP:[0-9]+]]]
// CHECK-NEXT: [hostgemm]   lda=arg1[+[[G0:[0-9]+]]]*arg1[+{{[0-9]+}}]  ldb=arg1[+[[Q0:[0-9]+]]]*arg1[+{{[0-9]+}}]*arg1[+{{[0-9]+}}]  ldc=arg1[+{{[0-9]+}}]*arg1[+{{[0-9]+}}]
// CHECK-NEXT: [hostgemm]   aColMajor=0 bColMajor=1 cColMajor=1  C=param1[+{{[0-9]+}}] A=param1[+{{[0-9]+}}] B=param1[+{{[0-9]+}}]
// CHECK-DAG: [hostgemm]   dense only if param1[+[[NP]]] == param1[+[[Q0]]] (checked at the launch)
// CHECK-DAG: [hostgemm]   dense only if param1[+[[NP]]] == param1[+[[G0]]] (checked at the launch)
// CHECK: Poseidon error-budget: matmul -> ozaki-ii
// CHECK: [ozaki-host-dispatch] replaced the launch in stub {{.*}}CuKernel2D{{.*}} with __poseidon_ozaki_dgemm_ex

// OPT: DENSE: rel err ok
// OPT: PADDED: rel err ok
