// A forall_2D body carrying POSEIDON_OPTIMIZE_TAU, end to end through
// poseidon-clang++: a per-element dense product reduce over DeviceTensor
// captures, which reach the kernel as one by-value lambda. Profile
// generation shadows the three data pointers inside the lambda and seeds the
// one written; the solve recognizes the product with its extents read from the
// lambda's fields, and the stub dispatches it from the launch's arguments.
// The dense reading needs each DeviceTensor's leading extent to equal the loop
// bound captured next to it, which the stub checks at the launch: a padded
// tensor runs the kernel as written.
//
// REQUIRES: enzyme, cuda-device
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate -poseidon-cache=%t/pcache -poseidon-launch-dir=%t/pcache %s -o %t/exe
// RUN: cat %t/pcache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: env POSEIDON_PROFILE_DIR=%t/profile %t/exe profile \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
// RUN: cat %t/profile/*.fpprofile | FileCheck --check-prefix=PROF %s
//
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t/profile -poseidon-cache=%t/cache \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-rewrite=0 -poseidon-enable-precision=0 -poseidon-enable-matmul \
// RUN:     %s -o %t/opt.exe 2>&1 | FileCheck %s
// RUN: %t/opt.exe run | FileCheck --check-prefix=OPT %s

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

// The lambda (argument 1) holds nd at +0, np at +4, and the DeviceTensors of Q,
// G and y with their data pointers at +16, +48, +80 and their sizes from +24,
// +56, +88. The kernel, its two launch arguments, and the three pointers, all
// fields of the lambda; only the output's is seeded.
// DESC: "profile generation": [
// DESC-NEXT: "_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_ 0 2 3 1 1 1 0 0 8 off 16 48 80"{{$}}
// DESC-NEXT: ],

// PRIMAL: DENSE: rel err ok

// Each of the three FMAs the m loop unrolls into runs nd ne np times and the
// accumulation into y nd ne times, each with a gradient of one.
// PROF: MaxBlockDims = 1 64 1
// PROF-NEXT: MaxGridDims = 192 1 1
// PROF-NEXT: LaunchCount = 192
// PROF-NEXT: SiteId = 0
// PROF-NEXT: CanonicalHash = ec8a91ad004ef65f
// PROF-NEXT: {{^0$}}
// PROF: SumGrad = 2.35929600000000000e+06
// PROF-NEXT: SumAbsGrad = 2.35929600000000000e+06
// PROF-NEXT: Exec = 2359296
// PROF-NEXT: NumOperands = 3
// PROF: {{^1$}}
// PROF: SumGrad = 2.35929600000000000e+06
// PROF-NEXT: SumAbsGrad = 2.35929600000000000e+06
// PROF-NEXT: Exec = 2359296
// PROF-NEXT: NumOperands = 3
// PROF: {{^2$}}
// PROF: SumGrad = 3.68640000000000000e+04
// PROF-NEXT: SumAbsGrad = 3.68640000000000000e+04
// PROF-NEXT: Exec = 36864
// PROF-NEXT: NumOperands = 2
// PROF: {{^3$}}
// PROF: SumGrad = 2.35929600000000000e+06
// PROF-NEXT: SumAbsGrad = 2.35929600000000000e+06
// PROF-NEXT: Exec = 2359296
// PROF-NEXT: NumOperands = 3
// PROF-NOT: {{^[0-9]+$}}

// M = nd, Ncols = the kernel's N, K = 3 np, every leading dimension the product
// of a DeviceTensor's sizes, and each DeviceTensor's leading extent assumed to
// be np, which the stub checks before it dispatches the product at nm = 12.
// CHECK: [poseidon] site void CuKernel2D<reduce(int, int, int, int, double*, double*, double*, int)::'lambda'(int)>(int, reduce(int, int, int, int, double*, double*, double*, int)::'lambda'(int)): accuracy target 1e-12
// CHECK-NEXT: [poseidon] Optimizing [[BODY:preprocess__ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT__poseidon_body]] with relative error tolerance: 1.000000e-12
// CHECK: [poseidon]   lda=arg1[+56]*arg1[+60]  ldb=arg1[+24]*arg1[+28]*arg1[+32]  ldc=arg1[+88]*arg1[+92]
// CHECK-NEXT: [poseidon]   aColMajor=0 bColMajor=1 cColMajor=1  C=param1[+80] A=param1[+48] B=param1[+16]
// CHECK: [poseidon] Found 1 matrix product(s) in [[BODY]]
// CHECK-NEXT: Matmul[0]: 192x1x192 type=f64 origin=RuntimeShape
// CHECK: [poseidon] error budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err {{[0-9.]+e-[0-9]+}} <= 1.000000e-12 at confidence 9.500000e-01)
// CHECK: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#4)
// CHECK: [poseidon] pin: [[KERNEL:_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_]]:M0=ozaki-ii nm=12 wmma m16n16k16 s8/s32{{$}}
// CHECK: [poseidon] wrote descriptor for [[KERNEL]]: C=arg1[+80] A=arg1[+48] B=arg1[+16] arg1[+0]xarg0x3*arg1[+4] nm=12 (standalone), if arg1[+4] == arg1[+24], if arg1[+4] == arg1[+56]{{$}}
// CHECK-NEXT: [poseidon] {{.*}}annotated_lambda_host_gemm.cu: 1 function(s) a rewrite reached; every other function as written
// CHECK: [poseidon] replaced the launch in stub _ZL25__device_stub__CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_ with __poseidon_ozakiii_dgemm (C=arg1[+80] A=arg1[+48] B=arg1[+16]), if arg1[+4] == arg1[+24], if arg1[+4] == arg1[+56]{{$}}

// OPT: DENSE: rel err ok
// OPT-NEXT: PADDED: rel err ok
