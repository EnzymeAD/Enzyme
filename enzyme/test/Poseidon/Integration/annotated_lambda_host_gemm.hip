// annotated_lambda_host_gemm.cu on an AMD GPU (MI300A), end to end through
// poseidon-clang++ -x hip: profile generation shadows the three data pointers
// inside the lambda and seeds the one written, and records what the CUDA twin
// records on an RTX 5090 (the same launch geometry, slot execution counts and
// seeded adjoint counts, and the same canonical form). No gfx942 cost model
// exists yet, so the optimized build takes its dispatch descriptor from
// Inputs/host_gemm_hip, the one the CUDA twin's solve writes, under this
// kernel's name: the stub dispatches the dense product to the HIP Ozaki-II
// runtime from the launch's arguments, and a padded tensor still runs the
// kernel as written. The descriptor is in the manifest of this source under
// the unit name the host half alone is given, so the manifest the device half
// writes is another file; the source is compiled from a copy under a fixed
// name, which the manifest's file name follows.
//
// RUN: rm -rf %t.pcache && %poseidon_clangxx -x hip --offload-arch=%amdgpu_arch \
// RUN:     --rocm-path=%rocm_path -O2 -poseidon-profile-generate \
// RUN:     -poseidon-cache=%t.pcache %s -o %t.exe
// RUN: cat %t.pcache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe profile \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
// RUN: cat %t.profile/*.fpprofile | FileCheck --check-prefix=PROF %s
//
// RUN: rm -rf %t.cache %t.none %t.src && mkdir %t.cache %t.none %t.src
// RUN: cp %S/../Inputs/host_gemm_hip/launches_*.json %t.cache/
// RUN: cp %s %t.src/annotated_lambda_host_gemm.hip
// RUN: cd %t.src && %poseidon_clangxx -x hip --offload-arch=%amdgpu_arch \
// RUN:     --rocm-path=%rocm_path -O2 -poseidon-profile-use=%t.none \
// RUN:     -poseidon-cache=%t.cache -Xarch_host -mllvm=-poseidon-unit=seeded \
// RUN:     -poseidon-enable-herbie=0 annotated_lambda_host_gemm.hip \
// RUN:     -o %t.opt.exe 2>&1 | FileCheck %s
// RUN: %t.opt.exe run | FileCheck --check-prefix=OPT %s
//
// REQUIRES: poseidon, enzyme, hip-runtime

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#define CK(x)                                                                  \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "hip: %s\n", hipGetErrorString(e));                      \
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

template <typename BODY> __global__ static void HipKernel2D(const int N, BODY body) {
  const int k = blockIdx.x * blockDim.z + threadIdx.z;
  if (k >= N)
    return;
  body(k);
}

template <typename BODY> void forall_2D(int N, int X, int Y, BODY &&body) {
  HipKernel2D<<<N, dim3(X, Y, 1)>>>(N, body);
  CK(hipGetLastError());
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
  CK(hipMalloc(&dY, szY * 8));
  CK(hipMemset(dY, 0, szY * 8));
  reduce(np, ldq, ne, nd, dQ, dG, dY, blockY);
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(hY, dY, szY * 8, hipMemcpyDeviceToHost));
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
  CK(hipFree(dY));
  std::free(hY);
  return std::sqrt(num / den);
}

int main(int argc, char **argv) {
  const bool profile = argc > 1 && !std::strcmp(argv[1], "profile");
  // The reverse pass caches through the device heap and needs more registers
  // than a full block allows.
  CK(hipDeviceSetLimit(hipLimitMallocHeapSize, 1ull << 30));
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
    CK(hipMalloc(&dQ, szQ * 8));
    CK(hipMalloc(&dG, szG * 8));
    CK(hipMemcpy(dQ, hQ, szQ * 8, hipMemcpyHostToDevice));
    CK(hipMemcpy(dG, hG, szG * 8, hipMemcpyHostToDevice));
    double err = check(np, ldq, ne, nd, hQ, hG, dQ, dG, profile ? 64 : nd);
    // The dispatch is held to the target, the kernel as written to FP64.
    const double bound = pad ? 1e-14 : 1e-11;
    printf("%s: rel err %s\n", pad ? "PADDED" : "DENSE",
           err <= bound ? "ok" : "too large");
    if (!(err <= bound)) {
      fprintf(stderr, "rel err %.3e > %.0e\n", err, bound);
      failed = 1;
    }
    CK(hipFree(dQ));
    CK(hipFree(dG));
    std::free(hQ);
    std::free(hG);
  }
  return failed;
}

// DESC: HipKernel2D{{.*}} 0 2 3 1 1 1 0 0 8 off 16 48 80

// PRIMAL: DENSE: rel err ok

// PROF: MaxBlockDims = 1 64 1
// PROF-NEXT: MaxGridDims = 192 1 1
// PROF-NEXT: LaunchCount = 192
// PROF: CanonicalHash = ec8a91ad004ef65f
// PROF: SumGrad = 2.35929600000000000e+06
// PROF: Exec = 2359296
// PROF: SumGrad = 2.35929600000000000e+06
// PROF: Exec = 2359296
// PROF: SumGrad = 3.68640000000000000e+04
// PROF: Exec = 36864
// PROF: SumGrad = 2.35929600000000000e+06
// PROF: Exec = 2359296

// CHECK: [ozaki-host-dispatch] replaced the launch in stub {{.*}}HipKernel2D{{.*}} with __poseidon_ozaki_dgemm_ex (C=arg1[+80] A=arg1[+48] B=arg1[+16]), if arg1[+4] == arg1[+24], if arg1[+4] == arg1[+56]

// OPT: DENSE: rel err ok
// OPT: PADDED: rel err ok
