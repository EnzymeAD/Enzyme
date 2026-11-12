// RUN: rm -rf %t && mkdir -p %t/prof
// RUN: sed -e 's/^MaxGridDims = 192 1 1$/MaxGridDims = 1536 1 1/' -e 's/^LaunchCount = 192$/LaunchCount = 1536/' %S/../Inputs/host_gemm_hip/preprocess__ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT__poseidon_body.fpprofile > %t/prof/preprocess__ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT__poseidon_body.fpprofile
// RUN: cp %cost_models/cm_sm_120_synthetic_dispatch.csv %t/cm.csv && echo 'ozaki_dispatch_rect_rel,dgemm_m24n1536k192,0.05' >> %t/cm.csv
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/c -mllvm -poseidon-cost-model=%t/cm.csv -mllvm -poseidon-print -mllvm -poseidon-enable-herbie=0 -mllvm -poseidon-enable-pt=0 -mllvm -poseidon-raise-wmma %s -o /dev/null 2>&1 | FileCheck --implicit-check-not='the native DGEMM runs at' %s

// The native DGEMM candidate of a product whose column count is a run-time
// value: the stage-3 reduce of annotated_lambda_host_gemm_amdgpu.hip with its
// profile (Inputs/host_gemm_hip) spread over 1536 CTAs, so the profiled
// product is 24 x 1536 x 192 (padWaste 512). The surrogate's extents are not
// the ones the call runs at, so the row measured at them is not read and the
// square row prices the candidate, with no warning naming a row to measure.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  template <typename... I> HD double &operator()(I... i) const {
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

template <typename BODY>
__attribute__((global)) static void CuKernel2D(const int N, BODY body) {
  const int k = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_z() +
                __nvvm_read_ptx_sreg_tid_z();
  if (k >= N)
    return;
  body(k);
}

void reduce(int np, int ldq, int ne, int nd, double *Q, double *Gp, double *y,
            int blockY) {
  constexpr int d = 3, qSize = 1;
  const auto QRead = Reshape(Q, ldq, d, qSize, ne);
  const auto G = Reshape(Gp, np, d, nd);
  auto yDev = Reshape(y, nd, qSize, ne);
  auto body = [=] HD POSEIDON_OPTIMIZE_TAU(1e-12)(int e) {
    for (int i = __nvvm_read_ptx_sreg_tid_y(); i < nd;
         i += __nvvm_read_ptx_sreg_ntid_y()) {
      double sum = 0.;
      for (int m = 0; m < d; m++)
        for (int p = 0; p < np; p++)
          sum += QRead(p, m, 0, e) * G(p, m, i);
      yDev(i, 0, e) += sum;
    }
  };
  void *k = (void *)CuKernel2D<decltype(body)>;
  (void)k;
}

// CHECK: [hostgemm]   profile shape M=24 Ncols=1536 K=192
// CHECK: native-dgemm cublas f64/f64  compCost/MAC={{.*}}  rel=5.000000e-01
// CHECK: Poseidon error-budget: matmul -> native-dgemm cublas f64/f64
