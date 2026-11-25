// REQUIRES: cuda-toolkit
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: rm -rf %t && mkdir -p %t/prof
// RUN: sed "s/^CanonicalHash = .*/CanonicalHash = 35f30366bcfd6f97/" %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile > %t/prof/_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~593be11166e236ce.fpprofile
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/c -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -mllvm -poseidon-print -mllvm -poseidon-enable-rewrite=0 -mllvm -poseidon-enable-precision=0 -mllvm -poseidon-enable-matmul %s -o /dev/null 2>&1 | FileCheck --implicit-check-not='wrote descriptor' --implicit-check-not='recognized GEMM' %s

// The runtime-shape product of native_dgemm_runtime_cols_nvptx.cu with its
// store kept to the lower triangle, i <= e. A library call writes every element
// of the product, so recognition refuses it and no dispatch is written. The
// profile of Inputs/host_gemm_hip is used with the guarded body's canonical
// hash: recognition reads its header only.

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
      if (i <= e)
        yDev(i, 0, e) += sum;
    }
  };
  void *k = (void *)CuKernel2D<decltype(body)>;
  (void)k;
}

// CHECK: [poseidon] [[BODY:preprocess__ZL10CuKernel2D[^ ]+_poseidon_body]]: reduction chain of 3 loop(s) is not a dispatchable runtime-shape product: the output store is gated by a condition that is not a bound of the product's indices
