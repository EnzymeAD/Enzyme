// REQUIRES: cuda-toolkit
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// DEFINE: %{compile} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -mllvm -poseidon-print -mllvm -poseidon-enable-rewrite=0 -mllvm -poseidon-enable-precision=0 -mllvm -poseidon-enable-matmul
// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/param && cp %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile %t/param/_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EvsT_~572c7c860d062c97.fpprofile
// RUN: %{compile} -DNT=short -DDT=int -mllvm -poseidon-profile-use=%t/param -mllvm -poseidon-cache=%t/cparam %s -o /dev/null 2>&1 | FileCheck --check-prefix=PARAM --implicit-check-not='wrote descriptor' --implicit-check-not='library-call' %s
// RUN: mkdir -p %t/field && cp %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile %t/field/_ZL10CuKernel2DIZ6reduceiiisPdS0_S0_iEUliE_EviT_~5a6b68e8950a81c1.fpprofile
// RUN: %{compile} -DNT=int -DDT=short -mllvm -poseidon-profile-use=%t/field -mllvm -poseidon-cache=%t/cfield %s -o /dev/null 2>&1 | FileCheck --check-prefix=FIELD --implicit-check-not='wrote descriptor' --implicit-check-not='library-call' %s

// The runtime-shape product of native_dgemm_runtime_cols_nvptx.cu with a 16-bit
// dimension: the column count as a kernel parameter, or the row count as a
// field of the lambda. The host stub reads each dimension from the launch as
// a 32-bit integer, so recognition refuses the product and no dispatch is
// written. The profile is the HIP build's (Inputs/host_gemm_hip), filed under
// each CUDA kernel's name: both builds compile the kernel to one canonical
// form, which the compile checks against the profile's CanonicalHash.

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
__attribute__((global)) static void CuKernel2D(const NT N, BODY body) {
  const int k = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_z() +
                __nvvm_read_ptx_sreg_tid_z();
  if (k >= N)
    return;
  body(k);
}

void reduce(int np, int ldq, int ne, DT nd, double *Q, double *Gp, double *y,
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

// PARAM: [poseidon] [[BODY:preprocess__ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EvsT__poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: a GEMM dimension depends on param0, an i16, but the host reads each launch value as an i32{{$}}
// PARAM: [poseidon] no rewrite applied for [[BODY]];
// FIELD: [poseidon] [[BODY:preprocess__ZL10CuKernel2DIZ6reduceiiisPdS0_S0_iEUliE_EviT__poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: a GEMM dimension depends on param1[+0], an i16, but the host reads each launch value as an i32{{$}}
// FIELD: [poseidon] no rewrite applied for [[BODY]];
