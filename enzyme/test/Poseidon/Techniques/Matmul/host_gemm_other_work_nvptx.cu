// REQUIRES: cuda-toolkit
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: rm -rf %t && mkdir -p %t/prof
// RUN: cp %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile %t/prof/_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~8ee02cf9c0312e47.fpprofile
// DEFINE: %{cc} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -mllvm -poseidon-print -mllvm -poseidon-enable-rewrite=0 -mllvm -poseidon-enable-precision=0 -mllvm -poseidon-enable-matmul %s -o /dev/null
// RUN: %{cc} -mllvm -poseidon-cache=%t/none 2>&1 | FileCheck --check-prefix=NONE %s
// RUN: %{cc} -mllvm -poseidon-cache=%t/atomic '-DWORK=__nvvm_atom_add_gen_i(&hits, 1)' 2>&1 | FileCheck --check-prefix=WORK --implicit-check-not='wrote descriptor' %s
// RUN: %{cc} -mllvm -poseidon-cache=%t/global '-DWORK=hits = 1' 2>&1 | FileCheck --check-prefix=WORK --implicit-check-not='wrote descriptor' %s
// RUN: %{cc} -mllvm -poseidon-cache=%t/call '-DWORK=record(e)' 2>&1 | FileCheck --check-prefix=WORK --implicit-check-not='wrote descriptor' %s

// The runtime-shape product of native_dgemm_runtime_cols_nvptx.cu, whose
// library call takes the place of its kernel's whole launch. A body that also
// adds to a global atomically, stores to one, or calls a function that may
// write memory would lose that work with the launch, so it is refused. The
// profile of Inputs/host_gemm_hip is used: the canonical form leaves the work
// out.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))

__attribute__((device)) int hits;
__attribute__((device)) void record(int);

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
#ifdef WORK
      WORK;
#endif
    }
  };
  void *k = (void *)CuKernel2D<decltype(body)>;
  (void)k;
}


// NONE: [poseidon] wrote descriptor for _ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_: C=arg1[+80] A=arg1[+48] B=arg1[+16] arg1[+0]xarg0x3*arg1[+4] nm=12 (standalone), if arg1[+4] == arg1[+24], if arg1[+4] == arg1[+56]{{$}}
// WORK: [poseidon] {{[^ ]+}}: reduction chain of 3 loop(s) is not a library-call runtime-shape product: the body has an effect besides the product's store, which the library call in place of its launch would lose{{$}}
