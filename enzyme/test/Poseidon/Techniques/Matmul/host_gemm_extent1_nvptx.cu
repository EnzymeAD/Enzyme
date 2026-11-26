// REQUIRES: cuda-toolkit
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: rm -rf %t && mkdir -p %t/prof %t/wide %t/gated
// RUN: cp %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile %t/prof/_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~4e0dd8e341e6247a.fpprofile
// RUN: sed 's/^MaxBlockDims = 1 64 1$/MaxBlockDims = 4 64 1/' %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile > %t/wide/_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~4e0dd8e341e6247a.fpprofile
// RUN: sed 's/^CanonicalHash = .*/CanonicalHash = 35f30366bcfd6f97/' %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile > %t/gated/_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~4e0dd8e341e6247a.fpprofile
// DEFINE: %{cc} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -mllvm -poseidon-print -mllvm -poseidon-enable-algebraic=0 -mllvm -poseidon-enable-precision=0 -mllvm -poseidon-enable-matmul %s -o /dev/null
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/eq 2>&1 | FileCheck --check-prefixes=EQ,ONE %s
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/wide -mllvm -poseidon-cache=%t/wide-c 2>&1 | FileCheck --check-prefixes=WIDE,ONE %s
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/and '-DRETURN_IF=__nvvm_read_ptx_sreg_tid_x() != 0' 2>&1 | FileCheck --check-prefix=ONE %s
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/ne '-DRETURN_IF=__nvvm_read_ptx_sreg_tid_y() >= nd || __nvvm_read_ptx_sreg_tid_x() != 0' 2>&1 | FileCheck --check-prefix=ONE %s
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/plane '-DXSTART=__nvvm_read_ptx_sreg_tid_x() - 1' '-DRETURN_IF=__nvvm_read_ptx_sreg_tid_x() != 1' 2>&1 | FileCheck --check-prefix=PLANE --implicit-check-not='wrote descriptor' %s
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/second '-DRETURN_IF=__nvvm_read_ptx_sreg_tid_y() >= nd || e >= ldq' 2>&1 | FileCheck --check-prefix=SECOND --implicit-check-not='wrote descriptor' %s
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/seven '-DRETURN_IF=e >= 7' 2>&1 | FileCheck --check-prefix=SECOND --implicit-check-not='wrote descriptor' %s
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/gated -mllvm -poseidon-cache=%t/gated-c '-DSTORE_IF=i <= e' 2>&1 | FileCheck --check-prefix=GATED --implicit-check-not='wrote descriptor' %s
// RUN: %{cc} -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/merged '-DRETURN_IF=__nvvm_read_ptx_sreg_tid_x() != 0 || __nvvm_read_ptx_sreg_tid_y() > e' 2>&1 | FileCheck --check-prefix=GATED --implicit-check-not='wrote descriptor' %s

// The runtime-shape product of native_dgemm_runtime_cols_nvptx.cu inside a
// block-strided thread loop of extent 1 along x, for (q = threadIdx.x; q < 1;
// q += blockDim.x), which LLVM turns into the guard threadIdx.x == 0 with q
// replaced by 0. The guard is the bound q < 1 of an index source that moves no
// address, so the threads that pass it write every element of the product,
// at any block width: the library call writes what the kernel writes. Its
// variants, by check prefix:
//   EQ     the guard as a branch of its own, on the profiled 1 x 64 x 1 block;
//   WIDE   the same on a 4 x 64 x 1 block, whose other threads write nothing;
//   ONE    also threadIdx.x == 0 and the wrapper's k < N merged into one
//          branch, and threadIdx.y >= nd || threadIdx.x != 0 on the edge where
//          it fails, which holds both of its bounds;
//   PLANE  threadIdx.x == 1, one plane of the index, whose threads exist only
//          in a block wider than 1, is refused;
//   SECOND e < ldq merged with an index bound, and e >= 7, which reaches the
//          pass as e u> 6 merged with k >= N, bound e a second time, which the
//          signature check refuses;
//   GATED  a store gated by i <= e, and threadIdx.x != 0 || threadIdx.y > e
//          merged into one branch, of which only the first is a bound, are
//          still refused.
// The profile of Inputs/host_gemm_hip is used, with the hash of the gated body
// for GATED; recognition reads only its header there.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))
#ifndef XSTART
#define XSTART __nvvm_read_ptx_sreg_tid_x()
#endif

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
#ifdef RETURN_IF
    if (RETURN_IF)
      return;
#endif
    for (int i = __nvvm_read_ptx_sreg_tid_y(); i < nd;
         i += __nvvm_read_ptx_sreg_ntid_y())
      for (int q = XSTART; q < qSize;
           q += __nvvm_read_ptx_sreg_ntid_x()) {
        double sum = 0.;
        for (int m = 0; m < d; m++)
          for (int p = 0; p < np; p++)
            sum += QRead(p, m, q, e) * G(p, m, i);
#ifdef STORE_IF
        if (STORE_IF)
#endif
          yDev(i, q, e) += sum;
      }
  };
  void *k = (void *)CuKernel2D<decltype(body)>;
  (void)k;
}

// EQ: [poseidon] profile header for {{[^ ]+}}: MaxBlockDims=(1,64,1) LaunchCount=192{{$}}
// WIDE: [poseidon] profile header for {{[^ ]+}}: MaxBlockDims=(4,64,1) LaunchCount=192{{$}}
// ONE: [poseidon] [[BODY:preprocess__ZL10CuKernel2D[^ ]+_poseidon_body]]: recognized GEMM from 3 reduction loop(s)
// ONE-NEXT: [poseidon]   C = A^T*B + beta*C  beta=1.000000e+00
// ONE-NEXT: [poseidon]   M=arg1[+0]  Ncols=arg0  K=3*arg1[+4]
// ONE: [poseidon]   profile shape M=192 Ncols=192 K=192 (per-CTA tile 192x1, launches=1 CTAs/launch=192 MACs=7077888)
// ONE: [poseidon] wrote descriptor for _ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_: C=arg1[+80] A=arg1[+48] B=arg1[+16] arg1[+0]xarg0x3*arg1[+4] nm=12 (standalone), if arg1[+4] == arg1[+24], if arg1[+4] == arg1[+56]{{$}}
// PLANE: [poseidon] [[BODY:preprocess__ZL10CuKernel2D[^ ]+_poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: the contraction runs only where a launch index equals 1, one plane of the index, which the library call does not place{{$}}
// SECOND: [poseidon] [[BODY:preprocess__ZL10CuKernel2D[^ ]+_poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: two index sources share a signature term{{$}}
// GATED: [poseidon] [[BODY:preprocess__ZL10CuKernel2D[^ ]+_poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: the output store is gated by a condition that is not a bound of the product's indices{{$}}
