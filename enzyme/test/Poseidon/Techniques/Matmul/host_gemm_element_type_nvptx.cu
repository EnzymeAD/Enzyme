// REQUIRES: cuda-toolkit
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: rm -rf %t && mkdir -p %t/prof
// RUN: cp %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile %t/prof/_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~f33fa5d595998a55.fpprofile
// RUN: sed 's/^CanonicalHash = .*/CanonicalHash = a225a267b58a203f/' %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile > %t/prof/_ZL10CuKernel2DIZ6reduceiiiiPfS0_S0_iEUliE_EviT_~a3cb65c01991466d.fpprofile
// RUN: cp %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile %t/prof/_ZL10CuKernel2DIZ6reduceiiiiPdS0_PfiEUliE_EviT_~ed62f5c54f9e31d4.fpprofile
// DEFINE: %{cc} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -mllvm -poseidon-print -mllvm -poseidon-enable-algebraic=0 -mllvm -poseidon-enable-precision=0 -mllvm -poseidon-enable-matmul %s -o /dev/null
// RUN: %{cc} -mllvm -poseidon-cache=%t/double 2>&1 | FileCheck --check-prefix=DOUBLE %s
// RUN: %{cc} -mllvm -poseidon-cache=%t/float -DREAL=float 2>&1 | FileCheck --check-prefix=FLOAT --implicit-check-not='wrote descriptor' %s
// RUN: %{cc} -mllvm -poseidon-cache=%t/stored -DSTORED=float 2>&1 | FileCheck --check-prefix=STORED --implicit-check-not='wrote descriptor' %s

// The runtime-shape product of native_dgemm_runtime_cols_nvptx.cu, whose only
// realization is a library call, and every library-call entry is a GEMM of
// doubles. In doubles it is recognized and dispatched; in floats it is
// refused, since the call would read and write its matrices as doubles; and
// computed in doubles but stored to floats it is refused, since the call
// stores the product as computed. The profile of Inputs/host_gemm_hip is used,
// with the float body's hash for it.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))
#ifndef REAL
#define REAL double
#endif
#ifndef STORED
#define STORED REAL
#endif

template <typename T, int D> struct DeviceTensor {
  int capacity;
  T *data;
  int sizes[D];
  template <typename... I> HD T &operator()(I... i) const {
    const int idx[D] = {i...};
    int r = idx[D - 1];
    for (int k = D - 2; k >= 0; --k)
      r = idx[k] + sizes[k] * r;
    return data[r];
  }
};

template <typename T, typename... I>
DeviceTensor<T, sizeof...(I)> Reshape(T *p, I... n) {
  DeviceTensor<T, sizeof...(I)> t{1, p, {n...}};
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

void reduce(int np, int ldq, int ne, int nd, REAL *Q, REAL *Gp, STORED *y,
            int blockY) {
  constexpr int d = 3, qSize = 1;
  const auto QRead = Reshape(Q, ldq, d, qSize, ne);
  const auto G = Reshape(Gp, np, d, nd);
  auto yDev = Reshape(y, nd, qSize, ne);
  auto body = [=] HD POSEIDON_OPTIMIZE_TAU(1e-12)(int e) {
    for (int i = __nvvm_read_ptx_sreg_tid_y(); i < nd;
         i += __nvvm_read_ptx_sreg_ntid_y()) {
      REAL sum = 0.;
      for (int m = 0; m < d; m++)
        for (int p = 0; p < np; p++)
          sum += QRead(p, m, 0, e) * G(p, m, i);
      yDev(i, 0, e) += sum;
    }
  };
  void *k = (void *)CuKernel2D<decltype(body)>;
  (void)k;
}


// DOUBLE: [poseidon] {{[^ ]+}}: recognized GEMM from 3 reduction loop(s)
// DOUBLE: [poseidon] wrote descriptor for _ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_: C=arg1[+80] A=arg1[+48] B=arg1[+16] arg1[+0]xarg0x3*arg1[+4] nm=12 (standalone), if arg1[+4] == arg1[+24], if arg1[+4] == arg1[+56]{{$}}
// FLOAT: [poseidon] {{[^ ]+}}: reduction chain of 3 loop(s) is not a library-call runtime-shape product: the product's elements are f32, and every library-call entry is a GEMM of doubles{{$}}
// STORED: [poseidon] {{[^ ]+}}: reduction chain of 3 loop(s) is not a library-call runtime-shape product: the accumulator is converted before it is stored, and the library call stores it as computed{{$}}
