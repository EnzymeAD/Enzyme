// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s

// A view that holds null when its extent is empty and is made inside the site
// writes through a select between the pointer it was made from and null. The
// pointer is still the site's output and its shadow takes the seed: unseeded,
// every gradient of the reverse pass is zero.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  HD DeviceTensor(double *p, int n0, int n1) {
    sizes[0] = n0;
    sizes[1] = n1;
    capacity = n0 * n1;
    data = capacity > 0 ? p : nullptr;
  }
  HD double &operator()(int a, int b) const { return data[a + sizes[0] * b]; }
};

POSEIDON_OPTIMIZE __attribute__((global)) void
param_view(double *y, const double *x, int nd, int ne) {
  DeviceTensor<2> Y(y, nd, ne);
  int e = __nvvm_read_ptx_sreg_ctaid_x();
  for (int i = __nvvm_read_ptx_sreg_tid_x(); i < nd;
       i += __nvvm_read_ptx_sreg_ntid_x())
    Y(i, e) += x[i] * x[i];
}

template <typename BODY>
__attribute__((global)) static void CuKernel1D(int N, BODY body) {
  int k = __nvvm_read_ptx_sreg_ctaid_x();
  if (k >= N)
    return;
  body(k);
}

void capture_view(int nd, int ne, const double *x, double *y) {
  auto body = [=] HD(int e) POSEIDON_OPTIMIZE {
    DeviceTensor<2> Y(y, nd, ne);
    for (int i = __nvvm_read_ptx_sreg_tid_x(); i < nd;
         i += __nvvm_read_ptx_sreg_ntid_x())
      Y(i, e) += x[i] * x[i];
  };
  void *k = (void *)CuKernel1D<decltype(body)>;
  (void)k;
}

// GEN-DAG: [poseidon] _Z10param_viewPdPKdii: site {{[0-9]+}}, 2 shadow buffer(s), output(s) 0{{$}}
// GEN-DAG: [poseidon] _ZL10CuKernel1DIZ12capture_viewiiPKdPdEUliE_EviT_: site {{[0-9]+}}, 2 shadow buffer(s), output(s) 1[+0]{{$}}

// The pointers among the launch arguments and the seed width of each.
// DESC-DAG: "_Z10param_viewPdPKdii {{[0-9]+}} 4 2 0 1 8 0"
// DESC-DAG: "_ZL10CuKernel1DIZ12capture_viewiiPKdPdEUliE_EviT_ {{[0-9]+}} 2 2 1 1 8 0 off 0 16"
