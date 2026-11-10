// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: ls %t/cache | FileCheck --check-prefix=FILE %s
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// REQUIRES: poseidon, enzyme

// The launch descriptor the device compile leaves for the host compile is a
// line of the manifest of the unit, a file named after the source. The mangled
// name of a kernel instantiated with a lambda (an MFEM forall body) can be
// longer than a file name may be; the manifest keeps the full name, where the
// host compile reads it.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  HD double &operator()(int a, int b) const { return data[a + sizes[0] * b]; }
};

template <typename BODY> __attribute__((global)) static void CuKernel2D(int N, BODY body) {
  int k = __nvvm_read_ptx_sreg_ctaid_x();
  if (k >= N)
    return;
  body(k);
}

void reduce_quadrature_values_to_element_degrees_of_freedom_with_a_name_as_long_as_the_template_instantiations_of_a_finite_element_library_produce_for_their_forall_lambdas_so_that_the_kernel_name_no_longer_fits_in_a_file_name(
    int nq, int ne, int nd, double *Q, double *G, double *y) {
  DeviceTensor<2> QR{nq * ne, Q, {nq, ne}};
  DeviceTensor<2> GR{nq * nd, G, {nq, nd}};
  DeviceTensor<2> Y{nd * ne, y, {nd, ne}};
  auto body = [=] HD (int e) POSEIDON_OPTIMIZE_TAU(1e-9) {
    for (int i = __nvvm_read_ptx_sreg_tid_y(); i < nd;
         i += __nvvm_read_ptx_sreg_ntid_y()) {
      double sum = 0.0;
      for (int p = 0; p < nq; ++p)
        sum += QR(p, e) * GR(p, i);
      Y(i, e) += sum;
    }
  };
  void *k = (void *)CuKernel2D<decltype(body)>;
  (void)k;
}

// GEN-NOT: cannot write
// GEN: [poseidon-profgen] _ZL10CuKernel2DIZ220reduce_quadrature_values_to_element_degrees_of_freedom_with_a_name_as_long_as_the_template_instantiations_of_a_finite_element_library_produce_for_their_forall_lambdas_so_that_the_kernel_name_no_longer_fits_in_a_file_nameiiiPdS0_S0_EUliE_EviT_: site 0, 3 shadow buffer(s), output(s) 1[+64]
// GEN-NOT: cannot write

// FILE: {{^}}launches_long_kernel_name_nvptx.cu_{{[0-9A-F]+}}.json{{$}}

// DESC: {{^ *}}"_ZL10CuKernel2DIZ220reduce_quadrature_values_to_element_degrees_of_freedom_with_a_name_as_long_as_the_template_instantiations_of_a_finite_element_library_produce_for_their_forall_lambdas_so_that_the_kernel_name_no_longer_fits_in_a_file_nameiiiPdS0_S0_EUliE_EviT_ 0 2 3 1 1 1 0 0 8 off 16 40 64"{{$}}
