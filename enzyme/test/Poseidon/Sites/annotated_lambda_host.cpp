// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x c++ -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll 2>&1 | FileCheck --implicit-check-not='warning:' %s
// RUN: %clang -x c++ -O1 -S -emit-llvm -I%poseidon_profile_inc %s -o %t/plain.ll
// RUN: sed 's/ "poseidon-not-a-site"//' %t/gen.ll | diff %t/plain.ll -

// An annotated forall lambda compiled for the host only has no kernel to mark:
// it is left as written, the module being the plugin-less one but for the
// no-site attribute.

#include "poseidon/poseidon.h"

template <typename BODY> void forall(int n, BODY &&body) {
  for (int i = 0; i < n; ++i)
    body(i);
}

void axpy(int n, double a, const double *x, double *y) {
  forall(n, [=] (int i) POSEIDON_OPTIMIZE_TAU(1e-9) { y[i] += a * x[i]; });
}

// CHECK: annotated_lambda_host.cpp:[[@LINE-3]]:{{[0-9]+}}: warning: Poseidon: axpy(int, double, double const*, double*)::$_0::operator()(int) const is a lambda and this compilation has no GPU kernel that calls it, so it is left as written; POSEIDON_OPTIMIZE on a lambda marks the kernels that call it in a CUDA or HIP compilation
