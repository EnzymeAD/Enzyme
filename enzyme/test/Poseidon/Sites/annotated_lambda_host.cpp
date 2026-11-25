// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x c++ -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll 2>&1 | FileCheck --implicit-check-not='warning:' %s
// RUN: %clang -x c++ -O1 -S -emit-llvm -I%poseidon_profile_inc %s -o %t/plain.ll
// RUN: sed 's/ "poseidon-not-a-site"//' %t/gen.ll | diff %t/plain.ll -

// An mfem::forall body compiled for the host only: the annotation on the lambda
// marks the GPU kernels that call it, and a host build has none. The lambda's
// closure is no allocation the profiler could shadow, so the body is left as
// written rather than made a host site: the module is the one a compile without
// the plugin emits, but for the attribute that marks the lambda as no site.
// Outside CUDA an attribute parses only after the lambda's parameter list.

#include "poseidon/poseidon.h"

template <typename BODY> void forall(int n, BODY &&body) {
  for (int i = 0; i < n; ++i)
    body(i);
}

void axpy(int n, double a, const double *x, double *y) {
  forall(n, [=] (int i) POSEIDON_OPTIMIZE_TAU(1e-9) { y[i] += a * x[i]; });
}

// CHECK: annotated_lambda_host.cpp:[[@LINE-3]]:{{[0-9]+}}: warning: Poseidon: axpy(int, double, double const*, double*)::$_0::operator()(int) const is a lambda; POSEIDON_OPTIMIZE on a lambda marks the GPU kernels that call it, and this compilation has none. Emitting the original computation.
