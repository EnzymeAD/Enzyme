// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x c++ -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll

// An mfem::forall body compiled for the host only: the annotation on the lambda
// marks the GPU kernels that call it, and a host build has none. The lambda's
// closure is no allocation the profiler could shadow, so the body is left as
// written rather than made a host site. Outside CUDA an attribute parses only
// after the lambda's parameter list.

#include "poseidon/poseidon.h"

template <typename BODY> void forall(int n, BODY &&body) {
  for (int i = 0; i < n; ++i)
    body(i);
}

void axpy(int n, double a, const double *x, double *y) {
  forall(n, [=] (int i) POSEIDON_OPTIMIZE_TAU(1e-9) { y[i] += a * x[i]; });
}

// CHECK: warning: {{.*}}Poseidon: axpy(int, double, double const*, double*)::$_0::operator()(int) const is a lambda; POSEIDON_OPTIMIZE on a lambda marks the GPU kernels that call it, and this compilation has none. Emitting the original computation.
// IR-NOT: __enzyme_autodiff
// IR-NOT: poseidonLogValue
// IR: define {{.*}}@_Z4axpyidPKdPd(
