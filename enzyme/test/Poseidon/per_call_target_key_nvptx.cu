// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda -std=c++20 --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites -DA=1e-6 -DB=1e-9 %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -x cuda -std=c++20 --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites -DA=1e-12 -DB=1e-3 %s -o /dev/null 2>&1 | FileCheck %s
// REQUIRES: poseidon

// The profile of a call site does not depend on its accuracy target, also when
// the caller takes the target as a template argument, as a value or inside a
// type: the call's copy is named after the caller without it, so both compiles
// look for the same profiles. Callers that differ in anything else, or only in
// their targets, keep sites of their own.

#include "poseidon/poseidon.h"

template <auto t> struct Tolerance {
  static constexpr double value = t;
};

__attribute__((device)) double sq(double x) { return x * x + 1.0; }

template <auto t, int n> __attribute__((device)) double scaled(double x) {
  return n * __poseidon_optimize_tau(t, sq, x);
}

template <typename tag> __attribute__((device)) double tagged(double x) {
  return __poseidon_optimize_tau(tag::value, sq, x);
}

__attribute__((global)) void k(double *x) {
  x[0] = scaled<A, 1>(x[0]);
  x[1] = scaled<A, 2>(x[1]);
  x[2] = scaled<B, 2>(x[2]);
  x[3] = tagged<Tolerance<A>>(x[3]);
  x[4] = scaled<(float)A, 3>(x[4]);
}

// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi1EEdd.0: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi1EEdd.0.fpprofile); left unchanged
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi2EEdd.0: the profiled program did not contain this site
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi2EEdd.1: the profiled program did not contain this site
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6taggedI9ToleranceIL_Z3tauEEEdd.0: the profiled program did not contain this site
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi3EEdd.0: the profiled program did not contain this site
