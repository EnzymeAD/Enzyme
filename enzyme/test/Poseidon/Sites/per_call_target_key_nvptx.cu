// REQUIRES: cuda-runtime
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda -std=c++20 --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -DA=1e-6 -DB=1e-9 -DCA=4517329193108106637 -DCB=4472406533629990549 %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -x cuda -std=c++20 --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -DA=1e-12 -DB=1e-3 -DCA=4427486594234968593 -DCB=4562254508917369340 %s -o /dev/null 2>&1 | FileCheck %s

// The profile of a call site does not depend on its accuracy target, also when
// the caller takes the target as a template argument, as a value or inside a
// type, or as the bits of a double in a 64-bit unsigned integer template
// argument, the form a C++17 program uses: the call's copy is named after the
// caller without it, so both compiles look for the same profiles. A and B are
// 1e-6 and 1e-9, then 1e-12 and 1e-3, and CA and CB their bits. Callers that
// differ in anything else, or only in their targets, keep sites of their own.
// An integer argument whose digits contain those of a target is no target.

#include "poseidon/poseidon.h"

#define CAT(a, b) CAT_(a, b)
#define CAT_(a, b) a##b

template <auto t> struct Tolerance {
  static constexpr double value = t;
};

template <unsigned long code> struct Code {
  static constexpr double value = __builtin_bit_cast(double, code);
};

__attribute__((device)) double sq(double x) { return x * x + 1.0; }

template <auto t, int n> __attribute__((device)) double scaled(double x) {
  return n * __poseidon_optimize_tau(t, sq, x);
}

template <typename tag> __attribute__((device)) double tagged(double x) {
  return __poseidon_optimize_tau(tag::value, sq, x);
}

template <unsigned long code, int n>
__attribute__((device)) double coded(double x) {
  return n * __poseidon_optimize_tau(Code<code>::value, sq, x);
}

template <unsigned long long code>
__attribute__((device)) double wide(double x) {
  return __poseidon_optimize_tau(Code<code>::value, sq, x);
}

template <unsigned long code, unsigned long other>
__attribute__((device)) double pair(double x) {
  return __poseidon_optimize_tau(Code<code>::value, sq, x);
}

__attribute__((global)) void k(double *x) {
  x[0] = scaled<A, 1>(x[0]);
  x[1] = scaled<A, 2>(x[1]);
  x[2] = scaled<B, 2>(x[2]);
  x[3] = tagged<Tolerance<A>>(x[3]);
  x[4] = scaled<(float)A, 3>(x[4]);
  x[5] = coded<CA, 1>(x[5]);
  x[6] = coded<CA, 2>(x[6]);
  x[7] = coded<CB, 2>(x[7]);
  x[8] = tagged<Code<CA>>(x[8]);
  x[9] = wide<CA>(x[9]);
  x[10] = pair<CA, CAT(CAT(1, CA), ul)>(x[10]);
}

// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi1EEdd.0.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi2EEdd.0.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi2EEdd.1.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z6taggedI9ToleranceIL_Z3tauEEEdd.0.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z6scaledITnDaL_Z3tauELi3EEdd.0.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z5codedIL_Z3tauELi1EEdd.0.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z5codedIL_Z3tauELi2EEdd.0.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z5codedIL_Z3tauELi2EEdd.1.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z6taggedI4CodeIL_Z3tauEEEdd.0.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z4wideIL_Z3tauEEdd.0.fpprofile:
// CHECK-DAG: warning: Poseidon: no profile at {{.*}}/_Z2sqd.poseidon._Z4pairIL_Z3tauELm1{{[0-9]+}}EEdd.0.fpprofile:
