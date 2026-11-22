// REQUIRES: cuda-runtime
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda -std=c++17 --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -DA=4517329193108106637 -DB=4472406533629990549 %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -x cuda -std=c++17 --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -DA=4427486594234968593 -DB=4562254508917369340 %s -o /dev/null 2>&1 | FileCheck %s

// The profile of a call site does not depend on its accuracy target, also when
// the caller takes the target as the bits of a double in a 64-bit unsigned
// integer template argument, the form a C++17 program uses: the call's copy is
// named after the caller without it, so both compiles look for the same
// profiles. A and B are the bits of 1e-6 and 1e-9, then of 1e-12 and 1e-3. An
// integer argument whose digits contain those of a target is no target.

#include "poseidon/poseidon.h"

#define CAT(a, b) CAT_(a, b)
#define CAT_(a, b) a##b

template <unsigned long code> struct Tolerance {
  static constexpr double value = __builtin_bit_cast(double, code);
};

__attribute__((device)) double sq(double x) { return x * x + 1.0; }

template <unsigned long code, int n>
__attribute__((device)) double scaled(double x) {
  return n * __poseidon_optimize_tau(Tolerance<code>::value, sq, x);
}

template <typename tag> __attribute__((device)) double tagged(double x) {
  return __poseidon_optimize_tau(tag::value, sq, x);
}

template <unsigned long long code>
__attribute__((device)) double wide(double x) {
  return __poseidon_optimize_tau(Tolerance<code>::value, sq, x);
}

template <unsigned long code, unsigned long other>
__attribute__((device)) double pair(double x) {
  return __poseidon_optimize_tau(Tolerance<code>::value, sq, x);
}

__attribute__((global)) void k(double *x) {
  x[0] = scaled<A, 1>(x[0]);
  x[1] = scaled<A, 2>(x[1]);
  x[2] = scaled<B, 2>(x[2]);
  x[3] = tagged<Tolerance<A>>(x[3]);
  x[4] = wide<A>(x[4]);
  x[5] = pair<A, CAT(CAT(1, A), ul)>(x[5]);
}

// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6scaledIL_Z3tauELi1EEdd.0: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z2sqd.poseidon._Z6scaledIL_Z3tauELi1EEdd.0.fpprofile); left unchanged
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6scaledIL_Z3tauELi2EEdd.0: the profiled program did not contain this site
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6scaledIL_Z3tauELi2EEdd.1: the profiled program did not contain this site
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z6taggedI9ToleranceIL_Z3tauEEEdd.0: the profiled program did not contain this site
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z4wideIL_Z3tauEEdd.0: the profiled program did not contain this site
// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z4pairIL_Z3tauELm1{{[0-9]+}}EEdd.0: the profiled program did not contain this site
