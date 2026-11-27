// REQUIRES: cuda-toolkit
// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-print %s -o %t/left.ll 2>&1 | FileCheck --check-prefix=LEFT %s
// RUN: diff %t/plain.ll %t/left.ll
// RUN: FileCheck %s < %t/left.ll

// A module whose call sites are all left alone is emitted as a build without a
// profile compiles it: the determinant caller and callee share is computed once.

#include "poseidon/poseidon.h"

__attribute__((device)) void vol(const double *a, double *v) {
  double J = a[0] * a[1] - a[2] * a[3];
  *v = J * (J - 1.0);
}

__attribute__((global)) void k(double *x) {
  double a[4] = {x[0], x[1], x[2], x[3]}, v;
  double J = a[0] * a[1] - a[2] * a[3];
  __poseidon_optimize_tau(1e-10, vol, a, &v);
  x[4] = v / J;
}

// LEFT: [poseidon] every site of {{.*}} was left alone; emitting the module as written
// CHECK-LABEL: define {{.*}} @_Z1kPd(
// CHECK-COUNT-1: fsub
// CHECK-NOT: fsub
// CHECK: ret void
