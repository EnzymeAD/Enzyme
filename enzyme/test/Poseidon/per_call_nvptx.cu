// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidon -mllvm -poseidon-profile-use=%t/none %s -o %t/k.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/k.ll
// REQUIRES: poseidon

// In device code a per-call site has no effect: kernels are marked with
// POSEIDON_OPTIMIZE. The call runs as written.

#include "poseidon/poseidon.h"

__attribute__((device)) double sq(double x) { return x * x + 1.0; }

__attribute__((global)) void k(double *x) {
  x[0] = __poseidon_optimize(sq, x[1]);
}

// CHECK: warning: {{.*}}Poseidon: __poseidon_optimize has no effect in device code; a kernel is marked with POSEIDON_OPTIMIZE. The call of sq(double) runs as written.
// IR-LABEL: define {{.*}} @_Z1kPd(
// IR: fmul
// IR-NOT: __poseidon_site
