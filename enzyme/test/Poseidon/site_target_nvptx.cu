// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidon -mllvm -poseidon-profile-use=%t/none -mllvm -poseidon-print %s -o %t/k.ll 2>&1 | FileCheck %s
// REQUIRES: poseidon

// The accuracy target on a kernel, in the device compilation of a CUDA source.

#include "poseidon/poseidon.h"

#define MY_TOL 1e-7

POSEIDON_OPTIMIZE_TAU(1e-6) __attribute__((global)) void lit(double *x) { x[0] = x[0] * x[0] + x[1]; }
POSEIDON_OPTIMIZE_TAU(MY_TOL) __attribute__((global)) void mac(double *x) { x[0] = x[0] * x[1] + x[1]; }

// CHECK: [poseidon] site lit(double*): accuracy target 1e-06
// CHECK: [poseidon] site mac(double*): accuracy target 1e-07
