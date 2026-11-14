// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x c -O1 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidon -mllvm -poseidon-profile-use=%t/none -mllvm -poseidon-print %s -o /dev/null 2>&1 | FileCheck %s
// REQUIRES: poseidon

// The accuracy target in C: a literal or a macro.

#include "poseidon/poseidon.h"

#define MY_TOL 3e-8

POSEIDON_OPTIMIZE_TAU(1e-6) double lit(const double *x) { return x[0] * x[0] + x[1]; }
POSEIDON_OPTIMIZE_TAU(MY_TOL) double mac(const double *x) { return x[0] * x[1] + x[0]; }

// CHECK: [poseidon] site lit: accuracy target 1e-06
// CHECK: [poseidon] site mac: accuracy target 3e-08
