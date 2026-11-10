// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// REQUIRES: poseidon, enzyme

// The derivative a profiled kernel runs keeps a shadow next to each of its
// floating-point shared buffers. A kernel whose buffers would then exceed the
// static shared memory a kernel can declare (48 KiB) is not profiled and keeps
// its body; the others are profiled as usual.

#include "poseidon/poseidon.h"

#define SHARED __attribute__((shared))

template <int N>
__attribute__((global)) POSEIDON_OPTIMIZE void stage(const double *x,
                                                     double *y) {
  SHARED double buf[N];
  int t = __nvvm_read_ptx_sreg_tid_x();
  for (int i = t; i < N; i += __nvvm_read_ptx_sreg_ntid_x())
    buf[i] = x[i] * x[i];
  for (int i = t; i < N; i += __nvvm_read_ptx_sreg_ntid_x())
    y[i] = buf[N - 1 - i] * 3.0;
}

template __attribute__((global)) void stage<256>(const double *, double *);
template __attribute__((global)) void stage<4096>(const double *, double *);

// GEN-NOT: [poseidon-profgen] _Z5stageILi4096EEvPKdPd
// GEN: warning: {{.*}}Poseidon: void stage<4096>(double const*, double*) would declare 65536 bytes of static shared memory once profiled (a shadow next to each floating-point shared buffer), above the 49152 a kernel can declare. The kernel is not profiled and stays as written.
// GEN-NOT: [poseidon-profgen] _Z5stageILi4096EEvPKdPd
// GEN: [poseidon-profgen] _Z5stageILi256EEvPKdPd: site 0, 2 shadow buffer(s)
// GEN-NOT: [poseidon-profgen] _Z5stageILi4096EEvPKdPd

// DESC: _Z5stageILi256EEvPKdPd 0
// DESC-NOT: _Z5stageILi4096EEvPKdPd
