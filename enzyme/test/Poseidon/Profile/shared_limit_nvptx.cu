// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/over -DOVER %s -o /dev/null 2>&1 | FileCheck --check-prefix=OVER %s

// The derivative a profiled kernel runs keeps a shadow next to each of its
// floating-point shared buffers. A kernel whose buffers and their shadows
// exceed the static shared memory a kernel can declare (48 KiB) stops the
// compile with an error that names them.

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

#ifdef OVER
template __attribute__((global)) void stage<4096>(const double *, double *);
#else
template __attribute__((global)) void stage<256>(const double *, double *);
#endif

// GEN: {{^}}[poseidon-profgen] _Z5stageILi256EEvPKdPd: site 0, 2 shadow buffer(s), output(s) 1{{$}}

// OVER: shared_limit_nvptx.cu:[[@LINE-18]]:48: error: Poseidon: void stage<4096>(double const*, double*) reaches 65536 bytes of static shared memory once profiled, above the 49152 a kernel can declare: its own buffers void stage<4096>(double const*, double*)::buf (32768 bytes); the shadows of its buffers void stage<4096>(double const*, double*)::buf (32768 bytes); the shadows of other kernels' buffers none. Profile the kernel with smaller shared buffers, or the kernels whose shadows it reaches in another translation unit.

// IR: @[[SHADOW:_ZZ5stageILi256EEvPKdPdE3buf_shadow]] = {{.*}}addrspace(3) global [256 x double] undef,
// IR: @_ZZ5stageILi256EEvPKdPdE3buf = {{.*}}addrspace(3) global [256 x double] undef, {{.*}}!enzyme_shadow ![[SHMD:[0-9]+]]{{$}}
// IR-LABEL: define {{.*}}ptx_kernel void @_Z5stageILi256EEvPKdPd(
// IR-NEXT: call {{.*}}void @diffepreprocess__Z5stageILi256EEvPKdPd_poseidon_body(
// IR-LABEL: define internal {{.*}}void @diffepreprocess__Z5stageILi256EEvPKdPd_poseidon_body(
// IR: call void @llvm.memset.p3.i64(ptr addrspace(3) {{.*}}@[[SHADOW]], i8 0, i64 2048, i1 false)
// IR: call void @poseidonLogValueCUDA(
// IR: ![[SHMD]] = !{ptr addrspace(3) @[[SHADOW]]}

// DESC: "profile generation": [
// DESC-NEXT: "_Z5stageILi256EEvPKdPd 0 2 2 0 1 0 8"
// DESC-NEXT: ],
