// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=IR --implicit-check-not=_ZZ5stageILi4096EEvPKdPdE3buf_shadow %s < %t/gen.ll
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC --implicit-check-not=_Z5stageILi4096EEvPKdPd %s

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
// GEN: shared_limit_nvptx.cu:[[@LINE-14]]:48: warning: Poseidon: void stage<4096>(double const*, double*) would declare 65536 bytes of static shared memory once profiled (a shadow next to each floating-point shared buffer), above the 49152 a kernel can declare. The kernel is not profiled and stays as written.
// GEN-NOT: [poseidon-profgen] _Z5stageILi4096EEvPKdPd
// GEN: {{^}}[poseidon-profgen] _Z5stageILi256EEvPKdPd: site 0, 2 shadow buffer(s), output(s) 1{{$}}
// GEN-NOT: [poseidon-profgen] _Z5stageILi4096EEvPKdPd

// IR: @[[SHADOW:_ZZ5stageILi256EEvPKdPdE3buf_shadow]] = {{.*}}addrspace(3) global [256 x double] undef,
// IR: @_ZZ5stageILi256EEvPKdPdE3buf = {{.*}}addrspace(3) global [256 x double] undef, {{.*}}!enzyme_shadow ![[SHMD:[0-9]+]]{{$}}
// IR: @_ZZ5stageILi4096EEvPKdPdE3buf = {{.*}}addrspace(3) global [4096 x double] undef, comdat, align 8{{$}}
// IR-LABEL: define {{.*}}ptx_kernel void @_Z5stageILi4096EEvPKdPd(
// IR-SAME: ptr {{[^,]*}}[[X:%[0-9]+]], ptr {{[^,]*}}[[Y:%[0-9]+]]) {{.*}}{
// IR-NOT: call {{.*}}@{{poseidon|diffe}}
// IR: [[XP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[X]], i64 {{%[0-9]+}}
// IR-NEXT: [[XV:%[0-9]+]] = load double, ptr [[XP]],
// IR-NEXT: [[SQ:%[0-9]+]] = fmul contract double [[XV]], [[XV]]
// IR-NEXT: [[BP:%[0-9]+]] = getelementptr inbounds {{.*}}@_ZZ5stageILi4096EEvPKdPdE3buf to ptr), i64 {{%[0-9]+}}
// IR-NEXT: store double [[SQ]], ptr [[BP]],
// IR-NOT: call {{.*}}@{{poseidon|diffe}}
// IR: [[BP2:%[0-9]+]] = getelementptr inbounds {{.*}}@_ZZ5stageILi4096EEvPKdPdE3buf to ptr), i64 {{%[0-9]+}}
// IR-NEXT: [[BV:%[0-9]+]] = load double, ptr [[BP2]],
// IR-NEXT: [[M:%[0-9]+]] = fmul contract double [[BV]], 3.000000e+00
// IR-NEXT: [[YP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 {{%[0-9]+}}
// IR-NEXT: store double [[M]], ptr [[YP]],
// IR-NOT: call {{.*}}@{{poseidon|diffe}}
// IR: {{^}}}
// IR-LABEL: define {{.*}}ptx_kernel void @_Z5stageILi256EEvPKdPd(
// IR-NEXT: call {{.*}}void @diffepreprocess__Z5stageILi256EEvPKdPd_poseidon_body(
// IR-LABEL: define internal {{.*}}void @diffepreprocess__Z5stageILi256EEvPKdPd_poseidon_body(
// IR: call void @llvm.memset.p3.i64(ptr addrspace(3) {{.*}}@[[SHADOW]], i8 0, i64 2048, i1 false)
// IR: call void @poseidonLogValueCUDA(
// IR: ![[SHMD]] = !{ptr addrspace(3) @[[SHADOW]]}

// DESC: "profile generation": [
// DESC-NEXT: "_Z5stageILi256EEvPKdPd 0 2 2 0 1 0 8"
// DESC-NEXT: ],
// DESC-NOT: _Z5stageILi4096EEvPKdPd
