; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/host.ll
; RUN: sed '/^target triple/a @__hip_cuid_0123abcd = global i8 0' %S/Inputs/stale_cache/matmul_device.ll > %t/dev_cuid.ll
; RUN: sed '/^target triple/a @__hip_cuid_0123abcd = global i8 0' %t/host.ll > %t/host_cuid.ll
; RUN: sed '/^target triple/a @__hip_cuid_89abcdef = global i8 0' %t/host.ll > %t/host_other_cuid.ll
;
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-cache=%t/options -S -o /dev/null 2>&1 | FileCheck --check-prefix=WROTE %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/raise_tcec -mllvm -poseidon-enable-herbie=false -mllvm -poseidon-enable-pt=false -mllvm -poseidon-raise-wmma -mllvm -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -mllvm -poseidon-tau=1e-12 -mllvm -poseidon-cache=%t/options -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=DISPATCH %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/raise_tcec -mllvm -poseidon-enable-herbie=false -mllvm -poseidon-enable-pt=false -mllvm -poseidon-raise-wmma -mllvm -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -mllvm -poseidon-tau=1e-7 -mllvm -poseidon-cache=%t/options -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon_ozaki_dgemm %s
;
; RUN: %opt %t/dev_cuid.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-cache=%t/cuid -S -o /dev/null 2>&1 | FileCheck --check-prefix=WROTE %s
; RUN: cat %t/cuid/launches_*.json | FileCheck --check-prefix=CUID %s
; RUN: %clang -O0 -x ir %t/host_cuid.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/raise_tcec -mllvm -poseidon-cache=%t/cuid -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=DISPATCH %s
; RUN: %clang -O0 -x ir %t/host_other_cuid.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/raise_tcec -mllvm -poseidon-cache=%t/cuid -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon_ozaki_dgemm %s
;
; RUN: %opt %t/dev_cuid.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-cache=%t/named -poseidon-unit=u -S -o /dev/null 2>&1 | FileCheck --check-prefix=WROTE %s
; RUN: %clang -O0 -x ir %t/host_other_cuid.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/raise_tcec -mllvm -poseidon-cache=%t/named -mllvm -poseidon-unit=u -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=DISPATCH %s
; RUN: %clang -O0 -x ir %t/host_cuid.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/raise_tcec -mllvm -poseidon-cache=%t/named -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon_ozaki_dgemm %s

; How the two halves of a compile name their unit when no -poseidon-unit is
; given: by the values of every Poseidon flag, so a host compile with the
; device compile's flags takes its manifest and one with another tolerance
; does not; and, ahead of that, by HIP's compilation-unit id, which ties halves
; compiled with different flags and keeps apart halves with different ids. A
; -poseidon-unit both halves are given takes precedence over the id.

; WROTE: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; CUID: "unit": "0123abcd"

; DISPATCH: [ozaki-host-dispatch] replaced stub body _Z28__device_stub__matmul_kernelPKdS0_PdS1_ -> __poseidon_ozaki_dgemm
; DISPATCH: call void @__poseidon_ozaki_dgemm(ptr %2, ptr %0, ptr %1, i32 16, {{.*}}, i32 12)

; LAUNCH: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; LAUNCH: call {{.*}} @cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernelPKdS0_PdS1_
