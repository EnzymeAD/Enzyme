; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: sed '1i CanonicalHash = fea337c617ef94c9' %S/Inputs/raise_tcec/preprocess_matmul_body.fpprofile > %t/prof/preprocess_matmul_body.fpprofile
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/host.ll
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/c -poseidon-unit=tcec -S -o /dev/null 2>&1 | FileCheck --check-prefix=DEV1 %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=tcec -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST1 %s
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/c -poseidon-unit=tcec -S -o /dev/null 2>&1 | FileCheck --check-prefix=DEV2 %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=tcec -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST2 --implicit-check-not=__poseidon_ozaki_dgemm %s

; A device compile that dispatches the product writes its descriptor into the
; manifest of the unit; a later device compile of the same kernel and unit into
; the same cache that dispatches nothing (raising off) must leave the host
; compile's launch stub unchanged. The halves are compiled with different
; options, so all four compiles name the unit.

; DEV1: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; HOST1: [ozaki-host-dispatch] replaced stub body _Z28__device_stub__matmul_kernelPKdS0_PdS1_ -> __poseidon_ozaki_dgemm
; HOST1: call void @__poseidon_ozaki_dgemm(ptr %2, ptr %0, ptr %1, i32 16, {{.*}}, i32 12)

; DEV2-NOT: wrote descriptor
; DEV2: [poseidon] no rewrite applied for preprocess_matmul_body
; DEV2-NOT: wrote descriptor

; HOST2: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST2: call {{.*}} @cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernelPKdS0_PdS1_
