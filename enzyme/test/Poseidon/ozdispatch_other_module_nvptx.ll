; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: sed '1i CanonicalHash = fea337c617ef94c9' %S/Inputs/raise_tcec/preprocess_matmul_body.fpprofile > %t/prof/preprocess_matmul_body.fpprofile
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/same.ll
; RUN: sed 's/ozaki_stub\.cu/other_matmul.cu/g' %S/Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/other.ll
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=DEV %s
; RUN: %clang -O0 -x ir %t/same.ll %clangLoadPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/c -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=SAME %s
; RUN: %clang -O0 -x ir %t/other.ll %clangLoadPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-cache=%t/c -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=OTHER --implicit-check-not=__poseidon_ozaki_dgemm %s
; REQUIRES: poseidon, scev-trunc-iv-nowrap
; XFAIL: *
; A .ozdispatch descriptor written for a kernel of another module with the same name rewrites this module's stub.

; The device compile of tcec_matmul.cu leaves a descriptor for matmul_kernel in
; the cache. The host compile of tcec_matmul.cu applies it; the host compile of
; other_matmul.cu, whose kernel has the same mangled name, must not.

; DEV: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; SAME: [ozaki-host-dispatch] replaced stub body _Z28__device_stub__matmul_kernelPKdS0_PdS1_ -> __poseidon_ozaki_dgemm
; SAME: call void @__poseidon_ozaki_dgemm(ptr %2, ptr %0, ptr %1, i32 16, {{.*}}, i32 12)

; OTHER: source_filename = "other_matmul.cu"
; OTHER: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; OTHER: call {{.*}} @cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernelPKdS0_PdS1_
