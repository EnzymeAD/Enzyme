; REQUIRES: enzyme, nvptx-device-profiler
; RUN: rm -rf %t && mkdir -p %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/host.ll
; RUN: grep -v llvm.global.annotations %S/../Inputs/stale_cache/matmul_annotated_device.ll > %t/plain.ll
; RUN: %opt %S/../Inputs/stale_cache/matmul_annotated_device.ll %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,globaldce" -enzyme-preopt=false -poseidon-profile-generate -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=DEV1 %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST1 %s
; RUN: %opt %t/plain.ll %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,globaldce" -enzyme-preopt=false -poseidon-profile-generate -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=DEV2 %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST2 --implicit-check-not=__poseidon_launch_profiled %s

; A profile-generate device compile of an annotated kernel writes its
; descriptor into the manifest of the unit; after the annotation is removed, a
; device compile into the same cache writes a manifest without it, and the host
; compile must launch the kernel as written instead of with shadow buffers it
; no longer takes.

; DEV1: [poseidon-profgen] _Z13matmul_kernelPKdS0_PdS1_: site 0, 4 shadow buffer(s), output(s) 2

; HOST1: [poseidon-profgen] launch of _Z13matmul_kernelPKdS0_PdS1_ now allocates and seeds 4 shadow buffer(s)
; HOST1: call void @__poseidon_launch_profiled(ptr @_Z28__device_stub__matmul_kernelPKdS0_PdS1_,

; DEV2-NOT: [poseidon-profgen]

; HOST2: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST2: call {{.*}} @cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernelPKdS0_PdS1_
