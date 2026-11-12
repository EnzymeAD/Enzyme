; A build of Poseidon that carries no device profiler for a GPU cannot profile
; that GPU's code: the compile that would link the profiler into an
; instrumented device module stops at its first probe, naming what the build
; lacks, instead of leaving the probes for the program's link to find
; undefined. A module without a probe compiles.
; UNSUPPORTED: nvptx-device-profiler && amdgpu-device-profiler
; RUN: rm -rf %t && split-file %s %t
; RUN: %if !nvptx-device-profiler %{ not %opt %t/nvptx.ll %loadPoseidon -passes=poseidon-finalize -poseidon-profile-generate -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=NVPTX --implicit-check-not=PLEASE %s %}
; RUN: %if !nvptx-device-profiler %{ %opt %t/nvptx_noprobe.ll %loadPoseidon -passes=poseidon-finalize -poseidon-profile-generate -poseidon-cache=%t/c -disable-output %}
; RUN: %if !amdgpu-device-profiler %{ not %opt %t/amdgpu.ll %loadPoseidon -passes=poseidon-finalize -poseidon-profile-generate -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=AMDGPU --implicit-check-not=PLEASE %s %}

; NVPTX: error: {{.*}}k{{.*}}Poseidon: this build of Poseidon has no NVPTX device profiler
; AMDGPU: error: {{.*}}k{{.*}}Poseidon: this build of Poseidon has no AMDGPU device profiler

;--- nvptx.ll
target triple = "nvptx64-nvidia-cuda"

@poseidon_site_k = private unnamed_addr addrspace(4) constant [2 x i8] c"k\00"

define ptx_kernel void @k(ptr addrspace(1) %x) {
entry:
  %v = load double, ptr addrspace(1) %x, align 8
  call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(4) @poseidon_site_k to ptr), i64 0, double %v, i64 0, ptr null)
  ret void
}

declare void @poseidonLogValueCUDA(ptr, i64, double, i64, ptr)

;--- nvptx_noprobe.ll
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @k(ptr addrspace(1) %x) {
entry:
  store double 1.0, ptr addrspace(1) %x, align 8
  ret void
}

;--- amdgpu.ll
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@poseidon_site_k = private unnamed_addr addrspace(4) constant [2 x i8] c"k\00"

define amdgpu_kernel void @k(ptr addrspace(1) %x) #0 {
entry:
  %v = load double, ptr addrspace(1) %x, align 8
  call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(4) @poseidon_site_k to ptr), i64 0, double %v, i64 0, ptr null)
  ret void
}

declare void @poseidonLogValueCUDA(ptr, i64, double, i64, ptr)

attributes #0 = { "target-cpu"="gfx942" }
