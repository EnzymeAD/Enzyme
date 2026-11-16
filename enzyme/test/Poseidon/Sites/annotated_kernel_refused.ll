; RUN: rm -rf %t && split-file %s %t
; RUN: not %opt %t/value.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/other_sites -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=VALUE %s
; RUN: not %opt %t/byref.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/other_sites -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=BYREF %s

; A kernel carrying POSEIDON_OPTIMIZE that cannot be made a wrapper around its
; own body is an error: one that returns a value, and an AMDGPU kernel that
; uses an aggregate it takes by reference otherwise than by loading or copying
; from it.

; VALUE: error: {{.*}}Poseidon: an annotated kernel must be a void, non-variadic function
; BYREF: error: {{.*}}Poseidon: k uses its aggregate parameter 1 otherwise than by loading or copying from it, so it cannot be a Poseidon site; copy the parameter into a local object first

;--- value.ll
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [5 x i8] c"k.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @k, ptr @.str, ptr @.str.1, i32 1, ptr null }], section "llvm.metadata"

define ptx_kernel double @k(ptr %x) {
  %v = load double, ptr %x
  %m = fmul double %v, %v
  ret double %m
}

;--- byref.ll
target triple = "amdgcn-amd-amdhsa"

%pair = type { ptr, double }

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [6 x i8] c"k.hip\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @k, ptr @.str, ptr @.str.1, i32 1, ptr null }], section "llvm.metadata"

define amdgpu_kernel void @k(ptr addrspace(4) byref(%pair) align 8 %p) {
  %x = load ptr, ptr addrspace(4) %p
  %s = getelementptr inbounds %pair, ptr addrspace(4) %p, i32 0, i32 1
  %v = load double, ptr addrspace(4) %s
  %m = fmul double %v, %v
  store double %m, ptr %x
  call void @keep(ptr addrspace(4) %p)
  ret void
}

declare void @keep(ptr addrspace(4))
