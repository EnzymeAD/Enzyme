; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm" -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm-end" -S | FileCheck %s
;
; An RDC list can contain only compile-time markers, a mixture of markers
; and genuine device symbols, or no markers. Retain the real symbols and
; unknown enzyme-prefixed names. A real use of a marker is not rewritten.

@enzyme_dup = external addrspace(1) global i32
@enzyme_width = external addrspace(1) global i32
@real_device_state = external addrspace(1) global i32
@enzyme_application_state = external addrspace(1) global i32

@__clang_gpu_used_external = internal global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_width to ptr)]
@__clang_gpu_used_external.1 = internal global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), ptr addrspacecast (ptr addrspace(1) @real_device_state to ptr)]
@__clang_gpu_used_external.2 = internal global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_application_state to ptr)]
@llvm.compiler.used = appending global [3 x ptr] [ptr @__clang_gpu_used_external, ptr @__clang_gpu_used_external.1, ptr @__clang_gpu_used_external.2], section "llvm.metadata"

define i32 @unconsumed_marker() {
  %marker = load i32, ptr addrspace(1) @enzyme_dup
  ret i32 %marker
}

; CHECK-NOT: @__clang_gpu_used_external =
; CHECK-DAG: @__clang_gpu_used_external.1 = internal global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @real_device_state to ptr)]
; CHECK-DAG: @__clang_gpu_used_external.2 = internal global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_application_state to ptr)]
; CHECK-DAG: @llvm.compiler.used = appending global [2 x ptr] [ptr @__clang_gpu_used_external.1, ptr @__clang_gpu_used_external.2], section "llvm.metadata"
; CHECK: define i32 @unconsumed_marker()
; CHECK: load i32, ptr addrspace(1) @enzyme_dup
