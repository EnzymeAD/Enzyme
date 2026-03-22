; RUN: %opt %loadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_sret(ptr noalias sret({ ptr addrspace(10) }) %0, ptr noalias writeonly "enzymejl_returnRoots"="1" %1, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[LOAD:%[0-9]+]] = load ptr addrspace(10), ptr %arg, align 8
; CHECK-NEXT:   store ptr addrspace(10) [[LOAD]], ptr %0, align 8
; CHECK-NEXT:   ret void

%jl_value = type opaque

define void @test_sret(ptr sret({ %jl_value addrspace(10)* }) %sret, ptr %arg) {
entry:
  %val = load %jl_value addrspace(10)*, ptr %arg, align 8
  store %jl_value addrspace(10)* %val, ptr %sret, align 8
  ret void
}
