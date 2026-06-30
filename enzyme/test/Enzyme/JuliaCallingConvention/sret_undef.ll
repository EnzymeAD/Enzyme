; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s
; XFAIL: *

; CHECK-LABEL: define void @caller
; CHECK: call void @callee({ {} addrspace(10)* }* undef, { {} addrspace(10)* }* %arg)

define void @caller({ {} addrspace(10)* }* %arg) {
entry:
  call void @callee({ {} addrspace(10)* }* "enzyme_sret"="test_type" "enzyme_type"="{[-1]:Pointer}" undef, { {} addrspace(10)* }* "enzyme_sret"="test_type" "enzyme_type"="{[-1]:Pointer}" %arg)
  ret void
}

define void @callee({ {} addrspace(10)* }* "enzyme_sret"="test_type" "enzyme_type"="{[-1]:Pointer}" %sret_return, { {} addrspace(10)* }* "enzyme_sret"="test_type" "enzyme_type"="{[-1]:Pointer}" %sret_return_prime) {
entry:
  ret void
}
