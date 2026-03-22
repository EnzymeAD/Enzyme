; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_without_stores(ptr noalias sret({ ptr addrspace(10) }) %0, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void

define void @test_without_stores(ptr sret({ ptr addrspace(10) }) %sret, ptr %arg) {
entry:
  ret void
}
