; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_without_stores(ptr noalias sret({ {{.*}} }) %0, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void

define void @test_without_stores({ i8 addrspace(10)* }* sret({ i8 addrspace(10)* }) %sret, i8* %arg) {
entry:
  ret void
}
