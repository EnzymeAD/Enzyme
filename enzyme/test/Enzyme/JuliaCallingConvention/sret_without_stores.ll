; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_without_stores(ptr noalias sret({ ptr addrspace(10) }) %sret, ptr noalias writeonly "enzymejl_returnRoots"="1" %0, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[GEP:%[0-9]+]] = getelementptr ptr addrspace(10), ptr %0, i64 0
; CHECK-NEXT:   store ptr addrspace(10) null, ptr [[GEP]], align 8
; CHECK-NEXT:   ret void

define void @test_without_stores(ptr sret({ ptr addrspace(10) }) %sret, ptr %arg) {
entry:
  ret void
}
