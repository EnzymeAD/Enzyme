; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_noop(ptr sret({ ptr addrspace(10) }) %sret, ptr "enzymejl_returnRoots"="1" %rroots)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void

define void @test_noop(ptr sret({ ptr addrspace(10) }) %sret, ptr "enzymejl_returnRoots"="1" %rroots) {
entry:
  ret void
}
