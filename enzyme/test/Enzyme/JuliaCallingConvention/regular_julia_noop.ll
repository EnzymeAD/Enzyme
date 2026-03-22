; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_noop(ptr sret({ {{.*}} }) %sret, ptr "enzymejl_returnRoots"="1" %rroots)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void

define void @test_noop({ i8 addrspace(10)* }* sret({ i8 addrspace(10)* }) %sret, i8** "enzymejl_returnRoots"="1" %rroots) {
entry:
  ret void
}
