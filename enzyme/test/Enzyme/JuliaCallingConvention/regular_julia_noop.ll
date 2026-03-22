; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_noop(ptr {{.*}}sret({ {{.*}} }) {{.*}}, {{.*}}"enzymejl_returnRoots"="1" {{.*}})
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void

define void @test_noop({ {} addrspace(10)* }* sret({ {} addrspace(10)* }) %sret, {} addrspace(10)** "enzymejl_returnRoots"="1" %rroots) {
entry:
  ret void
}
