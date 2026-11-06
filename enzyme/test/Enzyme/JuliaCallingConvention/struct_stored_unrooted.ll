; RUN: not %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s 2>&1 | FileCheck %s

; An aggregate returned by a call is stored whole into the sret, and its tracked
; field is never extracted, so nothing shows that it is stored into the
; returnRoots. Without a custom error handler this must be reported as a fatal
; error rather than falling through to the pointer-shaped handling, which used
; to assert with "Pointer of wrong type".

; CHECK-NOT: Pointer of wrong type
; CHECK: Could not find use of stored value
; CHECK-NEXT: sv: {{.*}} = call { {{[^,]+}}, i64 } @make(

declare { {} addrspace(10)*, i64 } @make({} addrspace(10)*)
declare void @use([1 x {} addrspace(10)*]*)

define {} addrspace(10)* @caller({} addrspace(10)* %x) {
entry:
  %sret = alloca { {} addrspace(10)*, i64 }, align 8
  %roots = alloca [1 x {} addrspace(10)*], align 8
  call void @callee({ {} addrspace(10)*, i64 }* "enzyme_sret"="test_type6" %sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %roots, {} addrspace(10)* %x)
  call void @use([1 x {} addrspace(10)*]* %roots)
  %p = getelementptr inbounds { {} addrspace(10)*, i64 }, { {} addrspace(10)*, i64 }* %sret, i64 0, i32 0
  %v = load {} addrspace(10)*, {} addrspace(10)** %p, align 8
  ret {} addrspace(10)* %v
}

define internal void @callee({ {} addrspace(10)*, i64 }* noalias writeonly "enzyme_sret"="test_type6" %sret, [1 x {} addrspace(10)*]* noalias writeonly "enzymejl_returnRoots"="1" %roots, {} addrspace(10)* %x) {
entry:
  %a = call { {} addrspace(10)*, i64 } @make({} addrspace(10)* %x)
  store { {} addrspace(10)*, i64 } %a, { {} addrspace(10)*, i64 }* %sret, align 8
  ret void
}
