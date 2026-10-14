; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s 2>&1 | FileCheck %s

; The value stored into the sret is built by inserting a scalar into an undef
; aggregate and then inserting the tracked pointer, which the callee also
; stores into the existing returnRoots. The undef base holds no live pointer,
; so the tracked pointer is fully rooted and the sret must not be rerooted:
; the returnRoots is assigned to the sret rather than the merged sret gaining
; a further rerooted member.

; CHECK-NOT: failed to find extracted pointer

; CHECK-LABEL: define {{.*}} @caller(
; CHECK: %stack_sret = alloca { {{[^,]+}}, i64 }, align 8
; CHECK: %stack_roots_AT = alloca [1 x {{.*}}]
; CHECK: call void @callee({{.*}} sret({ {{[^,]+}}, i64 }) %stack_sret, {{.*}} "enzymejl_returnRoots"="1" %stack_roots_AT, {{.*}} %v)

; CHECK-LABEL: define internal void @callee({{.*}} sret({ {{[^,]+}}, i64 }) %0, {{.*}} "enzymejl_returnRoots"="1" %1, {{.*}} %v)
; CHECK: store {{.*}} %v, {{.*}} %r0
; CHECK: store { {{[^,]+}}, i64 } %t1, {{.*}} %0
; CHECK-NEXT: ret void

declare void @use([1 x {} addrspace(10)*]*)

define void @caller({} addrspace(10)* %v) {
entry:
  %sret = alloca { {} addrspace(10)*, i64 }, align 8
  %roots = alloca [1 x {} addrspace(10)*], align 8
  call void @callee({ {} addrspace(10)*, i64 }* "enzyme_sret"="test_type6" %sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %roots, {} addrspace(10)* %v)
  call void @use([1 x {} addrspace(10)*]* %roots)
  ret void
}

define internal void @callee({ {} addrspace(10)*, i64 }* noalias writeonly "enzyme_sret"="test_type6" %sret, [1 x {} addrspace(10)*]* noalias writeonly "enzymejl_returnRoots"="1" %roots, {} addrspace(10)* %v) {
entry:
  %r0 = getelementptr inbounds [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %roots, i64 0, i64 0
  store {} addrspace(10)* %v, {} addrspace(10)** %r0, align 8
  %t0 = insertvalue { {} addrspace(10)*, i64 } undef, i64 7, 1
  %t1 = insertvalue { {} addrspace(10)*, i64 } %t0, {} addrspace(10)* %v, 0
  store { {} addrspace(10)*, i64 } %t1, { {} addrspace(10)*, i64 }* %sret, align 8
  ret void
}
