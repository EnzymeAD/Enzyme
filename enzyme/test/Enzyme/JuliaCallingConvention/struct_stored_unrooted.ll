; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s 2>&1 | FileCheck %s

; An aggregate returned by a call is stored whole into the sret, and its tracked
; field is never extracted, so nothing shows that it is stored into the
; returnRoots. The struct handling must reroot the sret conservatively rather
; than falling through to the pointer-shaped handling, which used to assert with
; "Pointer of wrong type".

; CHECK-NOT: Pointer of wrong type

; CHECK-LABEL: define {{.*}} @caller(
; CHECK: %stack_sret = alloca { { {{[^,]+}}, i64 }, [1 x {{.*}}] }
; CHECK: %stack_roots_AT = alloca [2 x {{.*}}]
; CHECK: call void @callee({{.*}} sret({{.*}}) %stack_sret, {{.*}} "enzymejl_returnRoots"="2" %stack_roots_AT, {{.*}} %x)

; CHECK-LABEL: define internal void @callee({{.*}} sret({ { {{[^,]+}}, i64 }, [1 x {{.*}}] }) %0, {{.*}} "enzymejl_returnRoots"="2" %1, {{.*}} %x)
; CHECK: [[SRET:%[0-9]+]] = getelementptr inbounds { { {{[^,]+}}, i64 }, [1 x {{.*}}] }, {{.*}} %0, i32 0, i32 0
; CHECK: %a = call %pair @make(
; CHECK-NEXT: store %pair %a, {{.*}} [[SRET]]
; CHECK-NEXT: [[R0:%[0-9]+]] = getelementptr inbounds [2 x {{.*}}], {{.*}} %1, i64 0, i32 0
; CHECK-NEXT: [[P:%[0-9]+]] = getelementptr inbounds { {{[^,]+}}, i64 }, {{.*}} [[SRET]], i64 0, i32 0
; CHECK-NEXT: [[V:%[0-9]+]] = load {{.*}}, {{.*}} [[P]]
; CHECK-NEXT: store {{.*}} [[V]], {{.*}} [[R0]]
; CHECK-NEXT: ret void

%pair = type { {} addrspace(10)*, i64 }

declare %pair @make({} addrspace(10)*)
declare void @use([1 x {} addrspace(10)*]*)

define {} addrspace(10)* @caller({} addrspace(10)* %x) {
entry:
  %sret = alloca %pair, align 8
  %roots = alloca [1 x {} addrspace(10)*], align 8
  call void @callee(%pair* "enzyme_sret"="test_type6" %sret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %roots, {} addrspace(10)* %x)
  call void @use([1 x {} addrspace(10)*]* %roots)
  %p = getelementptr inbounds %pair, %pair* %sret, i64 0, i32 0
  %v = load {} addrspace(10)*, {} addrspace(10)** %p, align 8
  ret {} addrspace(10)* %v
}

define internal void @callee(%pair* noalias writeonly "enzyme_sret"="test_type6" %sret, [1 x {} addrspace(10)*]* noalias writeonly "enzymejl_returnRoots"="1" %roots, {} addrspace(10)* %x) {
entry:
  %a = call %pair @make({} addrspace(10)* %x)
  store %pair %a, %pair* %sret, align 8
  ret void
}
