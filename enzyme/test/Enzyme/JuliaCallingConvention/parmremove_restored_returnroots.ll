; RUN: if [ %llvmver -ge 15 ]; then %opt %OPnewLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s; fi

; When dead-argument elimination removes an unused enzymejl_returnRoots
; parameter (original arg 1) from a Julia sret function returning a struct that
; mixes GC-tracked pointers and non-tracked fields, it records "1" in
; "enzyme_parmremove". Once FixupJuliaCallingConvention re-inserts the
; enzymejl_returnRoots parameter at arg 1, "1" must be stripped from
; "enzyme_parmremove" so downstream Julia rule dispatch maps the remaining
; parameters to the right original positions (EnzymeAD/Enzyme.jl#2305).

%mixed_sret = type { ptr addrspace(10), i64 }

; CHECK: define void @only_roots_removed(ptr noalias sret(%mixed_sret) %[[SRET0:[0-9]+]], ptr noalias writeonly "enzymejl_returnRoots"="1" %[[ROOTS0:[0-9]+]], ptr addrspace(10) %obj, i64 %tag) #[[ATTR0:[0-9]+]]
; CHECK: define void @roots_and_arg_removed(ptr noalias sret(%mixed_sret) %[[SRET1:[0-9]+]], ptr noalias writeonly "enzymejl_returnRoots"="1" %[[ROOTS1:[0-9]+]], ptr addrspace(10) %obj) #[[ATTR1:[0-9]+]]

; CHECK: attributes #[[ATTR0]] = { {{.*}}"enzymejl_mi"="12345"{{.*}} }
; CHECK: attributes #[[ATTR1]] = { {{.*}}"enzyme_parmremove"="2" "enzymejl_mi"="67890"{{.*}} }
; CHECK-NOT: "enzyme_parmremove"="1"
; CHECK-NOT: "enzyme_parmremove"="1,2"

define void @only_roots_removed(ptr noalias nocapture writeonly sret(%mixed_sret) %sret, ptr addrspace(10) %obj, i64 %tag) #0 {
entry:
  %f0 = getelementptr inbounds %mixed_sret, ptr %sret, i64 0, i32 0
  store ptr addrspace(10) %obj, ptr %f0, align 8
  %f1 = getelementptr inbounds %mixed_sret, ptr %sret, i64 0, i32 1
  store i64 %tag, ptr %f1, align 8
  ret void
}

define void @roots_and_arg_removed(ptr noalias nocapture writeonly sret(%mixed_sret) %sret, ptr addrspace(10) %obj) #1 {
entry:
  %f0 = getelementptr inbounds %mixed_sret, ptr %sret, i64 0, i32 0
  store ptr addrspace(10) %obj, ptr %f0, align 8
  %f1 = getelementptr inbounds %mixed_sret, ptr %sret, i64 0, i32 1
  store i64 7, ptr %f1, align 8
  ret void
}

attributes #0 = { "enzyme_parmremove"="1" "enzymejl_mi"="12345" }
attributes #1 = { "enzyme_parmremove"="1,2" "enzymejl_mi"="67890" }
