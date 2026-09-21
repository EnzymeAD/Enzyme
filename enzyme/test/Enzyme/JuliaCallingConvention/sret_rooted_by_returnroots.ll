; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; The primal sret holds a tracked pointer which the callee also stores into the
; existing returnRoots, whereas the tracked pointer of the shadow sret has no
; root. The returnRoots must be assigned to the primal sret instead of being
; returned as a further member of the merged sret: the merged sret then has
; three tracked pointers (return value, primal sret, shadow sret) and the three
; roots to match (return value, shadow sret, primal returnRoots).

; CHECK-LABEL: define {{.*}} @caller(
; CHECK: %stack_sret = alloca { { {{[^,]+}}, double }, { {{[^,]+}}, i64 }, { {{[^,]+}}, i64 } }
; CHECK: %stack_roots_AT = alloca [3 x {{.*}}]
; CHECK: call void @augmented({{.*}} sret({{.*}}) %stack_sret, {{.*}} "enzymejl_returnRoots"="3" %stack_roots_AT, {{.*}} %v, {{.*}} %dv)

; CHECK-LABEL: define internal void @augmented({{.*}} sret({ { {{[^,]+}}, double }, { {{[^,]+}}, i64 }, { {{[^,]+}}, i64 } }) %0, {{.*}} "enzymejl_returnRoots"="3" %1, {{.*}} %v, {{.*}} %dv)
; CHECK: [[RR:%[0-9]+]] = getelementptr inbounds [3 x {{.*}}], {{.*}} %1, i32 0, i32 2
; CHECK: store {{.*}} %v, {{.*}} %r0
; CHECK: [[R0:%[0-9]+]] = getelementptr inbounds [3 x {{.*}}], {{.*}} %1, i64 0, i32 0
; CHECK-NEXT: store {{.*}} %dv, {{.*}} [[R0]]
; CHECK-NEXT: [[R1:%[0-9]+]] = getelementptr inbounds [3 x {{.*}}], {{.*}} %1, i64 0, i32 1
; CHECK: store {{.*}}, {{.*}} [[R1]]
; CHECK-NEXT: ret void

declare void @use([1 x {} addrspace(10)*]*)

define {} addrspace(10)* @caller({} addrspace(10)* %v, {} addrspace(10)* %dv) {
entry:
  %sret = alloca { {} addrspace(10)*, i64 }, align 8
  %dsret = alloca { {} addrspace(10)*, i64 }, align 8
  %roots = alloca [1 x {} addrspace(10)*], align 8
  %tape = call { {} addrspace(10)*, double } @augmented({ {} addrspace(10)*, i64 }* "enzyme_sret"="test_type6" %sret, { {} addrspace(10)*, i64 }* "enzyme_sret"="test_type6" %dsret, [1 x {} addrspace(10)*]* "enzymejl_returnRoots"="1" %roots, {} addrspace(10)* %v, {} addrspace(10)* %dv)
  call void @use([1 x {} addrspace(10)*]* %roots)
  %res = extractvalue { {} addrspace(10)*, double } %tape, 0
  ret {} addrspace(10)* %res
}

define internal { {} addrspace(10)*, double } @augmented({ {} addrspace(10)*, i64 }* noalias writeonly "enzyme_sret"="test_type6" %sret, { {} addrspace(10)*, i64 }* writeonly "enzyme_sret"="test_type6" %dsret, [1 x {} addrspace(10)*]* noalias writeonly "enzymejl_returnRoots"="1" %roots, {} addrspace(10)* %v, {} addrspace(10)* %dv) {
entry:
  %r0 = getelementptr inbounds [1 x {} addrspace(10)*], [1 x {} addrspace(10)*]* %roots, i64 0, i64 0
  store {} addrspace(10)* %v, {} addrspace(10)** %r0, align 8
  %dp = getelementptr inbounds { {} addrspace(10)*, i64 }, { {} addrspace(10)*, i64 }* %dsret, i64 0, i32 0
  %p = getelementptr inbounds { {} addrspace(10)*, i64 }, { {} addrspace(10)*, i64 }* %sret, i64 0, i32 0
  store {} addrspace(10)* %dv, {} addrspace(10)** %dp, align 8
  store {} addrspace(10)* %v, {} addrspace(10)** %p, align 8
  %dk = getelementptr inbounds { {} addrspace(10)*, i64 }, { {} addrspace(10)*, i64 }* %dsret, i64 0, i32 1
  %k = getelementptr inbounds { {} addrspace(10)*, i64 }, { {} addrspace(10)*, i64 }* %sret, i64 0, i32 1
  store i64 7, i64* %dk, align 8
  store i64 7, i64* %k, align 8
  %t0 = insertvalue { {} addrspace(10)*, double } undef, {} addrspace(10)* %dv, 0
  %t1 = insertvalue { {} addrspace(10)*, double } %t0, double 1.000000e+00, 1
  ret { {} addrspace(10)*, double } %t1
}
