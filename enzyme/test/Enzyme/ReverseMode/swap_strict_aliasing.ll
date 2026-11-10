; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -enzyme-strict-aliasing=0 -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-strict-aliasing=0 -S | FileCheck %s

; Swapping two doubles through i64 loads/stores in a loop whose addresses are
; formed via a chained byte GEP (`getelementptr i8, ptr %d, i64 %.idx` followed
; by `+16` and `+32`) must be recognized as a float swap under
; `-enzyme-strict-aliasing=0` and reversed in the adjoint loop rather than
; mirrored in the forward loop.

@enzyme_const = external global i32
@enzyme_dup = external global i32

declare void @__enzyme_autodiff(...)

define void @f(i64 %n, ptr %d) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %.idx = shl i64 %iv, 5
  %base = getelementptr i8, ptr %d, i64 %.idx
  %a = getelementptr inbounds i8, ptr %base, i64 16
  %b = getelementptr inbounds i8, ptr %base, i64 32
  %va = load i64, ptr %a, align 8
  %vb = load i64, ptr %b, align 8
  store i64 %vb, ptr %a, align 8
  store i64 %va, ptr %b, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp eq i64 %iv.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  %p2 = getelementptr inbounds i8, ptr %d, i64 16
  %v2 = load double, ptr %p2, align 8
  %v0 = load double, ptr %d, align 8
  %add = fadd double %v0, %v2
  store double %add, ptr %d, align 8
  ret void
}

define void @df(i64 %n, ptr %d, ptr %dd) {
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_const", i64 %n, metadata !"enzyme_dup", ptr %d, ptr %dd)
  ret void
}

; CHECK-LABEL: define internal void @diffef(
; CHECK: loop:
; CHECK-NOT: store i64 %{{.*}}, ptr %"a'ipg"
; CHECK-NOT: store i64 %{{.*}}, ptr %"b'ipg"
; CHECK: exit:
; CHECK: invertloop:
; CHECK: %[[B0:.+]] = load i64, ptr %"b'ipg_unwrap"
; CHECK-NEXT: store i64 0, ptr %"b'ipg_unwrap"
; CHECK: %[[A0:.+]] = load i64, ptr %"a'ipg_unwrap"
; CHECK-NEXT: store i64 0, ptr %"a'ipg_unwrap"
; CHECK: store double %{{.+}}, ptr %"b'ipg_unwrap"
; CHECK: store double %{{.+}}, ptr %"a'ipg_unwrap"
