; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; The shadow of the active phi is built before the constant extractvalue is
; visited, which leaves a cached shadow for a constant value behind.

define double @tester({ double } %c, double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi double [ %x, %entry ], [ %cv, %loop ]
  %cv = extractvalue { double } %c, 0
  %inext = add i64 %i, 1
  %cond = icmp slt i64 %inext, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret double %acc
}

define double @test_derivative({ double } %c, double %x, i64 %n) {
entry:
  %r = call double (...) @__enzyme_fwddiff(double ({ double }, double, i64)* @tester, metadata !"enzyme_const", { double } %c, double %x, double 1.0, i64 %n)
  ret double %r
}

declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffetester({ double } %c, double %x, double %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
; CHECK: loop:
; CHECK-NEXT:   %0 = phi fast double [ %"x'", %entry ], [ 0.000000e+00, %loop ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cond = icmp slt i64 %iv.next, %n
; CHECK-NEXT:   br i1 %cond, label %loop, label %exit
; CHECK: exit:
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }
