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

define [2 x double] @test_derivative({ double } %c, double %x, i64 %n) {
entry:
  %r = call [2 x double] (...) @__enzyme_fwddiff(double ({ double }, double, i64)* @tester, metadata !"enzyme_width", i64 2, metadata !"enzyme_const", { double } %c, double %x, double 1.0, double 2.0, i64 %n)
  ret [2 x double] %r
}

declare [2 x double] @__enzyme_fwddiff(...)

; CHECK: define internal [2 x double] @fwddiffe2tester({ double } %c, double %x, [2 x double] %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   br label %loop
; CHECK: loop:
; CHECK-NEXT:   %2 = phi fast double [ %0, %entry ], [ 0.000000e+00, %loop ]
; CHECK-NEXT:   %3 = phi fast double [ %1, %entry ], [ 0.000000e+00, %loop ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %4 = insertvalue [2 x double] undef, double %2, 0
; CHECK-NEXT:   %5 = insertvalue [2 x double] %4, double %3, 1
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cond = icmp slt i64 %iv.next, %n
; CHECK-NEXT:   br i1 %cond, label %loop, label %exit
; CHECK: exit:
; CHECK-NEXT:   ret [2 x double] %5
; CHECK-NEXT: }
