; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=0 -enzyme -mem2reg  -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=0 -passes="enzyme,function(mem2reg)" -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

define double @sfunc(double %x, i32 %val) {
entry:
  switch i32 %val, label %b1 [
    i32 0, label %err
    i32 1, label %err
  ]

err:
  unreachable

b1:
  %q = sitofp i32 %val to double
  %m = fmul double %q, %x
  ret double %m
}

define double @outer(double %x, i32 %val) {
  %v = call double @sfunc(double %x, i32 %val)
  %m = fmul double %v, %v
  ret double %m
}

define void @main(double %q, i32 %val) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double, i32)* @outer to i8*), double %q, i32 %val)
  ret void
}

; CHECK: define internal { double } @diffesfunc(double %x, i32 %val, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   switch i32 2, label %b1 [
; CHECK-NEXT:     i32 0, label %err
; CHECK-NEXT:     i32 1, label %err
; CHECK-NEXT:   ]

; CHECK: err:                                              ; preds = %entry, %entry
; CHECK-NEXT:   unreachable

; CHECK: b1:                                               ; preds = %entry
; CHECK-NEXT:   %q = sitofp i32 %val to double
; CHECK-NEXT:   br label %invertb1

; CHECK: invertentry:                                      ; preds = %invertb1
; CHECK-NEXT:   %0 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %0

; CHECK: invertb1:                                         ; preds = %b1
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %q
; CHECK-NEXT:   %2 = fadd fast double 0.000000e+00, %1
; CHECK-NEXT:   br label %invertentry
; CHECK-NEXT: }
