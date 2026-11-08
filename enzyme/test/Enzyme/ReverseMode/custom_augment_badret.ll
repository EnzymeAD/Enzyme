; RUN: if [ %llvmver -lt 16 ]; then not %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S 2>&1 | FileCheck %s; fi
; RUN: not %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S 2>&1 | FileCheck %s
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -enzyme-runtime-error=1 -S | FileCheck %s --check-prefix=RUNTIME; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-runtime-error=1 -S | FileCheck %s --check-prefix=RUNTIME

@enzyme_dup = external global i32
@sink = global double 0.000000e+00

declare void @__enzyme_autodiff_void(ptr, ...)

define void @aug_f_bad(double %x) {
entry:
  ret void
}

define double @rev_f_bad(double %x, double %dret) {
entry:
  %two_x = fmul double 2.000000e+00, %x
  %grad = fmul double %two_x, %dret
  ret double %grad
}

define double @f_bad(double %x) !enzyme_augment !0 !enzyme_gradient !1 {
entry:
  store double %x, ptr @sink, align 8
  %sq = fmul double %x, %x
  ret double %sq
}

define void @caller_bad(ptr %x) {
entry:
  %x0 = load double, ptr %x, align 8
  %p1 = getelementptr inbounds double, ptr %x, i64 1
  %x1 = load double, ptr %p1, align 8
  %s = fadd double %x0, %x1
  %r = call double @f_bad(double %s)
  %p2 = getelementptr inbounds double, ptr %x, i64 2
  store double %r, ptr %p2, align 8
  ret void
}

define void @test(ptr %x, ptr %dx) {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
  call void (ptr, ...) @__enzyme_autodiff_void(ptr @caller_bad, i32 %dup, ptr %x, ptr %dx)
  ret void
}

!0 = !{ptr @aug_f_bad}
!1 = !{ptr @rev_f_bad}

; CHECK: Bad return type of custom augmented forward pass for function f_bad of type double (double)
; CHECK-NEXT: expected augmented forward function to return primal value of type double
; CHECK-NEXT: Instead found aug_f_bad of type void (double)

; RUNTIME: define internal void @diffecaller_bad(ptr nocapture %x, ptr nocapture %"x'")
; RUNTIME:   %[[PUTS:.+]] = call i32 @puts(ptr
; RUNTIME-NEXT:   call void @exit(i32 1)
; RUNTIME-NEXT:   %r = call fast double @badaugment_f_bad(double %s)
