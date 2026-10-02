; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; A loop marked by __enzyme_set_fixed_point (what a Fortran !DIR$ ENZYME
; FIXED_POINT directive lowers to), tested in its header as DO WHILE is.
; One iteration is outlined as the step, which returns whether the loop goes
; on, and a turn step that never leaves is what the adjoint differentiates.
; The error carried from one iteration to the next goes through the stack.

@u = global [2 x double] zeroinitializer, align 16
@p = global double 5.000000e-01, align 8
@enzyme_const = external global i32, align 4

declare void @__enzyme_set_fixed_point(double, i64, ptr, ...)
declare double @llvm.fabs.f64(double)

define double @f(double %tol) {
entry:
  br label %header

header:
  %err = phi double [ 1.000000e+00, %entry ], [ %err.next, %body ]
  %go = fcmp ogt double %err, %tol
  br i1 %go, label %body, label %exit

body:
  call void (double, i64, ptr, ...) @__enzyme_set_fixed_point(double -1.000000e+00, i64 -1, ptr null, ptr @u, i64 16)
  %a = load double, ptr @u, align 8
  %b = load double, ptr getelementptr inbounds (double, ptr @u, i64 1), align 8
  %pp = load double, ptr @p, align 8
  %ab = fmul double %a, %b
  %na = fadd double %ab, %pp
  %nb = fmul double %a, 5.000000e-01
  store double %na, ptr @u, align 8
  store double %nb, ptr getelementptr inbounds (double, ptr @u, i64 1), align 8
  %d = fsub double %na, %a
  %err.next = call double @llvm.fabs.f64(double %d)
  br label %header

exit:
  %r = load double, ptr @u, align 8
  ret double %r
}

define void @df(double %tol) {
entry:
  %0 = load i32, ptr @enzyme_const, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @f, i32 %0, double %tol)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; The loop is now a call; its state @u and the stack slot of the error are
; snapshot regions, and the defaults are filled in.
; CHECK: define double @f(double %tol)
; CHECK:   call void @enzyme.ckpt.fixedpoint.f.fp.step(i64 0, i64 1000, ptr null, double 0x3D719799812DEA11, ptr @u, i64 16, ptr %err.reg2mem, i64 8, ptr %err.reg2mem, double %tol)
; CHECK-NEXT:   br label %exit

; CHECK: define internal i32 @f.fp.step(i64 %k, ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %err.reg2mem, double %tol) !enzyme_fixed_point_turn ![[TURN:[0-9]+]]
; CHECK: header:
; CHECK:   br i1 %go, label %body, label %done
; CHECK: body:
; CHECK:   store double %err.next, ptr %err.reg2mem
; CHECK-NEXT:   br label %next
; CHECK: next:
; CHECK-NEXT:   ret i32 1
; CHECK: done:
; CHECK-NEXT:   ret i32 0

; The turn always runs the body, even at the converged state.
; CHECK: define internal i32 @f.fp.step.turn(
; CHECK: header:
; CHECK:   br label %body
; CHECK-NOT: ret i32 0
; CHECK: ret i32 1

; CHECK: define internal void @diffef(
; CHECK:   call void @diffeenzyme.ckpt.fixedpoint.f.fp.step(i64 0, i64 1000, ptr null, double 0x3D719799812DEA11, ptr @u, ptr @u_shadow, i64 16, ptr %err.reg2mem, i64 8, ptr %err.reg2mem, ptr %"err.reg2mem'ipa", double %tol)

; The adjoint iterations differentiate the turn.
; CHECK: define internal void @diffef.fp.step.turn(

; CHECK: define internal void @enzyme.ckpt.turn.enzyme.ckpt.fixedpoint.f.fp.step.dc(ptr %0, i64 %1)
; CHECK:   call void @diffef.fp.step.turn(

; CHECK: ![[TURN]] = !{ptr @f.fp.step.turn}
