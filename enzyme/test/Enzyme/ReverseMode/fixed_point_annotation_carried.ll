; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; A fixed-point loop as GVN leaves a DO WHILE whose condition does the work:
; load PRE carries the state @u from one iteration to the next in %a rather
; than through memory, and the parameter @p in %pp, reloaded each iteration
; from memory the loop does not write. %a is state as @u is, so the adjoint
; iterations measure and finally drop its shadow too; %pp is not, so its
; adjoint flows back out of the loop to @p.

@u = global double 0.000000e+00, align 8
@p = global double 5.000000e-01, align 8
@enzyme_const = external global i32, align 4

declare void @__enzyme_set_fixed_point(double, i64, ptr, ...)
declare double @llvm.fabs.f64(double)
declare double @llvm.sin.f64(double)

define double @f(double %tol) {
entry:
  %a0 = load double, ptr @u, align 8
  %p0 = load double, ptr @p, align 8
  br label %header

header:
  %a = phi double [ %a0, %entry ], [ %a.pre, %body ]
  %pp = phi double [ %p0, %entry ], [ %pp.pre, %body ]
  %s = call double @llvm.sin.f64(double %a)
  %sp = fmul double %s, 3.000000e-01
  %pq = fmul double %pp, %pp
  %na = fadd double %sp, %pq
  store double %na, ptr @u, align 8
  %d = fsub double %na, %a
  %err = call double @llvm.fabs.f64(double %d)
  %go = fcmp ogt double %err, %tol
  br i1 %go, label %body, label %exit

body:
  call void (double, i64, ptr, ...) @__enzyme_set_fixed_point(double -1.000000e+00, i64 -1, ptr null, ptr @u, i64 8)
  %a.pre = load double, ptr @u, align 8
  %pp.pre = load double, ptr @p, align 8
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

; The states are @u and the slot of %a; the slot of %pp is a region only.
; CHECK: define double @f(double %tol)
; CHECK:   call void @enzyme.ckpt.fixedpoint.f.fp.step(i64 0, i64 1000, ptr null, double 0x3D719799812DEA11, ptr @u, i64 8, ptr %a.reg2mem, i64 8, ptr %pp.reg2mem, i64 8, ptr %pp.reg2mem, ptr %a.reg2mem, double %tol)

; CHECK: call void @__enzyme_fp_rev(ptr %handle, ptr %regions, i64 4, i64 %{{.*}}, ptr %env, ptr @enzyme.ckpt.turn.enzyme.ckpt.fixedpoint.f.fp.step.ddc, ptr %states, i64 2,
; CHECK: "enzyme_checkpoint"="fixedpoint" "enzyme_checkpoint_nregions"="3" "enzyme_fixed_point_nstates"="2"
