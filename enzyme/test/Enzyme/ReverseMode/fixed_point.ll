; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; __enzyme_fixed_point is lowered to a loop function of kind "fixedpoint",
; whose state @u gets a shadow. Its augmented pass runs the loop and keeps a
; snapshot at the converged state; its reverse pass iterates the combined
; derivative of the step from that snapshot until the adjoint converges.

@p = global double 2.000000e-01, align 8
@u = global [4 x double] zeroinitializer, align 16
@enzyme_fp_state = external global i32, align 4
@enzyme_fp_reduction = external global i32, align 4
@enzyme_const = external global i32, align 4

; u[k] = 0.5 * u[k] * u[(k+1)%4] + p, while some entry changes by more than tol.
define i32 @step(i64 %i, ptr %tol) {
entry:
  %t = load double, ptr %tol, align 8
  br label %for.body

for.body:
  %k = phi i64 [ 0, %entry ], [ %k.next, %for.body ]
  %err = phi double [ 0.000000e+00, %entry ], [ %err.next, %for.body ]
  %k.next = add nuw nsw i64 %k, 1
  %k1 = and i64 %k.next, 3
  %a.p = getelementptr inbounds double, ptr @u, i64 %k
  %b.p = getelementptr inbounds double, ptr @u, i64 %k1
  %a = load double, ptr %a.p, align 8
  %b = load double, ptr %b.p, align 8
  %pp = load double, ptr @p, align 8
  %ab = fmul double %a, %b
  %h = fmul double %ab, 5.000000e-01
  %new = fadd double %h, %pp
  store double %new, ptr %a.p, align 8
  %d = fsub double %new, %a
  %ad = call double @llvm.fabs.f64(double %d)
  %err.next = call double @llvm.maxnum.f64(double %err, double %ad)
  %cmp = icmp ult i64 %k.next, 4
  br i1 %cmp, label %for.body, label %for.end

for.end:
  %go = fcmp ogt double %err.next, %t
  %r = zext i1 %go to i32
  ret i32 %r
}

declare double @llvm.fabs.f64(double)
declare double @llvm.maxnum.f64(double, double)

define double @f(ptr %tol) {
entry:
  %0 = load i32, ptr @enzyme_fp_state, align 4
  %1 = load i32, ptr @enzyme_fp_reduction, align 4
  call void (ptr, ...) @__enzyme_fixed_point(ptr @step, i32 %0, ptr @u, i64 32, i32 %1, double 1.000000e-20, ptr %tol)
  %2 = load double, ptr @u, align 8
  ret double %2
}

declare void @__enzyme_fixed_point(ptr, ...)

define void @df(ptr %tol) {
entry:
  %0 = load i32, ptr @enzyme_const, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @f, i32 %0, ptr %tol)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; CHECK: define double @f(
; CHECK:   call void @enzyme.ckpt.fixedpoint.step(i64 0, i64 1000, ptr null, double 0x3BC79CA10C924223, ptr @u, i64 32, ptr %tol)

; The state pointer is the one argument of the schedule that is not inactive.
; CHECK: define internal void @enzyme.ckpt.fixedpoint.step(i64 "enzyme_inactive" %0, i64 "enzyme_inactive" %1, ptr "enzyme_inactive" %2, double "enzyme_inactive" %3, ptr %4, i64 "enzyme_inactive" %5, ptr %6) #[[LOOPATTR:[0-9]+]] !enzyme_checkpoint_step
; CHECK: body:
; CHECK-NEXT:   %i = phi i64 [ %0, %entry ], [ %i.next, %body ]
; CHECK-NEXT:   %7 = call i32 @step(i64 %i, ptr %6)
; CHECK-NEXT:   %i.next = add i64 %i, 1
; CHECK-NEXT:   %8 = icmp ne i32 %7, 0
; CHECK-NEXT:   br i1 %8, label %body, label %exit

; CHECK: define internal void @diffef(
; CHECK:   call void @diffeenzyme.ckpt.fixedpoint.step(i64 0, i64 1000, ptr null, double 0x3BC79CA10C924223, ptr @u, ptr @u_shadow, i64 32, ptr %tol)

; CHECK: define internal void @diffestep(i64 %i, ptr {{.*}}%tol)

; CHECK: define internal ptr @augmented_enzyme.ckpt.fixedpoint.step(i64 %0, i64 %1, ptr %2, double %3, ptr %4, ptr %5, i64 %6, ptr %7)
; CHECK:   %handle = call ptr @__enzyme_fp_fwd(ptr %env, ptr @enzyme.ckpt.primal_while.enzyme.ckpt.fixedpoint.step.c, ptr %regions, i64 2, i64 %{{.*}})
; CHECK-NEXT:   ret ptr %handle

; The forward pass runs the loop to the end, then snapshots the regions.
; CHECK: define internal ptr @__enzyme_fp_fwd(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4)
; CHECK:   %go = call i32 %1(ptr %0, i64 %i)
; CHECK:   %h = call ptr @malloc(
; CHECK:   call void @__enzyme_fp_copy(ptr %2, i64 %3, ptr %{{.*}}, i1 true)

; The reverse pass gets the shadow of the state for its convergence test.
; CHECK: define internal void @diffeenzyme.ckpt.fixedpoint.step(i64 %0, i64 %1, ptr %2, double %3, ptr %4, ptr %5, i64 %6, ptr %7)
; CHECK:   %handle = call ptr @augmented_enzyme.ckpt.fixedpoint.step(i64 %0, i64 %1, ptr %2, double %3, ptr %4, ptr %5, i64 %6, ptr %7)
; CHECK:   %states = alloca [1 x { ptr, i64, i32, i32 }]
; CHECK:   store ptr %5, ptr
; CHECK:   store i64 %6, ptr
; CHECK:   call void @__enzyme_fp_rev(ptr %handle, ptr %regions, i64 2, i64 %{{.*}}, ptr %env, ptr @enzyme.ckpt.turn.enzyme.ckpt.fixedpoint.step.c, ptr %states, i64 1, double %3, i64 %1, ptr %2)
; CHECK-NEXT:   ret void

; Each adjoint iteration is the step's combined derivative.
; CHECK: define internal void @enzyme.ckpt.turn.enzyme.ckpt.fixedpoint.step.c(ptr %0, i64 %1)
; CHECK:   call void @diffestep(i64 %1, ptr %{{.*}})

; It restores the snapshot, runs the derivative, and measures the update.
; CHECK: define internal void @__enzyme_fp_rev(
; CHECK:   call void @__enzyme_fp_copy(ptr %1, i64 %2, ptr %entry_state, i1 true)
; CHECK: loop:
; CHECK:   call void @__enzyme_fp_copy(ptr %1, i64 %2, ptr %snapshot, i1 false)
; CHECK:   call void %5(ptr %4, i64 %last)
; CHECK:   %sqnorm = call double @__enzyme_fp_sqnorm(ptr %6, i64 %7, ptr null)
; CHECK: done:
; CHECK:   call void @__enzyme_fp_copy(ptr %1, i64 %2, ptr %entry_state, i1 false)

; CHECK: attributes #[[LOOPATTR]] = { noinline "enzyme_checkpoint"="fixedpoint" "enzyme_checkpoint_nregions"="1" "enzyme_fixed_point_nstates"="1" }
