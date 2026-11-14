; RUN: if [ %llvmver -ge 16 ]; then not %opt < %s %newLoadEnzyme -passes="enzyme" -S 2>&1 | FileCheck %s; fi

; Forward mode over the reverse mode of a checkpointed loop whose step keeps
; differentiable state in a global: the global's one shadow would hold both
; the tangent and the adjoint of the state, so Enzyme rejects it.

@state = internal global double 1.000000e+00, align 8
@enzyme_scheme = external global i32, align 4
@enzyme_checkpoint_region = external global i32, align 4
@enzyme_const = external global i32, align 4
@enzyme_dup = external global i32, align 4

define void @step(i64 %i, ptr %x) {
entry:
  %0 = load double, ptr %x, align 8
  %1 = call double @llvm.sin.f64(double %0)
  %g = load double, ptr @state, align 8
  %2 = fmul double %0, %1
  %3 = fmul double %2, %g
  store double %3, ptr %x, align 8
  store double %2, ptr @state, align 8
  ret void
}

declare double @llvm.sin.f64(double)

define double @f(ptr %x, i64 %n, ptr %s, ptr %c) {
entry:
  %0 = load i32, ptr @enzyme_scheme, align 4
  %1 = load i32, ptr @enzyme_checkpoint_region, align 4
  call void (ptr, i64, i64, ...) @__enzyme_checkpoint_for(ptr @step, i64 0, i64 %n, i32 %0, ptr %s, ptr %c, i32 %1, ptr %x, i64 8, ptr %x)
  %2 = load double, ptr %x, align 8
  %3 = fmul double %2, %2
  ret double %3
}

declare void @__enzyme_checkpoint_for(ptr, i64, i64, ...)

define void @df(ptr %x, ptr %dx, i64 %n, ptr %s, ptr %c) {
entry:
  %0 = load i32, ptr @enzyme_const, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, i32 %0, i64 %n, i32 %0, ptr %s, i32 %0, ptr %c)
  ret void
}

define void @hvp(ptr %x, ptr %vx, ptr %dx, ptr %hv, i64 %n, ptr %s, ptr %c) {
entry:
  %0 = load i32, ptr @enzyme_const, align 4
  %1 = load i32, ptr @enzyme_dup, align 4
  call void (ptr, ...) @__enzyme_fwddiff(ptr @df, i32 %1, ptr %x, ptr %vx, i32 %1, ptr %dx, ptr %hv, i32 %0, i64 %n, i32 %0, ptr %s, i32 %0, ptr %c)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)
declare void @__enzyme_fwddiff(ptr, ...)

; CHECK: Forward mode over the reverse mode of a checkpointed loop (augmented_enzyme.ckpt.for.step) is not supported when the step keeps differentiable state in a global (state)
