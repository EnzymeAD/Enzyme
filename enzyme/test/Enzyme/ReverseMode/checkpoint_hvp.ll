; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; Forward mode over the reverse mode of a checkpointed loop: the tangents of
; its passes call the same drivers, with trampolines that run the tangent of
; the step and of its reverse derivative, an env holding each argument, its
; tangent, its adjoint and the adjoint's tangent, and the tangent of the
; marked region among the regions.

@enzyme_scheme = external global i32, align 4
@enzyme_checkpoint_region = external global i32, align 4
@enzyme_const = external global i32, align 4
@enzyme_dup = external global i32, align 4

define void @step(i64 %i, ptr %x) {
entry:
  %0 = load double, ptr %x, align 8
  %1 = call double @llvm.sin.f64(double %0)
  %2 = fmul double %0, %1
  store double %2, ptr %x, align 8
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

; CHECK: define internal void @fwddiffediffef(ptr %x, ptr %"x'", ptr %"x'1", ptr %"x''", i64 %n, ptr %s, ptr %c, double %differeturn)
; CHECK:   %0 = call ptr @fwddiffeaugmented_enzyme.ckpt.for.step(i64 0, i64 %n, ptr %s, ptr %c, ptr %x, ptr %"x'", i64 8, ptr %x, ptr %"x'", ptr %"x'1", ptr %"x''")
; CHECK:   call void @fwddiffediffe_rev_enzyme.ckpt.for.step(i64 0, i64 %n, ptr %s, ptr %c, ptr %x, ptr %"x'", i64 8, ptr %x, ptr %"x'", ptr %"x'1", ptr %"x''", ptr %0)

; CHECK: define internal ptr @fwddiffeaugmented_enzyme.ckpt.for.step(i64 %0, i64 %1, ptr %2, ptr %3, ptr %4, ptr %5, i64 %6, ptr %7, ptr %8, ptr %9, ptr %10)
; CHECK:   %env = alloca { ptr, ptr, ptr, ptr }
; CHECK:   store ptr %7, ptr
; CHECK:   store ptr %8, ptr
; CHECK:   store ptr %9, ptr
; CHECK:   store ptr %10, ptr
; CHECK:   %regions = alloca [2 x { ptr, i64, i32, i32 }]
; CHECK:   store ptr %4, ptr
; CHECK:   store ptr %5, ptr
; CHECK:   %handle = call ptr @__enzyme_ckpt_fwd(ptr %2, ptr %3, i64 %0, i64 %1, ptr %regions, i64 2, i64 %{{.*}}, ptr %env, ptr @enzyme.ckpt.tangent.fwddiffestep, ptr @enzyme.ckpt.paths.step, i64 3, ptr null)
; CHECK-NEXT:   ret ptr %handle

; CHECK: define internal void @fwddiffestep(i64 %i, ptr {{.*}}%x, ptr {{.*}}%"x'")

; CHECK: define internal void @enzyme.ckpt.tangent.fwddiffestep(ptr %0, i64 %1)
; CHECK:   call void @fwddiffestep(i64 %1, ptr %{{.*}}, ptr %{{.*}})

; CHECK: define internal void @fwddiffediffe_rev_enzyme.ckpt.for.step(i64 %0, i64 %1, ptr %2, ptr %3, ptr %4, ptr %5, i64 %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11)
; CHECK:   %env = alloca { ptr, ptr, ptr, ptr }
; CHECK:   call void @__enzyme_ckpt_rev(ptr %11, ptr %regions, i64 2, ptr %env, ptr @enzyme.ckpt.tangent.fwddiffestep.{{[0-9]+}}, ptr @enzyme.ckpt.tangent_turn.fwddiffediffestep)
; CHECK-NEXT:   ret void

; CHECK: define internal void @fwddiffediffestep(i64 %i, ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%"x'1", ptr {{.*}}%"x''")

; CHECK: define internal void @enzyme.ckpt.tangent_turn.fwddiffediffestep(ptr %0, i64 %1)
; CHECK:   call void @fwddiffediffestep(i64 %1, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
