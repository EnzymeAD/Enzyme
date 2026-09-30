; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-print-checkpoint-regions -disable-output 2>&1 | FileCheck %s --check-prefix=REGIONS; fi

; __enzyme_checkpoint_for is lowered to a loop function; its reverse mode is
; the checkpointing driver running the augmented forward and reverse passes of
; one step at a time. A snapshot holds the marked region and @state, which the
; step writes, but not @param, which it only reads.


@param = global double 2.000000e+00, align 8
@cache = global double 0.000000e+00, align 8
@state = global [4 x double] zeroinitializer, align 16
@enzyme_scheme = external global i32, align 4
@enzyme_checkpoint_region = external global i32, align 4
@enzyme_const = external global i32, align 4

define void @step(i64 %i, ptr %x) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %k.010 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = zext i32 %k.010 to i64
  %arrayidx = getelementptr inbounds double, ptr @state, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8
  %1 = load double, ptr @param, align 8
  %idxprom1 = zext i32 %k.010 to i64
  %arrayidx2 = getelementptr inbounds double, ptr %x, i64 %idxprom1
  %2 = load double, ptr %arrayidx2, align 8
  %3 = call double @llvm.fmuladd.f64(double %0, double %1, double %2)
  %idxprom3 = zext i32 %k.010 to i64
  %arrayidx4 = getelementptr inbounds double, ptr @state, i64 %idxprom3
  store double %3, ptr %arrayidx4, align 8
  %idxprom5 = zext i32 %k.010 to i64
  %arrayidx6 = getelementptr inbounds double, ptr %x, i64 %idxprom5
  %4 = load double, ptr %arrayidx6, align 8
  %idxprom7 = zext i32 %k.010 to i64
  %arrayidx8 = getelementptr inbounds double, ptr @state, i64 %idxprom7
  %5 = load double, ptr %arrayidx8, align 8
  %mul = fmul double %4, %5
  %idxprom9 = zext i32 %k.010 to i64
  %arrayidx10 = getelementptr inbounds double, ptr %x, i64 %idxprom9
  store double %mul, ptr %arrayidx10, align 8
  %inc = add nsw i32 %k.010, 1
  %cmp = icmp ult i32 %inc, 4
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double)

define double @f(ptr %x, i64 %n, ptr %s, ptr %c) {
entry:
  %0 = load i32, ptr @enzyme_scheme, align 4
  %1 = load i32, ptr @enzyme_checkpoint_region, align 4
  call void (ptr, i64, i64, ...) @__enzyme_checkpoint_for(ptr @step, i64 0, i64 %n, i32 %0, ptr %s, ptr %c, i32 %1, ptr %x, i64 32, ptr %x)
  %2 = load double, ptr %x, align 8
  %3 = load double, ptr getelementptr inbounds (i8, ptr @state, i64 8), align 8
  %add = fadd double %2, %3
  ret double %add
}

declare void @__enzyme_checkpoint_for(ptr, i64, i64, ...)

define void @df(ptr %x, ptr %dx, i64 %n, ptr %s, ptr %c) {
entry:
  %0 = load i32, ptr @enzyme_const, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, i64 %n, i32 %0, ptr %s, i32 %0, ptr %c)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)


; REGIONS: checkpoint regions of step:
; REGIONS-NEXT:   marked region 0
; REGIONS-NEXT:   global state (32 bytes)
; REGIONS-NOT: param

; CHECK: @enzyme.ckpt.regions.step = private constant [1 x { ptr, i64, i32, i32 }] [{ ptr, i64, i32, i32 } { ptr @state, i64 32, i32 0, i32 0 }]

; CHECK: define double @f(
; CHECK:   call void @enzyme.ckpt.for.step(i64 0, i64 %n, ptr %s, ptr %c, ptr %x, i64 32, ptr %x)

; CHECK: define internal void @enzyme.ckpt.for.step(i64 "enzyme_inactive" %0, i64 "enzyme_inactive" %1, ptr "enzyme_inactive" %2, ptr "enzyme_inactive" %3, ptr "enzyme_inactive" %4, i64 "enzyme_inactive" %5, ptr %6) #[[LOOPATTR:[0-9]+]] !enzyme_checkpoint_step
; CHECK: body:
; CHECK-NEXT:   %i = phi i64 [ %0, %entry ], [ %i.next, %body ]
; CHECK-NEXT:   call void @step(i64 %i, ptr %6)

; CHECK: define internal void @diffef(
; CHECK:   call void @diffeenzyme.ckpt.for.step(i64 0, i64 %n, ptr %s, ptr %c, ptr %x, i64 32, ptr %x, ptr %"x'")

; CHECK: define internal ptr @augmented_enzyme.ckpt.for.step(i64 %0, i64 %1, ptr %2, ptr %3, ptr %4, i64 %5, ptr %6, ptr %7)
; CHECK:   %handle = call ptr @__enzyme_ckpt_fwd(ptr %2, ptr %3, i64 %0, i64 %1, ptr %regions, i64 2, i64 %{{.*}}, ptr %env, ptr @enzyme.ckpt.primal.enzyme.ckpt.for.step.d, ptr @enzyme.ckpt.aug.enzyme.ckpt.for.step.d)
; CHECK-NEXT:   ret ptr %handle

; CHECK: define internal void @enzyme.ckpt.primal.enzyme.ckpt.for.step.d(ptr %0, i64 %1)
; CHECK:   call void @step(i64 %1, ptr %{{.*}})

; CHECK: define internal ptr @enzyme.ckpt.aug.enzyme.ckpt.for.step.d(ptr %0, i64 %1)
; CHECK:   %[[TAPE:.+]] = call ptr @augmented_step(i64 %1, ptr %{{.*}}, ptr %{{.*}})
; CHECK-NEXT:   ret ptr %[[TAPE]]

; CHECK: define internal void @diffeenzyme.ckpt.for.step(i64 %0, i64 %1, ptr %2, ptr %3, ptr %4, i64 %5, ptr %6, ptr %7)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %handle = call ptr @augmented_enzyme.ckpt.for.step(i64 %0, i64 %1, ptr %2, ptr %3, ptr %4, i64 %5, ptr %6, ptr %7)
; The reverse pass takes the step's arguments from its own.
; CHECK:        %env = alloca { ptr, ptr }
; CHECK:        store ptr %6, ptr
; CHECK:        store ptr %7, ptr
; CHECK:   call void @__enzyme_ckpt_rev(ptr %handle, ptr %regions, i64 2, ptr %env, ptr @enzyme.ckpt.primal.enzyme.ckpt.for.step.d, ptr @enzyme.ckpt.aug.enzyme.ckpt.for.step.d, ptr @enzyme.ckpt.rev.enzyme.ckpt.for.step.d)
; CHECK-NEXT:   ret void

; CHECK: define internal void @enzyme.ckpt.rev.enzyme.ckpt.for.step.d(ptr %0, i64 %1, ptr %2)
; CHECK:   call void @diffestep(i64 %1, ptr %{{.*}}, ptr %{{.*}}, ptr %2)

; CHECK: attributes #[[LOOPATTR]] = { noinline "enzyme_checkpoint"="for" "enzyme_checkpoint_nregions"="1" }
