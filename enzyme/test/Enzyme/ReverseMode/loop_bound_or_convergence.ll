; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -enzyme-max-cache=1 -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-max-cache=1 -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-max-cache=1 -S | %lli - | FileCheck %s --check-prefix=EXEC
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EXEC

; A fixed-point iteration leaves its loop either on convergence or after a
; bounded number of steps. Once the two tests are combined into one branch
; condition, in binary or select form, the exact trip count is unknown but the
; iteration bound still caps it, so the cache of the loop-carried value is sized
; from that bound up front rather than grown by realloc each iteration.

declare double @__enzyme_autodiff(ptr, ...)
declare i32 @printf(ptr, ...)

@format = private constant [10 x i8] c"%.6f %.6f\00"
@newline = private constant [2 x i8] c"\0A\00"

define double @fixed_point_or(double %x, ptr %data) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %value = phi double [ %x, %entry ], [ %value.next, %loop ]
  %scale_ptr = getelementptr inbounds double, ptr %data, i64 %iv
  %scale = load double, ptr %scale_ptr, align 8
  %value.next = fmul double %value, %scale
  %iv.next = add nuw nsw i64 %iv, 1
  %converged = fcmp olt double %value.next, 1.000000e-03
  %at_limit = icmp eq i64 %iv.next, 100
  %done = or i1 %converged, %at_limit
  br i1 %done, label %exit, label %loop

exit:
  ret double %value.next
}

define double @fixed_point_select(double %x, ptr %data) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %value = phi double [ %x, %entry ], [ %value.next, %loop ]
  %scale_ptr = getelementptr inbounds double, ptr %data, i64 %iv
  %scale = load double, ptr %scale_ptr, align 8
  %value.next = fmul double %value, %scale
  %iv.next = add nuw nsw i64 %iv, 1
  %converged = fcmp olt double %value.next, 1.000000e-03
  %at_limit = icmp eq i64 %iv.next, 100
  %done = select i1 %converged, i1 true, i1 %at_limit
  br i1 %done, label %exit, label %loop

exit:
  ret double %value.next
}

define i32 @main() {
entry:
  %data = alloca [100 x double], align 8
  %shadow = alloca [100 x double], align 8
  br label %fill

fill:
  %index = phi i64 [ 0, %entry ], [ %index.next, %fill ]
  %slot = getelementptr inbounds double, ptr %data, i64 %index
  store double 5.000000e-01, ptr %slot, align 8
  %shadow_slot = getelementptr inbounds double, ptr %shadow, i64 %index
  store double 0.000000e+00, ptr %shadow_slot, align 8
  %index.next = add nuw nsw i64 %index, 1
  %filled = icmp eq i64 %index.next, 100
  br i1 %filled, label %run, label %fill

run:
  ; 0.5^10 = 1/1024 < 1e-3, so the loop converges after 10 steps.
  %derivative_or = call double (ptr, ...) @__enzyme_autodiff(ptr @fixed_point_or, double 1.0, ptr %data, ptr %shadow)
  %first_shadow_or = load double, ptr %shadow, align 8
  call i32 (ptr, ...) @printf(ptr @format, double %derivative_or, double %first_shadow_or)
  call i32 (ptr, ...) @printf(ptr @newline)
  store double 0.000000e+00, ptr %shadow, align 8
  %derivative_select = call double (ptr, ...) @__enzyme_autodiff(ptr @fixed_point_select, double 1.0, ptr %data, ptr %shadow)
  %first_shadow_select = load double, ptr %shadow, align 8
  call i32 (ptr, ...) @printf(ptr @format, double %derivative_select, double %first_shadow_select)
  call i32 (ptr, ...) @printf(ptr @newline)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffefixed_point_or(
; CHECK-NOT: @realloc
; CHECK: %[[CACHE1:.+]] = tail call noalias nonnull dereferenceable(800) dereferenceable_or_null(800) ptr @malloc(i64 800)
; CHECK-NOT: @realloc
; CHECK: ret { double }

; CHECK-LABEL: define internal { double } @diffefixed_point_select(
; CHECK-NOT: @realloc
; CHECK: %[[CACHE2:.+]] = tail call noalias nonnull dereferenceable(800) dereferenceable_or_null(800) ptr @malloc(i64 800)
; CHECK-NOT: @realloc
; CHECK: ret { double }

; d/dx of x * 0.5^10 is 0.5^10 = 0.000977, and d/d(data[0]) is x * 0.5^9 = 0.001953.
; EXEC: 0.000977 0.001953
; EXEC-NEXT: 0.000977 0.001953
