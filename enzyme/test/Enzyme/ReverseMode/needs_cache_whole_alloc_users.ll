; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli -

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare double @__enzyme_autodiff(ptr, ...)
declare void @abort()

define i32 @dummy_personality(...) {
entry:
  ret i32 0
}

; Non-nocapture scalar helper reading from an interior pointer.
define double @read_second(ptr %p) noinline readonly nofree {
entry:
  %v = load double, ptr %p, align 8
  ret double %v
}

; Nocapture pointer-returning helper invoked via InvokeInst.
define ptr @pick_ext(ptr nocapture readonly %tmp, ptr %ext) noinline readonly nofree {
entry:
  %v = load double, ptr %tmp, align 8
  %pos = fcmp ogt double %v, 0.000000e+00
  %gep = getelementptr inbounds double, ptr %ext, i64 1
  %res = select i1 %pos, ptr %gep, ptr %ext
  ret ptr %res
}

; Helper whose first pointer argument has zero uses inside the body.
define ptr @ignore_first(ptr %unused, ptr %ext) noinline readonly nofree {
entry:
  %gep = getelementptr inbounds double, ptr %ext, i64 1
  ret ptr %gep
}

; Helper returning an interior pointer alias of its argument.
define ptr @alias_subptr(ptr %p) noinline readonly nofree {
entry:
  ret ptr %p
}

define double @active_dot_ptr(ptr %p, double %x) noinline {
entry:
  %v = load double, ptr %p, align 8
  %r = fmul double %v, %x
  ret double %r
}

; 1) Cached scalar call result on an interior GEP does not require keeping %alloc alive.
define double @non_nocapture_scalar(double %x, ptr %szp) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %nsum, %loop ]
  %sz = load i64, ptr %szp, align 8
  store i64 2, ptr %szp, align 8
  %bytes = shl i64 %sz, 3
  %alloc = call noalias ptr @malloc(i64 %bytes)
  store double %x, ptr %alloc, align 8
  %gep = getelementptr inbounds double, ptr %alloc, i64 1
  store double 3.000000e+00, ptr %gep, align 8
  %val = call double @read_second(ptr %gep), !enzyme_inactive !{}
  call void @free(ptr %alloc)
  %prod = fmul double %x, %val
  %nsum = fadd double %sum, %prod
  %next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %next, 2
  br i1 %cond, label %exit, label %loop

exit:
  ret double %nsum
}

; 2) InvokeInst with nocapture arg should not assert or force whole-allocation caching.
define double @invoke_nocapture_ptr(double %x, ptr %szp, ptr %ext) personality ptr @dummy_personality {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %cont ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %nsum, %cont ]
  %sz = load i64, ptr %szp, align 8
  store i64 2, ptr %szp, align 8
  %bytes = shl i64 %sz, 3
  %alloc = call noalias ptr @malloc(i64 %bytes)
  store double %x, ptr %alloc, align 8
  %gep = getelementptr inbounds double, ptr %alloc, i64 1
  store double 1.000000e+00, ptr %gep, align 8
  %chosen = invoke ptr @pick_ext(ptr nocapture %gep, ptr %ext)
          to label %cont unwind label %lpad, !enzyme_inactive !{}

cont:
  call void @free(ptr %alloc)
  %term = call double @active_dot_ptr(ptr %chosen, double %x)
  %nsum = fadd double %sum, %term
  %next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %next, 2
  br i1 %cond, label %exit, label %loop

lpad:
  %lp = landingpad { ptr, i32 }
          cleanup
  call void @abort()
  unreachable

exit:
  ret double %nsum
}

; 3) Unused pointer parameter in helper does not retain the allocation.
define double @dead_arg_helper(double %x, ptr %szp, ptr %ext) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %nsum, %loop ]
  %sz = load i64, ptr %szp, align 8
  store i64 2, ptr %szp, align 8
  %bytes = shl i64 %sz, 3
  %alloc = call noalias ptr @malloc(i64 %bytes)
  store double %x, ptr %alloc, align 8
  %gep = getelementptr inbounds double, ptr %alloc, i64 1
  store double 5.000000e+00, ptr %gep, align 8
  %chosen = call ptr @ignore_first(ptr %gep, ptr %ext), !enzyme_inactive !{}
  call void @free(ptr %alloc)
  %term = call double @active_dot_ptr(ptr %chosen, double %x)
  %nsum = fadd double %sum, %term
  %next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %next, 2
  br i1 %cond, label %exit, label %loop

exit:
  ret double %nsum
}

; 4) Split-mode subfunction caching an interior pointer alias keeps %alloc alive until reverse.
define double @sub_aliasing_ptr(double %x, ptr %szp) noinline {
entry:
  %sz = load i64, ptr %szp, align 8
  store i64 2, ptr %szp, align 8
  %bytes = shl i64 %sz, 3
  %alloc = call noalias ptr @malloc(i64 %bytes)
  store double %x, ptr %alloc, align 8
  %gep = getelementptr inbounds double, ptr %alloc, i64 1
  %x5 = fmul double %x, 5.000000e+00
  store double %x5, ptr %gep, align 8
  %subp = call ptr @alias_subptr(ptr %gep), !enzyme_inactive !{}
  %res = call double @active_dot_ptr(ptr %subp, double %x)
  call void @free(ptr %alloc)
  ret double %res
}

define double @caller_split(double %x, ptr %szp) {
entry:
  %r1 = call double @sub_aliasing_ptr(double %x, ptr %szp)
  %r2 = call double @sub_aliasing_ptr(double %x, ptr %szp)
  %sum = fadd double %r1, %r2
  ret double %sum
}

define void @check_eq(double %got, double %expected) {
entry:
  %diff = fsub double %got, %expected
  %neg = fsub double 0.000000e+00, %diff
  %is_neg = fcmp olt double %diff, 0.000000e+00
  %abs = select i1 %is_neg, double %neg, double %diff
  %bad = fcmp ogt double %abs, 1.000000e-09
  br i1 %bad, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret void
}

define i32 @main() {
entry:
  %sz = alloca i64, align 8
  %ext = alloca [2 x double], align 8
  %dext = alloca [2 x double], align 8
  %ext1 = getelementptr inbounds double, ptr %ext, i64 1
  %dext1 = getelementptr inbounds double, ptr %dext, i64 1

  ; Test 1: f(x) = 2 * (3 * x) = 6 * x -> df/dx = 6.0
  store i64 2, ptr %sz, align 8
  %d1 = call double (ptr, ...) @__enzyme_autodiff(ptr @non_nocapture_scalar, double 4.000000e+00, metadata !"enzyme_const", ptr %sz)
  call void @check_eq(double %d1, double 6.000000e+00)

  ; Test 2: f(x) = 2 * (ext[1] * x) with ext[1] = 7.0 -> df/dx = 14.0, dext[1] = 2 * x = 8.0
  store i64 2, ptr %sz, align 8
  store double 2.000000e+00, ptr %ext, align 8
  store double 7.000000e+00, ptr %ext1, align 8
  store double 0.000000e+00, ptr %dext, align 8
  store double 0.000000e+00, ptr %dext1, align 8
  %d2 = call double (ptr, ...) @__enzyme_autodiff(ptr @invoke_nocapture_ptr, double 4.000000e+00, metadata !"enzyme_const", ptr %sz, metadata !"enzyme_dup", ptr %ext, ptr %dext)
  call void @check_eq(double %d2, double 14.000000e+00)
  %de2 = load double, ptr %dext1, align 8
  call void @check_eq(double %de2, double 8.000000e+00)

  ; Test 3: f(x) = 2 * (ext[1] * x) with ext[1] = 9.0 -> df/dx = 18.0
  store i64 2, ptr %sz, align 8
  store double 2.000000e+00, ptr %ext, align 8
  store double 9.000000e+00, ptr %ext1, align 8
  %d3 = call double (ptr, ...) @__enzyme_autodiff(ptr @dead_arg_helper, double 4.000000e+00, metadata !"enzyme_const", ptr %sz, metadata !"enzyme_const", ptr %ext)
  call void @check_eq(double %d3, double 18.000000e+00)

  ; Test 4: caller_split calls sub_aliasing_ptr twice.
  ; Note: %subp = @alias_subptr(%gep) is marked !enzyme_inactive so %subp's shadow is null,
  ; while @active_dot_ptr(%subp, %x) reads primal *%subp (= 5 * x) in reverse!
  ; Each subcall contributes d/dx (const(5*x) * x) = 5*x = 20.0 -> total 40.0!
  store i64 2, ptr %sz, align 8
  %d4 = call double (ptr, ...) @__enzyme_autodiff(ptr @caller_split, double 4.000000e+00, metadata !"enzyme_const", ptr %sz)
  call void @check_eq(double %d4, double 40.000000e+00)

  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffenon_nocapture_scalar(
; CHECK: loop:
; CHECK:   %[[ALLOC1:.+]] = call noalias ptr @malloc(
; CHECK:   %[[VAL1:.+]] = call double @read_second(
; CHECK:   call void @free(ptr %[[ALLOC1]])

; CHECK-LABEL: define internal { double } @diffeinvoke_nocapture_ptr(
; CHECK: loop:
; CHECK:   %[[ALLOC2:.+]] = call noalias ptr @malloc(
; CHECK:   call void @free(ptr %[[ALLOC2]])

; CHECK-LABEL: define internal { double } @diffedead_arg_helper(
; CHECK: loop:
; CHECK:   %[[ALLOC3:.+]] = call noalias ptr @malloc(
; CHECK:   %[[CHOSEN3:.+]] = call ptr @ignore_first(
; CHECK:   call void @free(ptr %[[ALLOC3]])

; CHECK-LABEL: define internal {{.*}} @augmented_sub_aliasing_ptr(
; CHECK:   %[[ALLOC4:.+]] = call noalias ptr @malloc(
; CHECK-NOT: call void @free(ptr %[[ALLOC4]])
; CHECK: ret

; CHECK-LABEL: define internal { double } @diffesub_aliasing_ptr{{.*}}(
; CHECK: invertentry:
; CHECK:   call void @free(ptr {{.*}})
