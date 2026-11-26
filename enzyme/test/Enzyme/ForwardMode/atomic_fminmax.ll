; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli -

declare double @__enzyme_fwddiff(ptr, ...)
declare { ptr, double } @__enzyme_augmentfwd(ptr, ...)
declare double @__enzyme_fwdsplit(ptr, ...)
declare double @llvm.fabs.f64(double)
declare void @abort()

define double @fmax_kernel(ptr %acc, double %x) noinline {
entry:
  %old = atomicrmw fmax ptr %acc, double %x monotonic, align 8
  ret double %old
}

define double @fmin_kernel(ptr %acc, double %x) noinline {
entry:
  %old = atomicrmw fmin ptr %acc, double %x acq_rel, align 8
  ret double %old
}

define double @fmax_loop(ptr %arr, i64 %n) noinline {
entry:
  %acc = alloca double, align 8
  store double -1.000000e+02, ptr %acc, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %gep = getelementptr inbounds double, ptr %arr, i64 %i
  %v = load double, ptr %gep, align 8
  %old = atomicrmw fmax ptr %acc, double %v monotonic, align 8
  %next = add nuw nsw i64 %i, 1
  %cond = icmp ult i64 %next, %n
  br i1 %cond, label %loop, label %exit

exit:
  %res = load double, ptr %acc, align 8
  ret double %res
}

define void @check_eq(double %actual, double %expected) {
entry:
  %diff = fsub double %actual, %expected
  %abs = call double @llvm.fabs.f64(double %diff)
  %ok = fcmp olt double %abs, 1.000000e-09
  br i1 %ok, label %pass, label %fail

fail:
  call void @abort()
  unreachable

pass:
  ret void
}

define i32 @main() {
entry:
  %acc = alloca double, align 8
  %dacc = alloca double, align 8

  ; Case 1: fmax where candidate wins (2.0 -> 5.0)
  store double 2.000000e+00, ptr %acc, align 8
  store double 5.000000e-01, ptr %dacc, align 8
  %dret1 = call double (ptr, ...) @__enzyme_fwddiff(ptr @fmax_kernel, metadata !"enzyme_dup", ptr %acc, ptr %dacc, metadata !"enzyme_dup", double 5.000000e+00, double 3.000000e+00)
  %new1 = load double, ptr %acc, align 8
  %dnew1 = load double, ptr %dacc, align 8
  call void @check_eq(double %new1, double 5.000000e+00)
  call void @check_eq(double %dret1, double 5.000000e-01)
  call void @check_eq(double %dnew1, double 3.000000e+00)

  ; Case 2: fmax where existing value stays (5.0 vs 2.0)
  store double 5.000000e+00, ptr %acc, align 8
  store double 5.000000e-01, ptr %dacc, align 8
  %dret2 = call double (ptr, ...) @__enzyme_fwddiff(ptr @fmax_kernel, metadata !"enzyme_dup", ptr %acc, ptr %dacc, metadata !"enzyme_dup", double 2.000000e+00, double 3.000000e+00)
  %new2 = load double, ptr %acc, align 8
  %dnew2 = load double, ptr %dacc, align 8
  call void @check_eq(double %new2, double 5.000000e+00)
  call void @check_eq(double %dret2, double 5.000000e-01)
  call void @check_eq(double %dnew2, double 5.000000e-01)

  ; Case 3: fmax equal-value tie (4.0 vs 4.0 -> unchanged, old derivative stays)
  store double 4.000000e+00, ptr %acc, align 8
  store double 5.000000e-01, ptr %dacc, align 8
  %dret3 = call double (ptr, ...) @__enzyme_fwddiff(ptr @fmax_kernel, metadata !"enzyme_dup", ptr %acc, ptr %dacc, metadata !"enzyme_dup", double 4.000000e+00, double 3.000000e+00)
  %dnew3 = load double, ptr %dacc, align 8
  call void @check_eq(double %dret3, double 5.000000e-01)
  call void @check_eq(double %dnew3, double 5.000000e-01)

  ; Case 4: fmin with acq_rel ordering (candidate wins: 5.0 -> 2.0, and tie: 2.0 vs 2.0)
  store double 5.000000e+00, ptr %acc, align 8
  store double 7.500000e-01, ptr %dacc, align 8
  %dret4 = call double (ptr, ...) @__enzyme_fwddiff(ptr @fmin_kernel, metadata !"enzyme_dup", ptr %acc, ptr %dacc, metadata !"enzyme_dup", double 2.000000e+00, double 4.000000e+00)
  %dnew4 = load double, ptr %dacc, align 8
  call void @check_eq(double %dret4, double 7.500000e-01)
  call void @check_eq(double %dnew4, double 4.000000e+00)

  store double 2.000000e+00, ptr %acc, align 8
  store double 7.500000e-01, ptr %dacc, align 8
  %dret5 = call double (ptr, ...) @__enzyme_fwddiff(ptr @fmin_kernel, metadata !"enzyme_dup", ptr %acc, ptr %dacc, metadata !"enzyme_dup", double 2.000000e+00, double 9.000000e+00)
  %dnew5 = load double, ptr %dacc, align 8
  call void @check_eq(double %dret5, double 7.500000e-01)
  call void @check_eq(double %dnew5, double 7.500000e-01)

  ; Case 5: loop reduction with ties in combined and split forward mode
  %arr = alloca [4 x double], align 8
  %darr = alloca [4 x double], align 8
  %p0 = getelementptr inbounds [4 x double], ptr %arr, i64 0, i64 0
  %p1 = getelementptr inbounds [4 x double], ptr %arr, i64 0, i64 1
  %p2 = getelementptr inbounds [4 x double], ptr %arr, i64 0, i64 2
  %p3 = getelementptr inbounds [4 x double], ptr %arr, i64 0, i64 3
  store double 3.000000e+00, ptr %p0, align 8
  store double 7.000000e+00, ptr %p1, align 8
  store double 7.000000e+00, ptr %p2, align 8
  store double 5.000000e+00, ptr %p3, align 8

  %dp0 = getelementptr inbounds [4 x double], ptr %darr, i64 0, i64 0
  %dp1 = getelementptr inbounds [4 x double], ptr %darr, i64 0, i64 1
  %dp2 = getelementptr inbounds [4 x double], ptr %darr, i64 0, i64 2
  %dp3 = getelementptr inbounds [4 x double], ptr %darr, i64 0, i64 3
  store double 1.100000e+01, ptr %dp0, align 8
  store double 2.200000e+01, ptr %dp1, align 8
  store double 3.300000e+01, ptr %dp2, align 8
  store double 4.400000e+01, ptr %dp3, align 8

  %dloop = call double (ptr, ...) @__enzyme_fwddiff(ptr @fmax_loop, metadata !"enzyme_dup", ptr %arr, ptr %darr, metadata !"enzyme_const", i64 4)
  call void @check_eq(double %dloop, double 2.200000e+01)

  %aug = call { ptr, double } (ptr, ...) @__enzyme_augmentfwd(ptr @fmax_loop, metadata !"enzyme_allocated", i64 -1, metadata !"enzyme_dup", ptr %arr, ptr %darr, metadata !"enzyme_const", i64 4)
  %tape = extractvalue { ptr, double } %aug, 0
  %aug_ret = extractvalue { ptr, double } %aug, 1
  call void @check_eq(double %aug_ret, double 7.000000e+00)
  %dsplit = call double (ptr, ...) @__enzyme_fwdsplit(ptr @fmax_loop, metadata !"enzyme_nofree", metadata !"enzyme_allocated", i64 -1, metadata !"enzyme_dup", ptr %arr, ptr %darr, metadata !"enzyme_const", i64 4, ptr %tape)
  call void @check_eq(double %dsplit, double 2.200000e+01)

  ret i32 0
}

; CHECK-LABEL: define internal double @fwddiffefmax_kernel(
; CHECK:   %[[INIT:.+]] = load atomic i64, ptr %acc monotonic, align 8
; CHECK: loop.i:
; CHECK:   %cur.int.i = phi i64 [ %[[INIT]], %entry ], [ %next.int.i, %cas.i ]
; CHECK:   %cur.i = bitcast i64 %cur.int.i to double
; CHECK:   fcmp ogt double %x, %cur.i
; CHECK: cas.i:
; CHECK:   cmpxchg ptr %acc, i64 %cur.int.i, i64 %{{.*}} monotonic monotonic, align 8
; CHECK: __enzyme_atomic_fmax_double_a8_o2.exit:
; CHECK:   %changed.i = phi i1 [ false, %loop.i ], [ true, %cas.i ]
; CHECK:   %[[DOLD:.+]] = load atomic double, ptr %"acc'" monotonic, align 8
; CHECK:   %[[DNEW:.+]] = select{{.*}} i1 %changed{{.*}}, double %"x'", double %[[DOLD]]
; CHECK:   store atomic double %[[DNEW]], ptr %"acc'" monotonic, align 8
; CHECK:   ret double %[[DOLD]]

; CHECK-LABEL: define internal { double, i1 } @__enzyme_atomic_fmax_double_a8_o2(
; CHECK: entry:
; CHECK:   %[[HINIT:.+]] = load atomic i64, ptr %ptr monotonic, align 8
; CHECK: loop:
; CHECK:   %cur.int = phi i64 [ %[[HINIT]], %entry ], [ %next.int, %cas ]
; CHECK:   %cur = bitcast i64 %cur.int to double
; CHECK:   fcmp ogt double %val, %cur
; CHECK: cas:
; CHECK:   cmpxchg ptr %ptr, i64 %cur.int, i64 %{{.*}} monotonic monotonic, align 8
; CHECK: exit:
; CHECK:   %changed = phi i1 [ false, %loop ], [ true, %cas ]

