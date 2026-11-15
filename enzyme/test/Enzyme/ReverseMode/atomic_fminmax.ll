; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli -

declare { double, double } @__enzyme_autodiff_pair(ptr, ...)
declare double @__enzyme_autodiff_single(ptr, ...)
declare void @__enzyme_autodiff_void(ptr, ...)
declare { ptr, double } @__enzyme_augmentfwd(ptr, ...)
declare void @__enzyme_reverse(ptr, ...)
declare double @llvm.fabs.f64(double)
declare void @abort()

define double @fmax_two(double %x, double %y) noinline {
entry:
  %acc = alloca double, align 8
  store double -1.000000e+01, ptr %acc, align 8
  %old0 = atomicrmw fmax ptr %acc, double %x monotonic, align 8
  %old1 = atomicrmw fmax ptr %acc, double %y monotonic, align 8
  %res = load double, ptr %acc, align 8
  ret double %res
}

define double @fmin_with_ret(ptr %acc, double %x) noinline {
entry:
  %old = atomicrmw fmin ptr %acc, double %x seq_cst, align 8
  %cur = load double, ptr %acc, align 8
  %p1 = fmul double 2.000000e+00, %old
  %p2 = fmul double 3.000000e+00, %cur
  %sum = fadd double %p1, %p2
  ret double %sum
}

define void @fmax_relu_inplace(ptr %acc) noinline {
entry:
  %old = atomicrmw fmax ptr %acc, double 0.000000e+00 monotonic, align 8
  ret void
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
  ; Case 1: fmax_two when second wins (3.0, 7.0) -> (0.0, 1.0)
  %r1 = call { double, double } (ptr, ...) @__enzyme_autodiff_pair(ptr @fmax_two, double 3.000000e+00, double 7.000000e+00)
  %r1_dx = extractvalue { double, double } %r1, 0
  %r1_dy = extractvalue { double, double } %r1, 1
  call void @check_eq(double %r1_dx, double 0.000000e+00)
  call void @check_eq(double %r1_dy, double 1.000000e+00)

  ; Case 2: fmax_two when first wins (7.0, 3.0) -> (1.0, 0.0)
  %r2 = call { double, double } (ptr, ...) @__enzyme_autodiff_pair(ptr @fmax_two, double 7.000000e+00, double 3.000000e+00)
  %r2_dx = extractvalue { double, double } %r2, 0
  %r2_dy = extractvalue { double, double } %r2, 1
  call void @check_eq(double %r2_dx, double 1.000000e+00)
  call void @check_eq(double %r2_dy, double 0.000000e+00)

  ; Case 3: fmax_two equal-element tie (4.0, 4.0) -> first stored, second did not -> (1.0, 0.0)
  %r3 = call { double, double } (ptr, ...) @__enzyme_autodiff_pair(ptr @fmax_two, double 4.000000e+00, double 4.000000e+00)
  %r3_dx = extractvalue { double, double } %r3, 0
  %r3_dy = extractvalue { double, double } %r3, 1
  call void @check_eq(double %r3_dx, double 1.000000e+00)
  call void @check_eq(double %r3_dy, double 0.000000e+00)

  ; Case 4: fmin_with_ret (active return + active pointer)
  %acc = alloca double, align 8
  %dacc = alloca double, align 8

  ; 4a: candidate wins (*acc = 5.0, x = 2.0): out = 2*old + 3*x -> dacc = 2.0, dx = 3.0
  store double 5.000000e+00, ptr %acc, align 8
  store double 0.000000e+00, ptr %dacc, align 8
  %dx_4a = call double (ptr, ...) @__enzyme_autodiff_single(ptr @fmin_with_ret, metadata !"enzyme_dup", ptr %acc, ptr %dacc, double 2.000000e+00)
  %da_4a = load double, ptr %dacc, align 8
  call void @check_eq(double %da_4a, double 2.000000e+00)
  call void @check_eq(double %dx_4a, double 3.000000e+00)

  ; 4b: tie (*acc = 2.0, x = 2.0): candidate does NOT store -> dacc = 5.0, dx = 0.0
  store double 2.000000e+00, ptr %acc, align 8
  store double 0.000000e+00, ptr %dacc, align 8
  %dx_4b = call double (ptr, ...) @__enzyme_autodiff_single(ptr @fmin_with_ret, metadata !"enzyme_dup", ptr %acc, ptr %dacc, double 2.000000e+00)
  %da_4b = load double, ptr %dacc, align 8
  call void @check_eq(double %da_4b, double 5.000000e+00)
  call void @check_eq(double %dx_4b, double 0.000000e+00)

  ; Case 5: in-place ReLU with constant operand (atomicrmw fmax ptr %acc, 0.0)
  store double -3.000000e+00, ptr %acc, align 8
  store double 1.000000e+00, ptr %dacc, align 8
  call void (ptr, ...) @__enzyme_autodiff_void(ptr @fmax_relu_inplace, metadata !"enzyme_dup", ptr %acc, ptr %dacc)
  %da_5a = load double, ptr %dacc, align 8
  call void @check_eq(double %da_5a, double 0.000000e+00)

  store double 3.000000e+00, ptr %acc, align 8
  store double 1.000000e+00, ptr %dacc, align 8
  call void (ptr, ...) @__enzyme_autodiff_void(ptr @fmax_relu_inplace, metadata !"enzyme_dup", ptr %acc, ptr %dacc)
  %da_5b = load double, ptr %dacc, align 8
  call void @check_eq(double %da_5b, double 1.000000e+00)

  ; Case 6: loop reduction [3.0, 7.0, 7.0, 5.0] in combined and split reverse mode
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
  store double 0.000000e+00, ptr %dp0, align 8
  store double 0.000000e+00, ptr %dp1, align 8
  store double 0.000000e+00, ptr %dp2, align 8
  store double 0.000000e+00, ptr %dp3, align 8

  call void (ptr, ...) @__enzyme_autodiff_void(ptr @fmax_loop, metadata !"enzyme_dup", ptr %arr, ptr %darr, metadata !"enzyme_const", i64 4)
  %g0 = load double, ptr %dp0, align 8
  %g1 = load double, ptr %dp1, align 8
  %g2 = load double, ptr %dp2, align 8
  %g3 = load double, ptr %dp3, align 8
  call void @check_eq(double %g0, double 0.000000e+00)
  call void @check_eq(double %g1, double 1.000000e+00)
  call void @check_eq(double %g2, double 0.000000e+00)
  call void @check_eq(double %g3, double 0.000000e+00)

  ; Split reverse mode (__enzyme_augmentfwd + __enzyme_reverse)
  store double 0.000000e+00, ptr %dp0, align 8
  store double 0.000000e+00, ptr %dp1, align 8
  store double 0.000000e+00, ptr %dp2, align 8
  store double 0.000000e+00, ptr %dp3, align 8
  %aug = call { ptr, double } (ptr, ...) @__enzyme_augmentfwd(ptr @fmax_loop, metadata !"enzyme_allocated", i64 -1, metadata !"enzyme_dup", ptr %arr, ptr %darr, metadata !"enzyme_const", i64 4)
  %tape = extractvalue { ptr, double } %aug, 0
  %aug_ret = extractvalue { ptr, double } %aug, 1
  call void @check_eq(double %aug_ret, double 7.000000e+00)
  call void (ptr, ...) @__enzyme_reverse(ptr @fmax_loop, metadata !"enzyme_allocated", i64 -1, metadata !"enzyme_dup", ptr %arr, ptr %darr, metadata !"enzyme_const", i64 4, double 1.000000e+00, ptr %tape)
  %sg0 = load double, ptr %dp0, align 8
  %sg1 = load double, ptr %dp1, align 8
  %sg2 = load double, ptr %dp2, align 8
  %sg3 = load double, ptr %dp3, align 8
  call void @check_eq(double %sg0, double 0.000000e+00)
  call void @check_eq(double %sg1, double 1.000000e+00)
  call void @check_eq(double %sg2, double 0.000000e+00)
  call void @check_eq(double %sg3, double 0.000000e+00)

  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffefmin_with_ret(
; CHECK:   %[[INIT:.+]] = load atomic i64, ptr %acc seq_cst, align 8
; CHECK: loop.i:
; CHECK:   %cur.int.i = phi i64 [ %[[INIT]], %entry ], [ %next.int.i, %cas.i ]
; CHECK:   %cur.i = bitcast i64 %cur.int.i to double
; CHECK:   fcmp olt double %x, %cur.i
; CHECK: cas.i:
; CHECK:   cmpxchg ptr %acc, i64 %cur.int.i, i64 %{{.*}} seq_cst seq_cst, align 8
; CHECK: __enzyme_atomic_fmin_double_a8_o7.exit:
; CHECK:   %changed.i = phi i1 [ false, %loop.i ], [ true, %cas.i ]
; CHECK: invertentry:
; CHECK:   br i1 %changed, label %invertentry_changed, label %invertentry_merge
; CHECK: invertentry_changed:
; CHECK:   %[[TAKEN:.+]] = atomicrmw xchg ptr %"acc'", double 0.000000e+00 seq_cst, align 8
; CHECK:   fadd fast double %{{.*}}, %[[TAKEN]]
; CHECK:   br label %invertentry_merge
; CHECK: invertentry_merge:
; CHECK:   atomicrmw fadd ptr %"acc'", double %{{.*}} seq_cst, align 8

; CHECK-LABEL: define internal { double, i1 } @__enzyme_atomic_fmin_double_a8_o7(
; CHECK: entry:
; CHECK:   %[[HINIT:.+]] = load atomic i64, ptr %ptr seq_cst, align 8
; CHECK: loop:
; CHECK:   %cur.int = phi i64 [ %[[HINIT]], %entry ], [ %next.int, %cas ]
; CHECK:   %cur = bitcast i64 %cur.int to double
; CHECK:   fcmp olt double %val, %cur
; CHECK: cas:
; CHECK:   cmpxchg ptr %ptr, i64 %cur.int, i64 %{{.*}} seq_cst seq_cst, align 8
; CHECK: exit:
; CHECK:   %changed = phi i1 [ false, %loop ], [ true, %cas ]

