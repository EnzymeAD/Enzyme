; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-loose-types=1 -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-loose-types=1 -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-loose-types=1 -passes="enzyme" -S | %lli -

@observed = global i64 0, align 8, !enzyme_inactive !0

declare void @abort()
declare double @__enzyme_fwddiff(ptr, ...)
declare [2 x double] @__enzyme_fwddiff_v2(ptr, ...)
declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_fwdsplit(...)

; CHECK-LABEL: define internal double @fwddiffetyped_umax_square(
; CHECK-NOT: dual_phi
; CHECK: atomicrmw umax ptr %box, i64 5 seq_cst, align 8
; CHECK-NOT: dual_phi
; CHECK: ret double

; CHECK-LABEL: define internal [2 x double] @fwddiffe2typed_umax_square(
; CHECK-NOT: dual_phi
; CHECK: atomicrmw umax ptr %box, i64 5 seq_cst, align 8
; CHECK-NOT: dual_phi
; CHECK: ret [2 x double]
define double @typed_umax_square(ptr %box) noinline {
entry:
  %tag = load i64, ptr %box, align 8, !enzyme_type !1
  %old = atomicrmw umax ptr %box, i64 5 seq_cst, align 8
  %sum = add i64 %old, %tag
  store i64 %sum, ptr @observed, align 8
  %val_ptr = getelementptr inbounds i8, ptr %box, i64 8
  %val = load double, ptr %val_ptr, align 8, !enzyme_type !3
  %sq = fmul double %val, %val
  ret double %sq
}

; CHECK-LABEL: define internal double @fwddiffeloose_retain_eval(
; CHECK-NOT: dual_phi
; CHECK: atomicrmw add ptr %box, i64 1 seq_cst, align 8
; CHECK-NOT: dual_phi
; CHECK: ret double
define double @loose_retain_eval(ptr %box) noinline {
entry:
  %old1 = atomicrmw add ptr %box, i64 1 seq_cst, align 8
  %old2 = atomicrmw add ptr %box, i64 1 seq_cst, align 8
  %sum = add i64 %old1, %old2
  store i64 %sum, ptr @observed, align 8
  %val_ptr = getelementptr inbounds i8, ptr %box, i64 8
  %val = load double, ptr %val_ptr, align 8
  %sq = fmul double %val, %val
  ret double %sq
}

; CHECK-LABEL: define internal void @fwddiffeloose_retain_scale(
; CHECK-NOT: dual_phi
; CHECK-NOT: atomicrmw
; CHECK: ret void
define void @loose_retain_scale(ptr %box) noinline {
entry:
  %old1 = atomicrmw add ptr %box, i64 1 seq_cst, align 8
  %old2 = atomicrmw add ptr %box, i64 1 seq_cst, align 8
  %sum = add i64 %old1, %old2
  store i64 %sum, ptr @observed, align 8
  %val_ptr = getelementptr inbounds i8, ptr %box, i64 8
  %val = load double, ptr %val_ptr, align 8
  %sq = fmul double %val, %val
  store double %sq, ptr %val_ptr, align 8
  ret void
}

define void @check_eq(double %actual, double %expected) {
entry:
  %err = fsub double %actual, %expected
  %bad_hi = fcmp ogt double %err, 0.0009765625
  %bad_lo = fcmp olt double %err, -0.0009765625
  %bad = or i1 %bad_hi, %bad_lo
  br i1 %bad, label %fail, label %done

fail:
  call void @abort()
  unreachable

done:
  ret void
}

define void @check_i64_eq(i64 %actual, i64 %expected) {
entry:
  %bad = icmp ne i64 %actual, %expected
  br i1 %bad, label %fail, label %done

fail:
  call void @abort()
  unreachable

done:
  ret void
}

define i32 @main() {
entry:
  %box = alloca { i64, double }, align 8
  %dbox = alloca { i64, double }, align 8
  %dbox2 = alloca { i64, double }, align 8
  %val_ptr = getelementptr inbounds i8, ptr %box, i64 8
  %dval_ptr = getelementptr inbounds i8, ptr %dbox, i64 8
  %dval2_ptr = getelementptr inbounds i8, ptr %dbox2, i64 8

  ; 1. Constant-instruction atomicrmw umax with active-placeholder return.
  store i64 2, ptr %box, align 8
  store double 3.0, ptr %val_ptr, align 8
  store i64 0, ptr %dbox, align 8
  store double 2.0, ptr %dval_ptr, align 8
  %d1 = call double (ptr, ...) @__enzyme_fwddiff(ptr @typed_umax_square, metadata !"enzyme_dup", ptr %box, ptr %dbox)
  call void @check_eq(double %d1, double 12.0)
  %u_cnt = load i64, ptr %box, align 8
  call void @check_i64_eq(i64 %u_cnt, i64 5)

  store i64 1, ptr %box, align 8
  store double 4.0, ptr %val_ptr, align 8
  store double 1.5, ptr %dval_ptr, align 8
  store i64 0, ptr %dbox2, align 8
  store double -0.5, ptr %dval2_ptr, align 8
  %db = call [2 x double] (ptr, ...) @__enzyme_fwddiff_v2(ptr @typed_umax_square, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %box, ptr %dbox, ptr %dbox2)
  %db0 = extractvalue [2 x double] %db, 0
  %db1 = extractvalue [2 x double] %db, 1
  call void @check_eq(double %db0, double 12.0)
  call void @check_eq(double %db1, double -4.0)

  ; 2. Loose-type integer atomicrmw in ForwardMode and ForwardModeSplit.
  store i64 10, ptr %box, align 8
  store double 5.0, ptr %val_ptr, align 8
  store i64 0, ptr %dbox, align 8
  store double 3.0, ptr %dval_ptr, align 8
  %dl = call double (ptr, ...) @__enzyme_fwddiff(ptr @loose_retain_eval, metadata !"enzyme_dup", ptr %box, ptr %dbox)
  call void @check_eq(double %dl, double 30.0)
  %c_after_fwd = load i64, ptr %box, align 8
  call void @check_i64_eq(i64 %c_after_fwd, i64 12)

  ; Split forward mode must execute the two primal atomicrmws only in augmentfwd.
  store i64 0, ptr %box, align 8
  store double 5.0, ptr %val_ptr, align 8
  store i64 0, ptr %dbox, align 8
  store double 3.0, ptr %dval_ptr, align 8
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @loose_retain_scale, metadata !"enzyme_dup", ptr %box, ptr %dbox)
  call void (...) @__enzyme_fwdsplit(ptr @loose_retain_scale, metadata !"enzyme_nofree", metadata !"enzyme_dup", ptr %box, ptr %dbox, ptr %tape)
  %p_out = load double, ptr %val_ptr, align 8
  %d_out = load double, ptr %dval_ptr, align 8
  %c_after_split = load i64, ptr %box, align 8
  call void @check_eq(double %p_out, double 25.0)
  call void @check_eq(double %d_out, double 30.0)
  call void @check_i64_eq(i64 %c_after_split, i64 2)

  ret i32 0
}

!0 = !{}
!1 = !{!"Unknown", i32 -1, !2}
!2 = !{!"Integer"}
!3 = !{!"Unknown", i32 -1, !4}
!4 = !{!"Float@double"}
