; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,default<O2>,enzyme,preserve-nvvm-end" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,default<O2>,enzyme,preserve-nvvm-end" -S | %lli -

; Custom derivative targets must not have body-only attributes (such as
; writeonly or readnone) inferred or call sites constant-folded by pre-Enzyme
; optimization passes, while still having their original linkage restored by
; preserve-nvvm-end.

@enzyme_dup = global i32 0, align 4

@__enzyme_register_derivative_ipabsterm = global [2 x ptr] [ptr @ipabsterm, ptr @derivative_ipabsterm], align 8
@__enzyme_register_derivative_stub_op = global [2 x ptr] [ptr @stub_op, ptr @derivative_stub_op], align 8

declare double @llvm.fabs.f64(double)
declare double @__enzyme_fwddiff(ptr, ...)

define void @ipabsterm(ptr %at, ptr %t) {
entry:
  %val = load double, ptr %t, align 8
  %abs = call double @llvm.fabs.f64(double %val)
  store double %abs, ptr %at, align 8
  ret void
}

define void @derivative_ipabsterm(ptr %at, ptr %d_at, ptr %t, ptr %d_t) {
entry:
  %tv = load double, ptr %t, align 8
  %dtv = load double, ptr %d_t, align 8
  %abstv = call double @llvm.fabs.f64(double %tv)
  %absdtv = call double @llvm.fabs.f64(double %dtv)
  %cmp = fcmp ogt double %abstv, %absdtv
  %mx = select i1 %cmp, double %abstv, double %absdtv
  store double %mx, ptr %at, align 8
  store double 0.000000e+00, ptr %d_at, align 8
  ret void
}

define double @series_exp(double %x) {
entry:
  %at = alloca double, align 8
  %term = alloca double, align 8
  store double 1.000000e+00, ptr %at, align 8
  store double 1.000000e+00, ptr %term, align 8
  br label %loop

loop:
  %k = phi i32 [ 1, %entry ], [ %k.next, %body ]
  %sum = phi double [ 1.000000e+00, %entry ], [ %sum.next, %body ]
  %at.cur = load double, ptr %at, align 8
  %gt = fcmp ogt double %at.cur, 9.765625e-04
  %lt = icmp slt i32 %k, 20
  %cont = select i1 %gt, i1 %lt, i1 false
  br i1 %cont, label %body, label %exit

body:
  %t.old = load double, ptr %term, align 8
  %kf = sitofp i32 %k to double
  %step = fdiv double %x, %kf
  %t.new = fmul double %t.old, %step
  store double %t.new, ptr %term, align 8
  %sum.next = fadd double %sum, %t.new
  call void @ipabsterm(ptr %at, ptr %term)
  %k.next = add i32 %k, 1
  br label %loop

exit:
  ret double %sum
}

define internal double @stub_op(ptr %x) {
entry:
  ret double 0.000000e+00
}

define internal { double, double } @derivative_stub_op(ptr %x, ptr %dx) {
entry:
  %xv = load double, ptr %x, align 8
  %dxv = load double, ptr %dx, align 8
  %p = fmul double %xv, %xv
  %twox = fmul double 2.000000e+00, %xv
  %d = fmul double %twox, %dxv
  %r0 = insertvalue { double, double } undef, double %p, 0
  %r1 = insertvalue { double, double } %r0, double %d, 1
  ret { double, double } %r1
}

define double @use_stub(double %x) {
entry:
  %buf = alloca double, align 8
  store double %x, ptr %buf, align 8
  %res = call double @stub_op(ptr %buf)
  ret double %res
}

define i32 @main() {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
  %d1 = call double (ptr, ...) @__enzyme_fwddiff(ptr @series_exp, i32 %dup, double 0.000000e+00, double 1.000000e+00)
  %err1 = fsub double %d1, 1.000000e+00
  %abs1 = call double @llvm.fabs.f64(double %err1)
  %ok1 = fcmp olt double %abs1, 9.765625e-04

  %d2 = call double (ptr, ...) @__enzyme_fwddiff(ptr @use_stub, i32 %dup, double 3.000000e+00, double 1.000000e+00)
  %err2 = fsub double %d2, 6.000000e+00
  %abs2 = call double @llvm.fabs.f64(double %err2)
  %ok2 = fcmp olt double %abs2, 9.765625e-04

  %ok = select i1 %ok1, i1 %ok2, i1 false
  %ret = select i1 %ok, i32 0, i32 1
  ret i32 %ret
}

; CHECK-LABEL: define void @ipabsterm(
; CHECK-NOT: writeonly
; CHECK-SAME: %at,
; CHECK-LABEL: define internal double @stub_op(
; CHECK-NOT: readnone
; CHECK-SAME: %x)
; CHECK-LABEL: define internal double @fwddiffeseries_exp(
; CHECK: call void @derivative_ipabsterm(
; CHECK-LABEL: define internal double @fwddiffeuse_stub(
; CHECK: call {{.*}}@{{.*}}derivative_stub_op(
