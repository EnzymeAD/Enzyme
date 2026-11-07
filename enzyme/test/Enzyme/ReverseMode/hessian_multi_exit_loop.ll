; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli -

declare void @abort()
declare double @llvm.fabs.f64(double)
declare double @__enzyme_autodiff(...)

define double @multi_exit_poly(ptr %arr, i64 %n, double %x) noinline {
entry:
  br label %header

header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi double [ %x, %entry ], [ %acc.next, %latch ]
  %inbounds = icmp ult i64 %iv, %n
  br i1 %inbounds, label %latch, label %exit

latch:
  %elem.ptr = getelementptr inbounds double, ptr %arr, i64 %iv
  %elem = load double, ptr %elem.ptr, align 8
  %sq = fmul double %acc, %acc
  %acc.next = fadd double %sq, %elem
  %iv.next = add nuw nsw i64 %iv, 1
  %pos = fcmp ogt double %elem, 0.0
  br i1 %pos, label %header, label %exit

exit:
  %res = phi double [ %acc, %header ], [ %acc, %latch ]
  ret double %res
}

define double @grad_multi_exit_poly(ptr %arr, i64 %n, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @multi_exit_poly, metadata !"enzyme_const", ptr %arr, metadata !"enzyme_const", i64 %n, double %x)
  ret double %dx
}

define i32 @main() {
entry:
  %arr = alloca [4 x double], align 8
  store double 1.0, ptr %arr, align 8
  %p1 = getelementptr inbounds double, ptr %arr, i64 1
  store double 2.0, ptr %p1, align 8
  %p2 = getelementptr inbounds double, ptr %arr, i64 2
  store double -5.0, ptr %p2, align 8
  %p3 = getelementptr inbounds double, ptr %arr, i64 3
  store double 4.0, ptr %p3, align 8

  ; Exit via latch (elem <= 0 at iv = 2): f(x) = (x^2 + 1)^2 + 2, f''(3) = 12*9 + 4 = 112
  %d2_latch = call double (...) @__enzyme_autodiff(ptr @grad_multi_exit_poly, metadata !"enzyme_const", ptr %arr, metadata !"enzyme_const", i64 4, double 3.0)
  %err1 = fsub double %d2_latch, 112.0
  %abs1 = call double @llvm.fabs.f64(double %err1)
  %bad1 = fcmp ogt double %abs1, 1.0e-9
  br i1 %bad1, label %fail, label %check_header_exit

check_header_exit:
  ; Exit via header (iv == n at iv = 1): f(x) = x^2 + 1, f''(3) = 2
  %d2_header = call double (...) @__enzyme_autodiff(ptr @grad_multi_exit_poly, metadata !"enzyme_const", ptr %arr, metadata !"enzyme_const", i64 1, double 3.0)
  %err2 = fsub double %d2_header, 2.0
  %abs2 = call double @llvm.fabs.f64(double %err2)
  %bad2 = fcmp ogt double %abs2, 1.0e-9
  br i1 %bad2, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffegrad_multi_exit_poly(
; CHECK: call {{.*}}@diffediffemulti_exit_poly(
; CHECK: ret { double }

; CHECK-LABEL: define internal {{.*}} @diffediffemulti_exit_poly(
; CHECK: irr.guard:
; CHECK: ret
