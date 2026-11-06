; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EXEC

; The norm is not differentiable at zero. Its reverse rule scales x by
; dret / nrm2(x), which used to divide zero by zero there; the scale is now
; taken to be zero, as in ChainRules.

declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
@fmt = private constant [7 x i8] c"%f %f\0A\00"

define double @f(i32 %n, ptr %x) {
  %r = call double @cblas_dnrm2(i32 %n, ptr %x, i32 1)
  ret double %r
}

define void @df(i32 %n, ptr %x, ptr %dx) {
  call double (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_const", i32 %n, metadata !"enzyme_dup", ptr %x, ptr %dx)
  ret void
}

define void @cblas_daxpy(i32 %n, double %alpha, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %xp = getelementptr double, ptr %x, i32 %i
  %yp = getelementptr double, ptr %y, i32 %i
  %xv = load double, ptr %xp
  %yv = load double, ptr %yp
  %m = fmul double %xv, %alpha
  %s = fadd double %yv, %m
  store double %s, ptr %yp
  %inext = add i32 %i, 1
  %done = icmp eq i32 %inext, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define double @cblas_ddot(i32 %n, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %accnext, %loop ]
  %xp = getelementptr double, ptr %x, i32 %i
  %yp = getelementptr double, ptr %y, i32 %i
  %xv = load double, ptr %xp
  %yv = load double, ptr %yp
  %m = fmul double %xv, %yv
  %accnext = fadd double %acc, %m
  %inext = add i32 %i, 1
  %done = icmp eq i32 %inext, %n
  br i1 %done, label %exit, label %loop
exit:
  ret double %accnext
}

define double @cblas_dnrm2(i32 %n, ptr %x, i32 %incx) noinline {
  %d = call double @cblas_ddot(i32 %n, ptr %x, i32 1, ptr %x, i32 1)
  %r = call double @llvm.sqrt.f64(double %d)
  ret double %r
}

declare double @llvm.sqrt.f64(double)

define i32 @main() {
  %x = alloca [2 x double], align 8
  %dx = alloca [2 x double], align 8
  store [2 x double] [double 3.0, double 4.0], ptr %x
  store [2 x double] zeroinitializer, ptr %dx
  call void @df(i32 2, ptr %x, ptr %dx)
  %away = load double, ptr %dx
  store [2 x double] zeroinitializer, ptr %x
  store [2 x double] zeroinitializer, ptr %dx
  call void @df(i32 2, ptr %x, ptr %dx)
  %zero = load double, ptr %dx
  call i32 (ptr, ...) @printf(ptr @fmt, double %away, double %zero)
  ret i32 0
}

; EXEC: 0.600000 0.000000

; CHECK-LABEL: define internal void @diffef(i32 %n, ptr nocapture readonly %x, ptr nocapture %"x'", double %differeturn)
; CHECK: %[[DRET:.+]] = load double, ptr %"r'de"
; CHECK-NEXT: %[[NRM:.+]] = call fast double @cblas_dnrm2(i32 %n, ptr %x, i32 1)
; CHECK-NEXT: %[[ISZERO:.+]] = fcmp fast oeq double %[[NRM]], 0.000000e+00
; CHECK-NEXT: %[[NRM2:.+]] = call fast double @cblas_dnrm2(i32 %n, ptr %x, i32 1)
; CHECK-NEXT: %[[DIV:.+]] = fdiv fast double %[[DRET]], %[[NRM2]]
; CHECK-NEXT: %[[ALPHA:.+]] = select fast i1 %[[ISZERO]], double 0.000000e+00, double %[[DIV]]
; CHECK-NEXT: call void @cblas_daxpy(i32 %n, double %[[ALPHA]], ptr %x, i32 1, ptr %"x'", i32 1)
