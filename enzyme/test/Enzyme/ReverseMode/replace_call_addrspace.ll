; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

target datalayout = "e-p:64:64-p5:32:32-A5"

define double @f(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

declare void @__enzyme_autodiff(ptr addrspace(1) sret({ i64 }), ...)
declare i64 @__enzyme_fwddiff(...)

define void @test_sret_as1(ptr addrspace(1) %out, double %x) {
  call void (ptr addrspace(1), ...) @__enzyme_autodiff(ptr addrspace(1) sret({ i64 }) %out, ptr @f, double %x)
  ret void
}

define i64 @test_alloca_as5(double %x, double %dx) {
  %r = call i64 (...) @__enzyme_fwddiff(ptr @f, double %x, double %dx)
  ret i64 %r
}

; CHECK: define void @test_sret_as1(
; CHECK-NOT: addrspacecast
; CHECK:   %[[D1:.+]] = call { double } @diffef(double %x, double 1.000000e+00)
; CHECK-NEXT:   store { double } %[[D1]], ptr addrspace(1) %out
; CHECK-NEXT:   ret void

; CHECK: define i64 @test_alloca_as5(
; CHECK-NOT: addrspacecast
; CHECK:   %[[A:.+]] = alloca i64, align 8, addrspace(5)
; CHECK-NEXT:   %[[D2:.+]] = call fast double @fwddiffef(double %x, double %dx)
; CHECK-NEXT:   store double %[[D2]], ptr addrspace(5) %[[A]]
; CHECK-NEXT:   %[[L:.+]] = load i64, ptr addrspace(5) %[[A]]
; CHECK-NEXT:   ret i64 %[[L]]
