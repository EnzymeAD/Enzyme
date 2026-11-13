; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -inline -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,inline,enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare i32 @llvm.nvvm.d2i.hi(double) #1
declare i32 @llvm.nvvm.d2i.lo(double) #1
declare double @llvm.nvvm.lohi.i2d(i32, i32) #1
declare double @llvm.nvvm.fmin.d(double, double) #1
declare double @llvm.nvvm.rsqrt.approx.d(double) #1

define double @__nv_copysign(double %a, double %b) #0 {
  %1 = call i32 @llvm.nvvm.d2i.hi(double %b)
  %2 = call i32 @llvm.nvvm.d2i.lo(double %a)
  %3 = call i32 @llvm.nvvm.d2i.hi(double %a)
  %4 = and i32 %1, -2147483648
  %5 = and i32 %3, 2147483647
  %6 = or i32 %5, %4
  %7 = call double @llvm.nvvm.lohi.i2d(i32 %2, i32 %6)
  ret double %7
}

define double @__nv_fmin(double %a, double %b) #0 {
  %r = call double @llvm.nvvm.fmin.d(double %a, double %b)
  ret double %r
}

define double @__nv_rsqrt(double %a) #0 {
  %r = call double @llvm.nvvm.rsqrt.approx.d(double %a)
  ret double %r
}

define double @tester_copysign(double %x) {
entry:
  %r = call double @__nv_copysign(double %x, double -1.0)
  ret double %r
}

define double @tester_fmin(double %x) {
entry:
  %r = call double @__nv_fmin(double %x, double 5.000000e-01)
  ret double %r
}

define double @tester_rsqrt(double %x) {
entry:
  %r = call double @__nv_rsqrt(double %x)
  ret double %r
}

declare double @__enzyme_autodiff(double (double)*, ...)

define double @d_copysign(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester_copysign, double %x)
  ret double %r
}

define double @d_fmin(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester_fmin, double %x)
  ret double %r
}

define double @d_rsqrt(double %x) {
  %r = call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester_rsqrt, double %x)
  ret double %r
}

attributes #0 = { alwaysinline nounwind }
attributes #1 = { nofree nosync nounwind willreturn readnone }

; CHECK: define double @__nv_copysign(double %a, double %b) #[[CSATTR:[0-9]+]]
; CHECK: define double @__nv_fmin(double %a, double %b) #[[FMINATTR:[0-9]+]]
; CHECK: define double @__nv_rsqrt(double %a) #[[RSQRTATTR:[0-9]+]]

; CHECK: define internal { double } @diffetester_copysign(double %x, double %differeturn)
; CHECK:   %[[C0:.+]] = call fast double @__nv_copysign(double 1.000000e+00, double %x)
; CHECK-NEXT:   %[[C1:.+]] = call fast double @__nv_copysign(double 1.000000e+00, double -1.000000e+00)
; CHECK-NEXT:   %[[C2:.+]] = fmul fast double %[[C0]], %[[C1]]
; CHECK-NEXT:   %[[C3:.+]] = fmul fast double %differeturn, %[[C2]]
; CHECK-NEXT:   %[[C4:.+]] = insertvalue { double } undef, double %[[C3]], 0
; CHECK-NEXT:   ret { double } %[[C4]]

; CHECK: define internal { double } @diffetester_fmin(double %x, double %differeturn)
; CHECK:   %[[M0:.+]] = fcmp fast olt double %x, 5.000000e-01
; CHECK-NEXT:   %[[M1:.+]] = select fast i1 %[[M0]], double %differeturn, double 0.000000e+00
; CHECK-NEXT:   %[[M2:.+]] = insertvalue { double } undef, double %[[M1]], 0
; CHECK-NEXT:   ret { double } %[[M2]]

; CHECK: define internal { double } @diffetester_rsqrt(double %x, double %differeturn)
; CHECK:   %[[R0:.+]] = call fast double @__nv_rsqrt(double %x)
; CHECK-NEXT:   %[[R1:.+]] = fmul fast double %differeturn, %[[R0]]
; CHECK-NEXT:   %[[R2:.+]] = fmul fast double 2.000000e+00, %x
; CHECK-NEXT:   %[[R3:.+]] = fdiv fast double %[[R1]], %[[R2]]
; CHECK-NEXT:   %[[R4:.+]] = fneg fast double %[[R3]]
; CHECK-NEXT:   %[[R5:.+]] = insertvalue { double } undef, double %[[R4]], 0
; CHECK-NEXT:   ret { double } %[[R5]]

; CHECK: attributes #[[CSATTR]] = { {{.*}}noinline{{.*}}"enzyme_math"="copysign" "implements"="llvm.copysign.f64" "implements2"="copysign"
; CHECK: attributes #[[FMINATTR]] = { {{.*}}noinline{{.*}}"enzyme_math"="fmin" "implements"="llvm.fmin.f64" "implements2"="fmin"
; CHECK: attributes #[[RSQRTATTR]] = { {{.*}}noinline{{.*}}"enzyme_math"="rsqrt" "implements"="llvm.rsqrt.f64" "implements2"="rsqrt"
