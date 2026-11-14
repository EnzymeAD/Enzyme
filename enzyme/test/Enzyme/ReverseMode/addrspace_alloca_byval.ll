; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare double @__enzyme_autodiff(...)

define double @f_alloca_as5(double %x, i64 %n) {
entry:
  %a = alloca double, i64 4, align 8, addrspace(5)
  %p = getelementptr inbounds double, ptr addrspace(5) %a, i64 %n
  store double %x, ptr addrspace(5) %p, align 8
  %v = load double, ptr addrspace(5) %p, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @f_byval_as1(ptr addrspace(1) byval([2 x double]) %p, double %x) {
entry:
  store double %x, ptr addrspace(1) %p, align 8
  %v = load double, ptr addrspace(1) %p, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @test_alloca_as5(double %x, i64 %n) {
  %r = call double (...) @__enzyme_autodiff(ptr @f_alloca_as5, double %x, i64 %n)
  ret double %r
}

define double @test_byval_as1(ptr addrspace(1) %p, double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f_byval_as1, metadata !"enzyme_const", ptr addrspace(1) %p, double %x)
  ret double %r
}

; CHECK: define internal { double } @diffef_alloca_as5(
; CHECK:   %"a'ipa" = alloca double, i64 4, align 8, addrspace(5)
; CHECK-NEXT:   call void @llvm.memset.p5.i64(ptr addrspace(5) nonnull align 8 %"a'ipa", i8 0, i64 32, i1 false)

; CHECK: define internal { double } @diffef_byval_as1(
; CHECK:   %"p'ipa" = alloca [2 x double], align 8, addrspace(1)
; CHECK-NEXT:   call void @llvm.memset.p1.i64(ptr addrspace(1) %"p'ipa", i8 0, i64 16, i1 false)
