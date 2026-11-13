; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify)" -S | FileCheck %s; fi

declare void @llvm.fake.use(...)
declare double @__enzyme_fwddiff(ptr, ...)
@enzyme_dup = external global i32

define double @square(double %x) {
entry:
  %a = alloca double, align 8
  store double %x, ptr %a, align 8
  %v = load double, ptr %a, align 8
  %sq = fmul double %v, %v
  call void (...) @llvm.fake.use(ptr %a, double %v, double %sq)
  ret double %sq
}

define double @ptr_fake_use(ptr %p) {
entry:
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  call void (...) @llvm.fake.use(ptr %p, double %v), !enzyme_active !0
  ret double %sq
}

!0 = !{}

define double @dsquare(double %x, double %dx) {
entry:
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @square, double %x, double %dx)
  ret double %r
}

define double @dptr(ptr %p, ptr %dp) {
entry:
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @ptr_fake_use, ptr @enzyme_dup, ptr %p, ptr %dp)
  ret double %r
}

; CHECK: define internal double @fwddiffesquare(double %x, double %"x'")
; CHECK:   %[[m:.+]] = fmul fast double %"x'", %x
; CHECK:   %[[a:.+]] = fadd fast double %[[m]], %[[m]]
; CHECK:   ret double %[[a]]

; CHECK: define internal double @fwddiffeptr_fake_use(ptr{{.*}} %p, ptr %"p'")
; CHECK:   %[[dv:.+]] = load double, ptr %"p'"
; CHECK:   %[[v:.+]] = load double, ptr %p
; CHECK:   %[[m2:.+]] = fmul fast double %[[dv]], %[[v]]
; CHECK:   %[[a2:.+]] = fadd fast double %[[m2]], %[[m2]]
; CHECK:   ret double %[[a2]]
