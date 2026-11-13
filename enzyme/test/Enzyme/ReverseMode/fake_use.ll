; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify)" -S | FileCheck %s; fi

declare void @llvm.fake.use(...)
declare double @__enzyme_autodiff(ptr, ...)
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

define double @dsquare(double %x) {
entry:
  %r = call double (ptr, ...) @__enzyme_autodiff(ptr @square, double %x)
  ret double %r
}

define void @dptr(ptr %p, ptr %dp) {
entry:
  call double (ptr, ...) @__enzyme_autodiff(ptr @ptr_fake_use, ptr @enzyme_dup, ptr %p, ptr %dp)
  ret void
}

; CHECK: define internal { double } @diffesquare(double %x, double %differeturn)
; CHECK:   %[[m:.+]] = fmul fast double %differeturn, %x
; CHECK:   %[[a:.+]] = fadd fast double %[[m]], %[[m]]
; CHECK:   %[[r:.+]] = insertvalue { double } undef, double %[[a]], 0
; CHECK:   ret { double } %[[r]]

; CHECK: define internal void @diffeptr_fake_use(ptr{{.*}} %p, ptr %"p'", double %differeturn)
; CHECK:   %[[v:.+]] = load double, ptr %p
; CHECK:   %[[m2:.+]] = fmul fast double %differeturn, %[[v]]
; CHECK:   %[[a2:.+]] = fadd fast double %[[m2]], %[[m2]]
; CHECK:   %[[cur:.+]] = load double, ptr %"p'"
; CHECK:   %[[upd:.+]] = fadd fast double %[[cur]], %[[a2]]
; CHECK:   store double %[[upd]], ptr %"p'"
; CHECK:   ret void
