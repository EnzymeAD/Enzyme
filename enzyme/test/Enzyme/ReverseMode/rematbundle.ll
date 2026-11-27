; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; A stack slot whose only unusual use is a "jl_roots" operand bundle on an intrinsic (Julia keeps objects
; alive across a call this way). The bundle is not a memory access, so the slot is still rematerialized in
; the reverse pass, as without it: no heap allocation (!enzyme_fromstack), and the augmented function
; returns no pointer to it (only the call's result, which is cached because the call carries a bundle).

declare double @llvm.fmuladd.f64(double, double, double)
declare void @__enzyme_autodiff(...)

define internal double @kern(double %x, double %y) noinline {
entry:
  %slot = alloca double, align 8
  store double %x, ptr %slot, align 8
  %f = call double @llvm.fmuladd.f64(double %y, double %y, double 1.0) [ "jl_roots"(ptr %slot) ]
  %v = load double, ptr %slot, align 8
  %sq = fmul double %v, %v
  %r = fmul double %sq, %f
  ret double %r
}

define double @outer(double %x, double %y) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %accn, %loop ]
  %k = call double @kern(double %x, double %y)
  %accn = fadd double %acc, %k
  %inext = add i64 %i, 1
  %done = icmp eq i64 %inext, 2
  br i1 %done, label %exit, label %loop
exit:
  ret double %accn
}

define void @dsquare(double %x, double %y) {
entry:
  call void (...) @__enzyme_autodiff(ptr @outer, double %x, double %y)
  ret void
}

; CHECK: define internal double @augmented_kern(double %x, double %y)
; CHECK-NOT: enzyme_fromstack
; CHECK: define internal { double, double } @diffekern(double %x, double %y, double %differeturn, double %f)
