; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A Fortran COMMON block /blk/ a, b of two real(8), as flang emits it. Enzyme
; gives it a common shadow, so every translation unit that differentiates
; through it or queries it gets the same storage, and __enzyme_shadow finds
; the shadow of a member at its offset.

@blk_ = common global [16 x i8] zeroinitializer, align 8

define double @f(double %x) {
entry:
  %b = load double, ptr getelementptr (i8, ptr @blk_, i64 8), align 8
  %m = fmul double %b, %x
  store double %m, ptr @blk_, align 8
  ret double %m
}

declare ptr @__enzyme_shadow(ptr, i32, i32)
declare double @__enzyme_autodiff(...)

define double @test(double %x) {
entry:
  %db = call ptr @__enzyme_shadow(ptr getelementptr (i8, ptr @blk_, i64 8), i32 1, i32 0)
  store double 0.000000e+00, ptr %db, align 8
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  %g = load double, ptr %db, align 8
  %s = fadd double %r, %g
  ret double %s
}

; CHECK: @blk_.ad.l1.w1 = common global [16 x i8] zeroinitializer, align 8
; CHECK: @blk_ = common global [16 x i8] zeroinitializer, align 8, !enzyme_shadows ![[shadows:[0-9]+]]

; CHECK: define double @test(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store double 0.000000e+00, ptr getelementptr inbounds (i8, ptr @blk_.ad.l1.w1, i64 8), align 8
; CHECK-NEXT:   %0 = call { double } @diffef(double %x, double 1.000000e+00)
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   %g = load double, ptr getelementptr inbounds (i8, ptr @blk_.ad.l1.w1, i64 8), align 8

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %b = load double, ptr getelementptr (i8, ptr @blk_, i64 8), align 8
; CHECK-NEXT:   %m = fmul double %b, %x
; CHECK-NEXT:   store double %m, ptr @blk_, align 8
; CHECK-NEXT:   %0 = load double, ptr @blk_.ad.l1.w1, align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr @blk_.ad.l1.w1, align 8
; CHECK-NEXT:   %1 = fadd fast double %differeturn, %0
; CHECK-NEXT:   %2 = fmul fast double %1, %x
; CHECK-NEXT:   %3 = fmul fast double %1, %b
; CHECK-NEXT:   %4 = load double, ptr getelementptr (i8, ptr @blk_.ad.l1.w1, i64 8), align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %2
; CHECK-NEXT:   store double %5, ptr getelementptr (i8, ptr @blk_.ad.l1.w1, i64 8), align 8
; CHECK-NEXT:   %6 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %6

; CHECK: ![[shadows]] = !{![[entry:[0-9]+]]}
; CHECK: ![[entry]] = !{i32 1, i32 1, ptr @blk_.ad.l1.w1}
