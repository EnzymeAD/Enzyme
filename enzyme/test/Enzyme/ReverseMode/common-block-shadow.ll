; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A Fortran COMMON block /blk/ a, b of two real(8), as flang emits it.
; Without a context, Enzyme gives it a common shadow, so that every
; translation unit that differentiates through it gets the same storage. In a
; context, its shadow is private to the context, and __enzyme_shadow finds the
; shadow of a member at its offset.

@blk_ = common global [16 x i8] zeroinitializer, align 8
@enzyme_context = external global i32

define double @f(double %x) {
entry:
  %b = load double, ptr getelementptr (i8, ptr @blk_, i64 8), align 8
  %m = fmul double %b, %x
  store double %m, ptr @blk_, align 8
  ret double %m
}

declare ptr @__enzyme_context(i32)
declare ptr @__enzyme_shadow(ptr, ptr, i32)
declare double @__enzyme_autodiff(...)

define double @test(double %x) {
entry:
  %ctx = call ptr @__enzyme_context(i32 1)
  %db = call ptr @__enzyme_shadow(ptr %ctx, ptr getelementptr (i8, ptr @blk_, i64 8), i32 0)
  store double 0.000000e+00, ptr %db, align 8
  %r = call double (...) @__enzyme_autodiff(ptr @f, ptr @enzyme_context, ptr %ctx, double %x)
  %g = load double, ptr %db, align 8
  %s = fadd double %r, %g
  ret double %s
}

define double @nocontext(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %r
}

; CHECK-DAG: @blk_.ad.enzyme.context = private global [16 x i8] zeroinitializer, align 8
; CHECK-DAG: @blk_.ad.w1 = common global [16 x i8] zeroinitializer, align 8
; CHECK-DAG: @blk_ = common global [16 x i8] zeroinitializer, align 8, !enzyme_shadows ![[shadows:[0-9]+]]

; CHECK: define double @test(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store double 0.000000e+00, ptr getelementptr inbounds (i8, ptr @blk_.ad.enzyme.context, i64 8), align 8
; CHECK-NEXT:   %0 = call { double } @diffef(double %x, double 1.000000e+00)
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   %g = load double, ptr getelementptr inbounds (i8, ptr @blk_.ad.enzyme.context, i64 8), align 8

; CHECK: define double @nocontext(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double } @diffef.1(double %x, double 1.000000e+00)

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %b = load double, ptr getelementptr (i8, ptr @blk_, i64 8), align 8
; CHECK-NEXT:   %m = fmul double %b, %x
; CHECK-NEXT:   store double %m, ptr @blk_, align 8
; CHECK-NEXT:   %0 = load double, ptr @blk_.ad.enzyme.context, align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr @blk_.ad.enzyme.context, align 8
; CHECK-NEXT:   %1 = fadd fast double %differeturn, %0
; CHECK-NEXT:   %2 = fmul fast double %1, %x
; CHECK-NEXT:   %3 = fmul fast double %1, %b
; CHECK-NEXT:   %4 = load double, ptr getelementptr (i8, ptr @blk_.ad.enzyme.context, i64 8), align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %2
; CHECK-NEXT:   store double %5, ptr getelementptr (i8, ptr @blk_.ad.enzyme.context, i64 8), align 8
; CHECK-NEXT:   %6 = insertvalue { double } undef, double %3, 0
; CHECK-NEXT:   ret { double } %6

; CHECK: define internal { double } @diffef.1(double %x, double %differeturn)
; CHECK:   %0 = load double, ptr @blk_.ad.w1, align 8

; CHECK: ![[shadows]] = !{![[ctxentry:[0-9]+]], ![[w1entry:[0-9]+]]}
; CHECK: ![[ctxentry]] = !{ptr @enzyme.context, ptr @blk_.ad.enzyme.context}
; CHECK: ![[w1entry]] = !{i32 1, ptr @blk_.ad.w1}
