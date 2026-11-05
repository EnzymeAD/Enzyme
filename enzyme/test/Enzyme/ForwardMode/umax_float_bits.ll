; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; The unsigned max of the bits of two nonnegative doubles is their max, as in
; Julia's `hypot`. Its derivative selects the shadow of the larger operand.

declare i64 @llvm.umax.i64(i64, i64)

define double @f(double %x, double %y) {
entry:
  %xi = bitcast double %x to i64
  %yi = bitcast double %y to i64
  %m = call i64 @llvm.umax.i64(i64 %xi, i64 %yi)
  %r = bitcast i64 %m to double
  ret double %r
}

declare double @__enzyme_fwddiff(...)

define double @df(double %x, double %dx, double %y, double %dy) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @f, double %x, double %dx, double %y, double %dy)
  ret double %r
}

; CHECK: define internal double @fwddiffef(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"xi'ipc" = bitcast double %"x'" to i64
; CHECK-NEXT:   %xi = bitcast double %x to i64
; CHECK-NEXT:   %"yi'ipc" = bitcast double %"y'" to i64
; CHECK-NEXT:   %yi = bitcast double %y to i64
; CHECK-NEXT:   %0 = icmp ult i64 %xi, %yi
; CHECK-NEXT:   %1 = select i1 %0, i64 %"yi'ipc", i64 %"xi'ipc"
; CHECK-NEXT:   %"r'ipc" = bitcast i64 %1 to double
; CHECK-NEXT:   ret double %"r'ipc"
; CHECK-NEXT: }
