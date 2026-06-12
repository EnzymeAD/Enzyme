; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Julia 1.13's atomic modify pseudo-intrinsic
;   {old, new} = julia.atomicmodify.iN.pAS(ptr, op, ordering, syncscope, args...)
; atomically performs old = *ptr; new = op(old, args...); *ptr = new.
; Since op is linear for fadd, the tangent is computed by applying the same op
; to the shadow location and the shadow value.

declare { i64, i64 } @julia.atomicmodify.i64.p0(ptr, ptr, i8, i8, ...)

define internal i64 @fadd_op(i64 %old, i64 %v) {
  %oldf = bitcast i64 %old to double
  %vf = bitcast i64 %v to double
  %r = fadd double %oldf, %vf
  %ri = bitcast double %r to i64
  ret i64 %ri
}

define double @foo(ptr %p, double %vf) {
  %v = bitcast double %vf to i64
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
  %new = extractvalue { i64, i64 } %on, 1
  %newf = bitcast i64 %new to double
  ret double %newf
}

define double @caller(ptr %a, ptr %b, double %v, double %dv) {
  %r1 = call double (...) @__enzyme_fwddiff(ptr nonnull @foo, ptr %a, ptr %b, double %v, double %dv)
  ret double %r1
}

declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffefoo(ptr %p, ptr %"p'", double %vf, double %"vf'")
; CHECK-NEXT:   %"v'ipc" = bitcast double %"vf'" to i64
; CHECK-NEXT:   %v = bitcast double %vf to i64
; CHECK-NEXT:   %1 = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %"p'", ptr nonnull @fadd_op, i8 5, i8 1, i64 %"v'ipc")
; CHECK-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
; CHECK-NEXT:   %"new'ipev" = extractvalue { i64, i64 } %1, 1
; CHECK-NEXT:   %"newf'ipc" = bitcast i64 %"new'ipev" to double
; CHECK-NEXT:   ret double %"newf'ipc"
; CHECK-NEXT: }
