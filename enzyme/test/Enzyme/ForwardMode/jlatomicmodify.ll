; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; Julia 1.13's atomic modify pseudo-intrinsic
;   {old, new} = julia.atomicmodify.iN.pAS(ptr, op, ordering, syncscope, args...)
; atomically performs old = *ptr; new = op(old, args...); *ptr = new.
; Since op is linear for fadd, the tangent is computed by applying the same op
; to the shadow location and the shadow value.

declare { i64, i64 } @julia.atomicmodify.i64.p0(ptr, ptr, i8, i8, ...)
declare { ptr, ptr } @julia.atomicmodify.p0.p0(ptr, ptr, i8, i8, ...)

define internal i64 @fadd_op(i64 %old, i64 %v) {
  %oldf = bitcast i64 %old to double
  %vf = bitcast i64 %v to double
  %r = fadd double %oldf, %vf
  %ri = bitcast double %r to i64
  ret i64 %ri
}

; op may take the forwarded argument in a type other than the element type of
; the result struct; a missing shadow has to be zero of the *argument* type.
define internal i64 @fadd_op_dbl(i64 %old, double %vf) {
  %oldf = bitcast i64 %old to double
  %r = fadd double %oldf, %vf
  %ri = bitcast double %r to i64
  ret i64 %ri
}

define internal i64 @fsub_op(i64 %old, i64 %v) {
  %oldf = bitcast i64 %old to double
  %vf = bitcast i64 %v to double
  %r = fsub double %oldf, %vf
  %ri = bitcast double %r to i64
  ret i64 %ri
}

define internal ptr @xchg_op(ptr %old, ptr %v) {
  ret ptr %v
}

define double @foo(ptr %p, double %vf) {
  %v = bitcast double %vf to i64
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
  %new = extractvalue { i64, i64 } %on, 1
  %newf = bitcast i64 %new to double
  ret double %newf
}

define double @baz(ptr %p, double %vf) {
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op_dbl, i8 5, i8 1, double %vf)
  %new = extractvalue { i64, i64 } %on, 1
  %newf = bitcast i64 %new to double
  ret double %newf
}

; an active value modifying inactive memory: the tangent is {0, dv}
define double @constmem(ptr %p, double %vf) {
  %v = bitcast double %vf to i64
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
  %new = extractvalue { i64, i64 } %on, 1
  %newf = bitcast i64 %new to double
  ret double %newf
}

; and {0, -dv} for a subtraction
define double @constmemsub(ptr %p, double %vf) {
  %v = bitcast double %vf to i64
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fsub_op, i8 5, i8 1, i64 %v)
  %new = extractvalue { i64, i64 } %on, 1
  %newf = bitcast i64 %new to double
  ret double %newf
}

; an inactive pointer exchanged into duplicated memory is written to the shadow
; location as is (like an inactive store of a pointer), not replaced by null
define double @xchgp(ptr %p, ptr %q) {
  %on = call { ptr, ptr } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.p0.p0(ptr align 8 %p, ptr nonnull @xchg_op, i8 5, i8 1, ptr %q)
  %old = extractvalue { ptr, ptr } %on, 0
  %d = load double, ptr %old, align 8
  ret double %d
}

define double @caller(ptr %a, ptr %b, ptr %q, double %v, double %dv) {
  %r1 = call double (...) @__enzyme_fwddiff(ptr nonnull @foo, ptr %a, ptr %b, double %v, double %dv)
  %r2 = call double (...) @__enzyme_fwddiff(ptr nonnull @baz, ptr %a, ptr %b, metadata !"enzyme_const", double %v)
  %r3 = call double (...) @__enzyme_fwddiff(ptr nonnull @constmem, metadata !"enzyme_const", ptr %a, double %v, double %dv)
  %r4 = call double (...) @__enzyme_fwddiff(ptr nonnull @constmemsub, metadata !"enzyme_const", ptr %a, double %v, double %dv)
  %r5 = call double (...) @__enzyme_fwddiff(ptr nonnull @xchgp, ptr %a, ptr %b, metadata !"enzyme_const", ptr %q)
  %f1 = fadd double %r1, %r2
  %f2 = fadd double %f1, %r3
  %f3 = fadd double %f2, %r4
  %f4 = fadd double %f3, %r5
  ret double %f4
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

; CHECK: define internal double @fwddiffebaz(ptr %p, ptr %"p'", double %vf)
; CHECK-NEXT:   %1 = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %"p'", ptr nonnull @fadd_op_dbl, i8 5, i8 1, double 0.000000e+00)
; CHECK-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op_dbl, i8 5, i8 1, double %vf)
; CHECK-NEXT:   %"new'ipev" = extractvalue { i64, i64 } %1, 1
; CHECK-NEXT:   %"newf'ipc" = bitcast i64 %"new'ipev" to double
; CHECK-NEXT:   ret double %"newf'ipc"
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffeconstmem(ptr %p, double %vf, double %"vf'")
; CHECK-NEXT:   %v = bitcast double %vf to i64
; CHECK-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
; CHECK-NEXT:   ret double %"vf'"
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffeconstmemsub(ptr %p, double %vf, double %"vf'")
; CHECK-NEXT:   %v = bitcast double %vf to i64
; CHECK-NEXT:   %1 = fneg fast double %"vf'"
; CHECK-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fsub_op, i8 5, i8 1, i64 %v)
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffexchgp(ptr %p, ptr %"p'", ptr %q)
; CHECK-NEXT:   %1 = call { ptr, ptr } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.p0.p0(ptr align 8 %"p'", ptr nonnull @xchg_op, i8 5, i8 1, ptr %q)
; CHECK-NEXT:   %on = call { ptr, ptr } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.p0.p0(ptr align 8 %p, ptr nonnull @xchg_op, i8 5, i8 1, ptr %q)
; CHECK-NEXT:   %"old'ipev" = extractvalue { ptr, ptr } %1, 0
; CHECK-NEXT:   %"d'ipl" = load double, ptr %"old'ipev", align 8
; CHECK-NEXT:   ret double %"d'ipl"
; CHECK-NEXT: }
