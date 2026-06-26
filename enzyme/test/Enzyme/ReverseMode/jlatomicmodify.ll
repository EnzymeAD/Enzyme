; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Julia 1.13's atomic modify pseudo-intrinsic
;   {old, new} = julia.atomicmodify.iN.pAS(ptr, op, ordering, syncscope, args...)
; atomically performs old = *ptr; new = op(old, args...); *ptr = new.

declare { i64, i64 } @julia.atomicmodify.i64.p0(ptr, ptr, i8, i8, ...)

define internal i64 @fadd_op(i64 %old, i64 %v) {
  %oldf = bitcast i64 %old to double
  %vf = bitcast i64 %v to double
  %r = fadd double %oldf, %vf
  %ri = bitcast double %r to i64
  ret i64 %ri
}

define internal i64 @iadd_op(i64 %old, i64 %v) {
  %r = add i64 %old, %v
  ret i64 %r
}

; atomic { x += v } on float data: the adjoint of v is read from the shadow
; location (with the ordering downgraded for the reverse pass).
define void @foo(ptr %p, double %vf) {
  %v = bitcast double %vf to i64
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
  ret void
}

; atomic counter increment within duplicated memory: the modification is
; replicated on the shadow location.
define double @bar(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer}" %p, double %x) {
  %c = getelementptr inbounds i8, ptr %p, i64 8
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %c, ptr nonnull @iadd_op, i8 7, i8 1, i64 1)
  %d = load double, ptr %p, align 8
  %m = fmul double %d, %x
  ret double %m
}

define double @caller(ptr %a, ptr %b, double %v) {
  %r1 = call double (...) @__enzyme_autodiff(ptr nonnull @foo, ptr %a, ptr %b, double %v)
  %r2 = call double (...) @__enzyme_autodiff(ptr nonnull @bar, ptr %a, ptr %b, double %v)
  %fr = fadd double %r1, %r2
  ret double %fr
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffefoo(ptr %p, ptr %"p'", double %vf)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %v = bitcast double %vf to i64
; CHECK-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
; CHECK-NEXT:   %0 = load atomic i64, ptr %"p'" monotonic, align 8
; CHECK-NEXT:   %1 = bitcast i64 %0 to double
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }

; CHECK: define internal { double } @diffebar(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer}" %p, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer}" %"p'", double %x, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %"c'ipg" = getelementptr inbounds i8, ptr %"p'", i64 8
; CHECK-NEXT:   %c = getelementptr inbounds i8, ptr %p, i64 8
; CHECK-NEXT:   %0 = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %"c'ipg", ptr nonnull @iadd_op, i8 7, i8 1, i64 1)
; CHECK-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %c, ptr nonnull @iadd_op, i8 7, i8 1, i64 1)
; CHECK-NEXT:   %d = load double, ptr %p, align 8
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %x
; CHECK-NEXT:   %2 = fmul fast double %differeturn, %d
; CHECK-NEXT:   %3 = load double, ptr %"p'", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %1
; CHECK-NEXT:   store double %4, ptr %"p'", align 8
; CHECK-NEXT:   %5 = insertvalue { double } undef, double %2, 0
; CHECK-NEXT:   ret { double } %5
; CHECK-NEXT: }
