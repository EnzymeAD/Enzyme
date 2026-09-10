; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; Julia 1.13's atomic modify pseudo-intrinsic
;   {old, new} = julia.atomicmodify.iN.pAS(ptr, op, ordering, syncscope, args...)
; atomically performs old = *ptr; new = op(old, args...); *ptr = new.
; The augmented primal of a split forward derivative runs in ReverseModePrimal,
; so it must replay the primal for a recognized op with an active result, and
; it must not repeat a shadow modification that the tangent pass also emits.

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

; atomic { x += v } on float data: the augmented primal only replays the
; original call, the tangent applies the same op to the shadow location.
define double @foo(ptr %p, double %vf) {
  %v = bitcast double %vf to i64
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
  %new = extractvalue { i64, i64 } %on, 1
  %newf = bitcast i64 %new to double
  ret double %newf
}

; atomic counter increment within duplicated memory: the shadow location is
; incremented once, by the augmented primal, and not again by the tangent.
define double @bar(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer}" %p, double %x) {
  %c = getelementptr inbounds i8, ptr %p, i64 8
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %c, ptr nonnull @iadd_op, i8 7, i8 1, i64 1)
  %d = load double, ptr %p, align 8
  %m = fmul double %d, %x
  ret double %m
}

define double @caller(ptr %a, ptr %b, double %v, double %dv) {
  %r1 = call double (...) @__enzyme_fwdsplit(ptr nonnull @foo, ptr %a, ptr %b, double %v, double %dv, ptr null)
  %r2 = call double (...) @__enzyme_fwdsplit(ptr nonnull @bar, ptr %a, ptr %b, double %v, double %dv, ptr null)
  %fr = fadd double %r1, %r2
  ret double %fr
}

declare double @__enzyme_fwdsplit(...)

; CHECK: define internal ptr @augmented_foo(ptr %p, ptr %"p'", double %vf, double %"vf'")
; CHECK-NEXT:   %v = bitcast double %vf to i64
; CHECK-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
; CHECK-NEXT:   ret ptr null
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffefoo(ptr %p, ptr %"p'", double %vf, double %"vf'", ptr %tapeArg)
; CHECK-NEXT:   %"v'ipc" = bitcast double %"vf'" to i64
; CHECK-NEXT:   %1 = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %"p'", ptr nonnull @fadd_op, i8 5, i8 1, i64 %"v'ipc")
; CHECK-NEXT:   %"new'ipev" = extractvalue { i64, i64 } %1, 1
; CHECK-NEXT:   %"newf'ipc" = bitcast i64 %"new'ipev" to double
; CHECK-NEXT:   ret double %"newf'ipc"
; CHECK-NEXT: }

; CHECK: define internal ptr @augmented_bar(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer}" %p, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer}" %"p'", double %x, double %"x'")
; CHECK-NEXT:   %tapemem = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) ptr @malloc(i64 8)
; CHECK-NEXT:   %"c'ipg" = getelementptr inbounds i8, ptr %"p'", i64 8
; CHECK-NEXT:   %c = getelementptr inbounds i8, ptr %p, i64 8
; CHECK-NEXT:   %1 = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %"c'ipg", ptr nonnull @iadd_op, i8 7, i8 1, i64 1)
; CHECK-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %c, ptr nonnull @iadd_op, i8 7, i8 1, i64 1)
; CHECK-NEXT:   %d = load double, ptr %p, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   store double %d, ptr %tapemem, align 8
; CHECK-NEXT:   ret ptr %tapemem
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffebar(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer}" %p, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer}" %"p'", double %x, double %"x'", ptr %tapeArg)
; CHECK-NEXT:   %d = load double, ptr %tapeArg, align 8, !enzyme_mustcache !5
; CHECK-NEXT:   tail call void @free(ptr nonnull %tapeArg)
; CHECK-NEXT:   %"d'ipl" = load double, ptr %"p'", align 8, !alias.scope !6, !noalias !9
; CHECK-NEXT:   %1 = fmul fast double %"d'ipl", %x
; CHECK-NEXT:   %2 = fmul fast double %"x'", %d
; CHECK-NEXT:   %3 = fadd fast double %1, %2
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }
