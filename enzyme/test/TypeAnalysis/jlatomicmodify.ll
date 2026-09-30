; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s --check-prefix=CALLER; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=fromload -S -o /dev/null | FileCheck %s --check-prefix=FROMLOAD; fi

; Julia 1.13's atomic modify pseudo-intrinsic
;   {old, new} = julia.atomicmodify.iN.pAS(ptr, op, ordering, syncscope, args...)
; atomically performs old = *ptr; new = op(old, args...); *ptr = new.
; Both result elements have the type of the modified memory, and op is
; analyzed interprocedurally to relate the forwarded arguments with it.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare { i64, i64 } @julia.atomicmodify.i64.p0(ptr, ptr, i8, i8, ...)

define internal i64 @fadd_op(i64 %old, i64 %v) {
  %oldf = bitcast i64 %old to double
  %vf = bitcast i64 %v to double
  %r = fadd double %oldf, %vf
  %ri = bitcast double %r to i64
  ret i64 %ri
}

; the modified memory and the results are floating point since the argument is
define double @caller(ptr %p, double %vf) {
  %v = bitcast double %vf to i64
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
  %old = extractvalue { i64, i64 } %on, 0
  %oldf = bitcast i64 %old to double
  ret double %oldf
}

; the argument (and the memory it is loaded from) is floating point since op
; adds it to the modified value as a double
define void @fromload(ptr %p, ptr %q) {
  %v = load i64, ptr %q, align 8
  %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v)
  ret void
}

; CALLER: caller - {[-1]:Float@double} |{[-1]:Pointer}:{} {[-1]:Float@double}:{}
; CALLER-NEXT: ptr %p: {[-1]:Pointer, [-1,0]:Float@double}
; CALLER-NEXT: double %vf: {[-1]:Float@double}
; CALLER-EMPTY:
; CALLER-NEXT:   %v = bitcast double %vf to i64: {[-1]:Float@double}
; CALLER-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v): {[0]:Float@double, [8]:Float@double}
; CALLER-NEXT:   %old = extractvalue { i64, i64 } %on, 0: {[-1]:Float@double}
; CALLER-NEXT:   %oldf = bitcast i64 %old to double: {[-1]:Float@double}
; CALLER-NEXT:   ret double %oldf: {}

; FROMLOAD: fromload - {} |{[-1]:Pointer}:{} {[-1]:Pointer}:{}
; FROMLOAD-NEXT: ptr %p: {[-1]:Pointer, [-1,0]:Float@double}
; FROMLOAD-NEXT: ptr %q: {[-1]:Pointer, [-1,0]:Float@double}
; FROMLOAD-EMPTY:
; FROMLOAD-NEXT:   %v = load i64, ptr %q, align 8: {[-1]:Float@double}
; FROMLOAD-NEXT:   %on = call { i64, i64 } (ptr, ptr, i8, i8, ...) @julia.atomicmodify.i64.p0(ptr align 8 %p, ptr nonnull @fadd_op, i8 5, i8 1, i64 %v): {[0]:Float@double, [8]:Float@double}
; FROMLOAD-NEXT:   ret void: {}
