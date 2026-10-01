; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Regression for wasm32 tape-cache malloc sizing.
;
; On 32-bit targets the C ABI's malloc takes size_t = i32. wasi-libc on
; wasm32-wasi declares `void *malloc(i32)`; if Enzyme's tape-cache math
; widens the size operand to i64, we emit `call ptr @malloc(i64 ...)`
; against the i32-parameter declaration and the module fails validation
; ("signature_mismatch:malloc") at load. CreateAllocation truncates
; Count to the target IntPtrTy whenever Count is wider, so this test
; asserts that on wasm32 every Enzyme-emitted cache malloc receives an
; i32 size operand.
;
; The primal is a two-level accumulation ((A*x - y) squared, summed over
; rows) reduced from the mean-squared-error reproducer in the patch
; description. The inner accumulator's LCSSA lift into the outer loop
; body forces per-outer-iteration caching, which is what triggers the
; problematic CreateAllocation path.

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-wasi"

define double @loss(double* nocapture readonly %A, double* nocapture readonly %x,
                    double* nocapture readonly %y, i64 %n) {
entry:
  br label %outer.loop

outer.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.exit ]
  %sum = phi double [ 0.0, %entry ], [ %next.sum, %outer.exit ]
  br label %inner.loop

inner.loop:
  %j = phi i64 [ 0, %outer.loop ], [ %j.next, %inner.loop ]
  %acc = phi double [ 0.0, %outer.loop ], [ %next.acc, %inner.loop ]
  %row = mul i64 %i, %n
  %ij = add i64 %row, %j
  %a.p = getelementptr inbounds double, double* %A, i64 %ij
  %a.v = load double, double* %a.p, align 8
  %x.p = getelementptr inbounds double, double* %x, i64 %j
  %x.v = load double, double* %x.p, align 8
  %ax = fmul double %a.v, %x.v
  %next.acc = fadd double %acc, %ax
  %j.next = add i64 %j, 1
  %inner.done = icmp eq i64 %j.next, %n
  br i1 %inner.done, label %outer.exit, label %inner.loop

outer.exit:
  %y.p = getelementptr inbounds double, double* %y, i64 %i
  %y.v = load double, double* %y.p, align 8
  %r = fsub double %next.acc, %y.v
  %r2 = fmul double %r, %r
  %next.sum = fadd double %sum, %r2
  %i.next = add i64 %i, 1
  %outer.done = icmp eq i64 %i.next, %n
  br i1 %outer.done, label %ret, label %outer.loop

ret:
  ret double %next.sum
}

define double @dloss(double* %A, double* %dA, double* %x, double* %dx,
                     double* %y, i64 %n) {
entry:
  %r = call double (i8*, ...) @__enzyme_autodiff(
    i8* bitcast (double (double*, double*, double*, i64)* @loss to i8*),
    metadata !"enzyme_dup", double* %A, double* %dA,
    metadata !"enzyme_dup", double* %x, double* %dx,
    metadata !"enzyme_const", double* %y,
    metadata !"enzyme_const", i64 %n)
  ret double %r
}

declare double @__enzyme_autodiff(i8*, ...)

; The emitted cache malloc must use the target IntPtrTy (i32 on wasm32),
; NOT the pre-cast i64 loop-bound-derived Count. Enzyme should insert
; a `trunc i64 ... to i32` and feed the i32 size into @malloc.

; CHECK: trunc i64 {{.*}} to i32
; CHECK: @malloc(i32
; CHECK-NOT: @malloc(i64
; CHECK: declare {{.*}} @malloc(i32)
