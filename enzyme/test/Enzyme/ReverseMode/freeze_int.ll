; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -enzyme-strict-aliasing=0 -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-strict-aliasing=0 -S | FileCheck %s

; Reduced from https://github.com/EnzymeAD/Enzyme/issues/3757.
; Without strict aliasing, an integer field loaded from a duplicated struct
; pointer and passed through `freeze` has `isConstantInstruction == true` and
; `isConstantValue == false` (potential pointer). Reverse mode must not emit an
; adjoint accumulation with `addingType = i64` on the `freeze`, and active
; non-float `freeze`s (bitcast float or mixed aggregate) must deduce their
; float adding type per slice.

declare void @__enzyme_autodiff(...)
declare double @__enzyme_autodiff_val(...)

define internal fastcc void @seg(ptr noalias align 8 %out, ptr noalias readonly align 8 %q, ptr noalias readonly align 8 %env) {
entry:
  %n_raw = load i64, ptr %q, align 8
  %n = freeze i64 %n_raw
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %exit, label %preheader

preheader:
  %x_ptr = getelementptr inbounds i8, ptr %q, i64 8
  %x = load double, ptr %x_ptr, align 8
  br label %loop

loop:
  %iv = phi i64 [ 0, %preheader ], [ %iv.next, %loop ]
  %idx = add nuw nsw i64 %iv, %n
  %ep = getelementptr inbounds double, ptr %env, i64 %idx
  %ev = load double, ptr %ep, align 8
  %prod = fmul double %ev, %x
  %cur = load double, ptr %out, align 8
  %sum = fadd double %cur, %prod
  store double %sum, ptr %out, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @f(ptr noalias align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %out,
               ptr noalias readonly align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}" %bas,
               i64 "enzyme_type"="{[0]:Integer}" %bas_len,
               ptr noalias readonly align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %env) {
entry:
  %q = alloca [16 x i8], align 8
  %empty = icmp eq i64 %bas_len, 0
  br i1 %empty, label %abort, label %body

body:
  %n = load i64, ptr %bas, align 8
  %x = load double, ptr %env, align 8
  store i64 %n, ptr %q, align 8
  %q_x = getelementptr inbounds i8, ptr %q, i64 8
  store double %x, ptr %q_x, align 8
  call fastcc void @seg(ptr %out, ptr %q, ptr %env)
  ret void

abort:
  unreachable
}

define double @bitcast_freeze(double %x) {
  %i = bitcast double %x to i64
  %f = freeze i64 %i
  %y = bitcast i64 %f to double
  %r = fmul double %y, %y
  ret double %r
}

define double @struct_freeze(double %x, i64 %n) {
  %s0 = insertvalue { i64, double } poison, i64 %n, 0
  %s1 = insertvalue { i64, double } %s0, double %x, 1
  %sf = freeze { i64, double } %s1
  %k = extractvalue { i64, double } %sf, 0
  %v = extractvalue { i64, double } %sf, 1
  %kf = sitofp i64 %k to double
  %r = fmul double %v, %kf
  ret double %r
}

define double @caller(ptr %out, ptr %dout, ptr %bas, i64 %bas_len, ptr %env, ptr %denv, double %x) {
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %bas, metadata !"enzyme_const", i64 %bas_len, metadata !"enzyme_dup", ptr %env, ptr %denv)
  %d1 = call double (...) @__enzyme_autodiff_val(ptr @bitcast_freeze, double %x)
  %d2 = call double (...) @__enzyme_autodiff_val(ptr @struct_freeze, double %x, metadata !"enzyme_const", i64 %bas_len)
  %s = fadd double %d1, %d2
  ret double %s
}

; CHECK-LABEL: define internal fastcc void @diffeseg(
; CHECK: invertentry:
; CHECK-NOT: freeze i64 %
; CHECK: ret void

; CHECK-LABEL: define internal { double } @diffebitcast_freeze(double %x, double %differeturn)
; CHECK: freeze i64

; CHECK-LABEL: define internal { double } @diffestruct_freeze(double %x, i64 %n, double %differeturn)
; CHECK: freeze { i64, double }
