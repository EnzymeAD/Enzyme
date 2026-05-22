; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=0 -passes="enzyme,function(mem2reg,instsimplify,early-cse,%simplifycfg)" -S -enzyme-detect-readthrow=0 | FileCheck %s

; Regression for a SIGSEGV in the reverse-mode Store adjoint when the
; store target is a pointer LOADED from a duplicated-pointer-of-pointer
; argument. In that case the shadow of the store target is itself loaded
; from the shadow tape, and can be NULL on paths where Enzyme failed to
; instantiate a shadow allocation in the forward pass. Before the fix,
; visitCommonStore unconditionally loaded from the (possibly NULL) shadow
; pointer and SIGSEGVd at runtime. The fix wraps the
; load / setPtrDiffe(0) / addToDiffe triple in a runtime
; `if (shadow_ptr != null)` diamond -- the BBs named <bb>_nnactive /
; <bb>_nnmerge -- whenever the store target is not syntactically known
; to be non-null (here: a Load, which is exactly the crash pattern).

declare double @__enzyme_autodiff(i8*, ...)

define internal void @inner_ptr(ptr %indirect, ptr %xptr) {
entry:
  %dst = load ptr, ptr %indirect, align 8
  %x = load double, ptr %xptr, align 8
  %y = fmul fast double %x, %x
  store double %y, ptr %dst, align 8
  ret void
}

define void @caller(ptr %indirect, ptr %dindirect, ptr %xptr, ptr %dxptr) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(
      i8* bitcast (void (ptr, ptr)* @inner_ptr to i8*),
      metadata !"enzyme_dup", ptr %indirect, ptr %dindirect,
      metadata !"enzyme_dup", ptr %xptr, ptr %dxptr)
  ret void
}

; The reverse function for @inner_ptr must materialize the NULL-shadow
; guard diamond around the Store adjoint, because the store target %dst
; is a Load (its shadow is loaded from the tape and can be NULL).
; CHECK: define internal {{.*}} @diffeinner_ptr(
; CHECK-DAG: _nnactive
; CHECK-DAG: _nnmerge
