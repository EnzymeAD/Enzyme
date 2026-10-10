; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify)" -S | FileCheck %s; fi

; Like retptr.ll, but @sub also copies active data it loads into memory of its
; own. It is still fully read-only-or-throw, as that memory does not outlive
; it, so the call is a constant instruction. Its result is active, though, so
; the call is still differentiated, and its reverse call must be passed the
; shadow of %this, as its augmented call is, rather than undef: the reverse of
; @sub may use it.

define float @f(ptr %this) {
entry:
  %call = tail call ptr @sub(ptr %this)
  %res = load float, ptr %call, align 4
  ret float %res
}

define ptr @sub(ptr %this) {
entry:
  %tmp = alloca float, align 4
  %0 = load ptr, ptr %this, align 8
  %v = load float, ptr %0, align 4
  store volatile float %v, ptr %tmp, align 4
  ret ptr %0
}

define float @g(ptr %this, ptr %dthis) {
entry:
  %0 = tail call float (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %this, ptr %dthis)
  ret float %0
}

declare float @__enzyme_autodiff(ptr, ...)

; CHECK: define internal void @diffef(ptr nocapture readonly %this, ptr nocapture %"this'", float %differeturn)
; CHECK: call void @diffesub(ptr %this, ptr %"this'")
