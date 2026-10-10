; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify)" -S | FileCheck %s; fi

; @sub only writes a temporary of its own, so it is read-only-or-throw and the
; call to it in @f is a constant instruction. Its result is active, so it is
; still differentiated. The reverse of @sub propagates the copy into the
; temporary back into the shadow of %x (adding zero), so its reverse call must
; be passed that shadow rather than undef.
;
; @id's reverse does nothing, so its reverse call may still be passed undef.

define float @f(ptr %x) {
entry:
  %call = tail call ptr @sub(ptr %x)
  %res = load float, ptr %call, align 4
  ret float %res
}

; Fully read-only-or-throw: it only writes a temporary of its own.
define ptr @sub(ptr %x) "enzyme_ReadOnlyOrThrow" {
entry:
  %tmp = alloca float, align 4
  call void @square_into(ptr %tmp, ptr %x)
  ret ptr %x
}

define void @square_into(ptr %dst, ptr %src) noinline {
entry:
  %v = load float, ptr %src, align 4
  %d = fmul float %v, %v
  store float %d, ptr %dst, align 4
  ret void
}

define float @g(ptr %x, ptr %dx) {
entry:
  %0 = tail call float (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx)
  ret float %0
}

declare float @__enzyme_autodiff(ptr, ...)

define ptr @id(ptr %x) "enzyme_ReadOnlyOrThrow" {
entry:
  ret ptr %x
}

define float @h(ptr %x) {
entry:
  %call = tail call ptr @id(ptr %x)
  %res = load float, ptr %call, align 4
  ret float %res
}

define float @dh(ptr %x, ptr %dx) {
entry:
  %0 = tail call float (ptr, ...) @__enzyme_autodiff(ptr @h, ptr %x, ptr %dx)
  ret float %0
}

; CHECK: define internal void @diffef(ptr %x, ptr %"x'", float %differeturn)
; CHECK:   call void @diffesub(ptr %x, ptr %"x'", float %subcache)

; CHECK: define internal void @diffesub(ptr %x, ptr %"x'", float %tapeArg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @diffesquare_into(ptr {{(undef|poison)}}, ptr %x, ptr %"x'", float %tapeArg1)

; CHECK: define internal void @diffeh(ptr %x, ptr %"x'", float %differeturn)
; CHECK:   call void @diffeid(ptr %x, ptr {{(undef|poison)}})
