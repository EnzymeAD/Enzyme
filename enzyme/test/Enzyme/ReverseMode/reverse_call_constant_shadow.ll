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
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { float, ptr } @augmented_sub(ptr %x, ptr %"x'")
; CHECK-NEXT:   %subcache = extractvalue { float, ptr } %call_augmented, 0
; CHECK-NEXT:   %"call'ac" = extractvalue { float, ptr } %call_augmented, 1
; CHECK-NEXT:   %0 = load float, ptr %"call'ac", align 4
; CHECK-NEXT:   %1 = fadd fast float %0, %differeturn
; CHECK-NEXT:   store float %1, ptr %"call'ac", align 4
; CHECK-NEXT:   call void @diffesub(ptr %x, ptr %"x'", float %subcache)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal float @augmented_square_into(ptr {{.*}}%dst, ptr {{.*}}%src, ptr {{.*}}%"src'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v = load float, ptr %src, align 4
; CHECK-NEXT:   %d = fmul float %v, %v
; CHECK-NEXT:   store float %d, ptr %dst, align 4
; CHECK-NEXT:   ret float %v
; CHECK-NEXT: }

; CHECK: define internal { float, ptr } @augmented_sub(ptr %x, ptr %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { float, ptr }, align 8
; CHECK-NEXT:   %malloccall = alloca float, i64 1, align 4
; CHECK-NEXT:   %_augmented = call fast float @augmented_square_into(ptr %malloccall, ptr %x, ptr %"x'")
; CHECK-NEXT:   store float %_augmented, ptr %0, align 4
; CHECK-NEXT:   %1 = getelementptr inbounds { float, ptr }, ptr %0, i32 0, i32 1
; CHECK-NEXT:   store ptr %"x'", ptr %1, align 8
; CHECK-NEXT:   %2 = load { float, ptr }, ptr %0, align 8
; CHECK-NEXT:   ret { float, ptr } %2
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(ptr %x, ptr %"x'", float %tapeArg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @diffesquare_into(ptr {{(undef|poison)}}, ptr %x, ptr %"x'", float %tapeArg1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesquare_into(ptr {{.*}}%dst, ptr {{.*}}%src, ptr {{.*}}%"src'", float %v)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load float, ptr %"src'", align 4
; CHECK-NEXT:   store float %0, ptr %"src'", align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffeh(ptr %x, ptr %"x'", float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call ptr @augmented_id(ptr %x, ptr %"x'")
; CHECK-NEXT:   %0 = load float, ptr %call_augmented, align 4
; CHECK-NEXT:   %1 = fadd fast float %0, %differeturn
; CHECK-NEXT:   store float %1, ptr %call_augmented, align 4
; CHECK-NEXT:   call void @diffeid(ptr %x, ptr {{(undef|poison)}})
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal ptr @augmented_id(ptr %x, ptr %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret ptr %"x'"
; CHECK-NEXT: }

; CHECK: define internal void @diffeid(ptr %x, ptr %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
