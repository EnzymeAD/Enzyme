; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; A global that no call in the function touches gets a local, zeroed alloca as
; its shadow. Constant GEPs into the global then shadow to GEP instructions on
; that alloca, e.g. the members of a Fortran COMMON block that has no shadow
; block (flang addresses members as byte offsets into an [N x i8] global).

@state = common global [8 x i8] zeroinitializer, align 4

define float @f(float %x) {
entry:
  %a = load float, ptr @state, align 4
  %m = fmul float %a, %x
  store float %m, ptr getelementptr inbounds (i8, ptr @state, i64 4), align 4
  %b = load float, ptr getelementptr inbounds (i8, ptr @state, i64 4), align 4
  ret float %b
}

declare float @__enzyme_fwddiff(ptr, ...)

define float @df(float %x) {
entry:
  %r = call float (ptr, ...) @__enzyme_fwddiff(ptr @f, float %x, float 1.0)
  ret float %r
}

; CHECK: define internal float @fwddiffef(float %x, float %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"state'ipa" = alloca [8 x i8], align 4
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull align 4 %"state'ipa", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"'ipg" = getelementptr inbounds i8, ptr %"state'ipa", i64 4
; CHECK-NEXT:   %"a'ipl" = load float, ptr %"state'ipa", align 4
; CHECK:        store float %{{.*}}, ptr %"'ipg", align 4
; CHECK:        %"b'ipl" = load float, ptr %"'ipg", align 4
; CHECK-NEXT:   ret float %"b'ipl"
