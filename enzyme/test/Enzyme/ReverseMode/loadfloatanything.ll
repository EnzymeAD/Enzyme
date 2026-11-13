; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A float followed by 4 bytes of padding (typed Anything), loaded as one i64, the way
; Julia copies a Float32 field together with its padding. The load's range [0, 8) must
; be split into the float [0, 4) and the padding [4, 8): merged, Float | Anything is
; Anything, which is not differentiated, and the float's derivative was dropped.

declare void @__enzyme_autodiff(...)

define float @copyfield(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Anything, [-1,5]:Anything, [-1,6]:Anything, [-1,7]:Anything}" %src) {
entry:
  %tmp = alloca i64, align 8
  %v = load i64, ptr %src, align 8
  store i64 %v, ptr %tmp, align 8
  %f = load float, ptr %tmp, align 8
  ret float %f
}

define void @dcopyfield(ptr %src, ptr %dsrc) {
entry:
  call void (...) @__enzyme_autodiff(ptr @copyfield, metadata !"enzyme_dup", ptr %src, ptr %dsrc)
  ret void
}

; CHECK: define internal void @diffecopyfield(ptr {{.*}}%src, ptr {{.*}}%"src'", float %differeturn)
; CHECK:        %[[dv:.+]] = load float, ptr %cast.alloca
; CHECK-NEXT:   %[[old:.+]] = load float, ptr %"src'", align 8
; CHECK-NEXT:   %[[new:.+]] = fadd fast float %[[old]], %[[dv]]
; CHECK-NEXT:   store float %[[new]], ptr %"src'", align 8
; CHECK-NEXT:   ret void
