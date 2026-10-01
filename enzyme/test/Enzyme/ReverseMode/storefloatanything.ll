; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; The pointee is a float everywhere ([-1,-1]) except for 4 bytes typed Anything, a legal
; type tree. A range of the store starts at byte 4, while dt starts from [-1] (a float): the
; check that ends a range where a float meets Anything must not fire at the first byte of a
; range, or the range is empty and the store is never done differentiating.

declare void @__enzyme_autodiff(...)

define void @copy(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@float, [-1,4]:Anything, [-1,5]:Anything, [-1,6]:Anything, [-1,7]:Anything}" %dst, ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@float, [-1,4]:Anything, [-1,5]:Anything, [-1,6]:Anything, [-1,7]:Anything}" %src) {
entry:
  %v = load i64, ptr %src, align 8
  store i64 %v, ptr %dst, align 8
  ret void
}

define void @dcopy(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc) {
entry:
  call void (...) @__enzyme_autodiff(ptr @copy, metadata !"enzyme_dup", ptr %dst, ptr %ddst, metadata !"enzyme_dup", ptr %src, ptr %dsrc)
  ret void
}

; CHECK: define internal void @diffecopy(ptr {{.*}}%dst, ptr {{.*}}%"dst'", ptr {{.*}}%src, ptr {{.*}}%"src'")
; CHECK:        store i64 %v, ptr %dst, align 8
; CHECK-NEXT:   %[[ddst:.+]] = load i64, ptr %"dst'", align 8
; CHECK:        store i32 %{{.+}}, ptr %"dst'", align 8
; CHECK:        %[[dv:.+]] = load float, ptr %cast.alloca
; CHECK-NEXT:   %[[old:.+]] = load float, ptr %"src'", align 8
; CHECK-NEXT:   %[[new:.+]] = fadd fast float %[[old]], %[[dv]]
; CHECK-NEXT:   store float %[[new]], ptr %"src'", align 8
; CHECK-NEXT:   ret void
