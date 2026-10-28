; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; A memset over a float and the 4 bytes of padding after it (typed Anything). The float
; part must be zeroed in the shadow in the reverse pass, since the memset overwrites the
; float. Merged, Float | Anything is Anything, which only gets the integer treatment (the
; shadow is set in the forward pass), and the cotangent of the load below leaked into %x.

declare void @__enzyme_autodiff(...)

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define float @clear(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Anything, [-1,5]:Anything, [-1,6]:Anything, [-1,7]:Anything}" %p, float %x) {
entry:
  store float %x, ptr %p, align 8
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 8, i1 false)
  %r = load float, ptr %p, align 8
  ret float %r
}

define void @dclear(ptr %p, ptr %dp, float %x) {
entry:
  call void (...) @__enzyme_autodiff(ptr @clear, metadata !"enzyme_dup", ptr %p, ptr %dp, float %x)
  ret void
}

; CHECK: define internal { float } @diffeclear(ptr {{.*}}%p, ptr {{.*}}%"p'", float %x, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   store float %x, ptr %p, align 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %[[pad:.+]] = getelementptr inbounds i8, ptr %"p'", i32 4
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr %[[pad]], i8 0, i64 4, i1 false)
; CHECK-NEXT:   %[[l:.+]] = load float, ptr %"p'", align 8
; CHECK-NEXT:   %[[a:.+]] = fadd fast float %[[l]], %differeturn
; CHECK-NEXT:   store float %[[a]], ptr %"p'", align 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr %"p'", i8 0, i64 4, i1 false)
; CHECK-NEXT:   %[[dx:.+]] = load float, ptr %"p'", align 8
; CHECK-NEXT:   store float 0.000000e+00, ptr %"p'", align 8
; CHECK-NEXT:   %[[r:.+]] = insertvalue { float } undef, float %[[dx]], 0
; CHECK-NEXT:   ret { float } %[[r]]
