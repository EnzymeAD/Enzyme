; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; The load's type tree has Anything bytes (struct padding) next to a float:
;   {[0]:Float@float, [4-7]:Anything, [8]:Float@double}
; The adjoint of the double at offset 8 must still be propagated into the
; shadow of %p, rather than being absorbed into an Anything run.

declare void @__enzyme_autodiff(...)

define double @f(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Anything, [-1,5]:Anything, [-1,6]:Anything, [-1,7]:Anything, [-1,8]:Float@double}" %p) {
entry:
  %agg = load { float, double }, ptr %p, align 8
  %w = extractvalue { float, double } %agg, 1
  ret double %w
}

define void @test(ptr %p, ptr %dp) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %p, ptr %dp)
  ret void
}

; CHECK: define internal void @diffef(ptr {{.*}}%p, ptr {{.*}}%"p'", double %differeturn)
; CHECK: %[[fold:.+]] = load float, ptr %"p'", align 8
; CHECK-NEXT: %[[fnew:.+]] = fadd fast float %[[fold]], %cast.load
; CHECK-NEXT: store float %[[fnew]], ptr %"p'", align 8
; CHECK: %[[gep:.+]] = getelementptr inbounds i8, ptr %"p'", i64 8
; CHECK: %gep.load = load double, ptr %gep.ptr, align 8
; CHECK-NEXT: %[[dold:.+]] = load double, ptr %[[gep]], align 8
; CHECK-NEXT: %[[dnew:.+]] = fadd fast double %[[dold]], %gep.load
; CHECK-NEXT: store double %[[dnew]], ptr %[[gep]], align 8
