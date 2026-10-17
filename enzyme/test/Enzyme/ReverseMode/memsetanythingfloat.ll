; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; The memset memory's type tree has Anything bytes (struct padding) next to
; floats: {[0]:Float@float, [4-7]:Anything, [8]:Float@double}
; The memset overwrites the double stored from %x, so d/dx must be zero: the
; reverse pass must zero the shadow of the float bytes, rather than treating
; the whole range as Anything.

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

declare double @__enzyme_autodiff(...)

define double @f(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Anything, [-1,5]:Anything, [-1,6]:Anything, [-1,7]:Anything, [-1,8]:Float@double}" %p, double %x) {
entry:
  %gep = getelementptr inbounds i8, ptr %p, i64 8
  store double %x, ptr %gep, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %p, i8 0, i64 16, i1 false)
  %r = load double, ptr %gep, align 8
  ret double %r
}

define double @test(ptr %p, ptr %dp, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %p, ptr %dp, double %x)
  ret double %r
}

; CHECK: define internal { double } @diffef(ptr {{.*}}%p, ptr {{.*}}%"p'", double %x, double %differeturn)
; CHECK: %[[fadd:.+]] = fadd fast double %{{.+}}, %differeturn
; CHECK-NEXT: store double %[[fadd]], ptr %"gep'ipg", align 8
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 8 %"p'", i8 0, i64 4, i1 false)
; CHECK-NEXT: %[[off:.+]] = getelementptr inbounds i8, ptr %"p'", i32 8
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 8 %[[off]], i8 0, i64 8, i1 false)
; CHECK-NEXT: %[[dx:.+]] = load double, ptr %"gep'ipg", align 8
