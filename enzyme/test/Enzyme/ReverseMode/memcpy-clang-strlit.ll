; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; A copy from a Clang/LLVM string literal (@.str, @.str.<N>) into memory of
; otherwise unknown type copies character data, which is an integer.

@.str = private unnamed_addr constant [8 x i8] c"alpha_x\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"alpha_y\00", align 1

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

define void @f(ptr %x, ptr %buf1, ptr %buf2) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %buf1, ptr @.str, i64 7, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %buf2, ptr @.str.1, i64 7, i1 false)
  %v = load double, ptr %x, align 8
  %m = fmul double %v, %v
  store double %m, ptr %x, align 8
  ret void
}

declare void @__enzyme_autodiff(...)

define void @df(ptr %x, ptr %dx, ptr %buf1, ptr %dbuf1, ptr %buf2, ptr %dbuf2) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %buf1, ptr %dbuf1, metadata !"enzyme_dup", ptr %buf2, ptr %dbuf2)
  ret void
}

; CHECK: define internal void @diffef(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%buf1, ptr {{.*}}%"buf1'", ptr {{.*}}%buf2, ptr {{.*}}%"buf2'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %"buf1'", ptr align 1 @.str, i64 7, i1 false)
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %buf1, ptr @.str, i64 7, i1 false)
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %"buf2'", ptr align 1 @.str.1, i64 7, i1 false)
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %buf2, ptr @.str.1, i64 7, i1 false)
; CHECK-NEXT:   %v = load double, ptr %x, align 8
