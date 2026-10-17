; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; A copy from a flang character literal (@_QQcl...) into memory of otherwise
; unknown type copies character data, which is an integer.

@_QQclX616263 = linkonce_odr constant [3 x i8] c"abc"

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

define void @f(double* %x, i8* %buf) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %buf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_QQclX616263, i64 0, i64 0), i64 3, i1 false)
  %v = load double, double* %x
  %m = fmul double %v, %v
  store double %m, double* %x
  ret void
}

declare void @__enzyme_autodiff(...)

define void @df(double* %x, double* %dx, i8* %buf, i8* %dbuf) {
entry:
  call void (...) @__enzyme_autodiff(void (double*, i8*)* @f, metadata !"enzyme_dup", double* %x, double* %dx, metadata !"enzyme_dup", i8* %buf, i8* %dbuf)
  ret void
}

; CHECK: define internal void @diffef(double* {{.*}}%x, double* {{.*}}%"x'", i8* {{.*}}%buf, i8* {{.*}}%"buf'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %"buf'", i8* align 1 getelementptr inbounds ([3 x i8], [3 x i8]* @_QQclX616263, i64 0, i64 0), i64 3, i1 false)
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %buf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_QQclX616263, i64 0, i64 0), i64 3, i1 false)
; CHECK-NEXT:   %v = load double, double* %x
