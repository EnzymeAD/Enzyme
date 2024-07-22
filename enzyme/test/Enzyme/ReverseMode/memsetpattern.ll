; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

@.memset_pattern = private unnamed_addr constant [2 x double] [double 0x3FE5555555555555, double 0x3FE5555555555555], align 16

declare void @__enzyme_autodiff(i8*, double*, double*)

declare void @memset_pattern16(double* nocapture writeonly, i8* nocapture readonly, i64)

define void @f(double* %x) {
  call void @memset_pattern16(double* %x, i8* bitcast ([2 x double]* @.memset_pattern to i8*), i64 16)
  ret void
}

define void @df(double* %x, double* %xp) {
  tail call void @__enzyme_autodiff(i8* bitcast (void (double*)* @f to i8*), double* %x, double* %xp)
  ret void
}

; CHECK: define internal void @diffef(double* %x, double* %"x'")
; CHECK-NEXT: invert:
; CHECK-NEXT:   call void @memset_pattern16(double* %x, i8* bitcast ([2 x double]* @.memset_pattern to i8*), i64 16)
; CHECK-NEXT:   %0 = bitcast double* %"x'" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 16, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
