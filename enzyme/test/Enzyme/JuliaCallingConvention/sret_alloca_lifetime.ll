; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; The sret alloca of the caller is replaced by a field of the merged sret, which
; is not an alloca, so its lifetime markers must be removed.

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define double @caller(double %arg) {
entry:
  %sret = alloca { [2 x double], double, i64 }, align 8
  %c = bitcast { [2 x double], double, i64 }* %sret to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %c)
  %res = call i8* @full_write_callee({ [2 x double], double, i64 }* writeonly "enzyme_sret"="test_type4" %sret, double %arg)
  %gep = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret, i64 0, i32 1
  %v = load double, double* %gep, align 8
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %c)
  ret double %v
}

define internal fastcc i8* @full_write_callee({ [2 x double], double, i64 }* noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(32) "enzyme_sret"="test_type4" %0, double %1) {
top:
  store { [2 x double], double, i64 } { [2 x double] [double 1.0, double 2.0], double 3.0, i64 4 }, { [2 x double], double, i64 }* %0, align 8
  ret i8* null
}

; CHECK-LABEL: define double @caller(double %arg)
; CHECK-NOT: llvm.lifetime
; CHECK: call void @full_write_callee(
; CHECK-NOT: llvm.lifetime
; CHECK: ret double
