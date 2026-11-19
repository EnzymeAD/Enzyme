; RUN: if [ %llvmver -ge 22 ]; then %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s; fi

; The sret alloca of the caller is replaced by a field of the merged sret, which
; is not an alloca, so its lifetime markers become llvm.enzyme.lifetime calls.

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)

define double @caller(double %arg) {
entry:
  %sret = alloca { [2 x double], double, i64 }, align 8
  call void @llvm.lifetime.start.p0(ptr %sret)
  %res = call ptr @full_write_callee(ptr writeonly "enzyme_sret"="test_type4" %sret, double %arg)
  %gep = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret, i64 0, i32 1
  %v = load double, ptr %gep, align 8
  call void @llvm.lifetime.end.p0(ptr %sret)
  ret double %v
}

define internal fastcc ptr @full_write_callee(ptr noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(32) "enzyme_sret"="test_type4" %0, double %1) {
top:
  store { [2 x double], double, i64 } { [2 x double] [double 1.0, double 2.0], double 3.0, i64 4 }, ptr %0, align 8
  ret ptr null
}

; CHECK-LABEL: define double @caller(double %arg)
; CHECK-NOT: @llvm.lifetime
; CHECK: call void (...) @llvm.enzyme.lifetime_start(ptr
; CHECK: call void @full_write_callee(
; CHECK: call void (...) @llvm.enzyme.lifetime_end(ptr
; CHECK-NOT: @llvm.lifetime
; CHECK: ret double
