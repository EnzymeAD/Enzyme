; RUN: if [ %llvmver -ge 22 ]; then %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s; fi

; The two sret allocas of the call are merged into one, and each is replaced by
; a GEP into it. Since LLVM 22 lifetime markers may only take an alloca, so the
; markers of the replaced allocas must be dropped for the module to verify.

declare void @llvm.lifetime.start.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr)

define void @caller(ptr %out) {
entry:
  %a = alloca { double }, align 8
  %b = alloca [6 x i64], align 8
  call void @llvm.lifetime.start.p0(ptr %a)
  call void @llvm.lifetime.start.p0(ptr %b)
  call void @callee(ptr "enzyme_sret"="test_type3" "enzyme_type"="{[-1]:Pointer}" %a, ptr "enzyme_sret"="test_type2" "enzyme_type"="{[-1]:Pointer}" %b)
  %v = load double, ptr %a, align 8
  store double %v, ptr %out, align 8
  call void @llvm.lifetime.end.p0(ptr %a)
  call void @llvm.lifetime.end.p0(ptr %b)
  ret void
}

define void @callee(ptr "enzyme_sret"="test_type3" "enzyme_type"="{[-1]:Pointer}" %x, ptr "enzyme_sret"="test_type2" "enzyme_type"="{[-1]:Pointer}" %y) {
entry:
  store double 1.0, ptr %x, align 8
  store [6 x i64] zeroinitializer, ptr %y, align 8
  ret void
}

; CHECK-LABEL: define void @caller(ptr %out)
; CHECK: %stack_sret = alloca { { double }, [6 x i64] }
; CHECK-NOT: @llvm.lifetime
; CHECK: call void @callee(ptr sret({ { double }, [6 x i64] }) %stack_sret)
; CHECK-NOT: @llvm.lifetime
; CHECK: ret void
