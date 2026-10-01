; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -S | FileCheck %s; fi

; Separate compilation: @copy is exported because of its attribute, and its
; untyped memcpy is differentiable only because the argument types are
; declared on the signature. @user calls @ext, which has no body here, through
; the external derivative table its defining module exports.

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

define void @copy(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %x, ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %y, i64 %n) "enzyme_export_derivative"="reverse,forward" {
  %b = shl i64 %n, 3
  call void @llvm.memcpy.p0.p0.i64(ptr %y, ptr %x, i64 %b, i1 false)
  ret void
}

declare void @ext(ptr, ptr)

define void @user(ptr %x, ptr %y) {
  call void @ext(ptr %x, ptr %y)
  ret void
}

declare void @__enzyme_autodiff(...)

define void @caller(ptr %x, ptr %dx, ptr %y, ptr %dy) {
  call void (...) @__enzyme_autodiff(ptr @user, ptr %x, ptr %dx, ptr %y, ptr %dy)
  ret void
}

; CHECK-DAG: @__enzyme_sep_rev_w1_ext = external constant { ptr, ptr }
; CHECK-DAG: @__enzyme_sep_fwd_w1_copy = constant ptr @fwddiffecopy
; CHECK-DAG: @__enzyme_sep_rev_w1_copy = constant { ptr, ptr } { ptr @augmented_copy, ptr @diffecopy }

; CHECK: define internal void @diffeuser(ptr %x, ptr %"x'", ptr %y, ptr %"y'")
; CHECK-NEXT:   %1 = load ptr, ptr @__enzyme_sep_rev_w1_ext, align 8
; CHECK-NEXT:   %_augmented = call { ptr } %1(ptr %x, ptr %"x'", ptr %y, ptr %"y'")
; CHECK-NEXT:   %subcache = extractvalue { ptr } %_augmented, 0
; CHECK:   %2 = load ptr, ptr getelementptr (ptr, ptr @__enzyme_sep_rev_w1_ext, i64 1), align 8
; CHECK-NEXT:   %3 = call {} %2(ptr %x, ptr %"x'", ptr %y, ptr %"y'", ptr %subcache)
