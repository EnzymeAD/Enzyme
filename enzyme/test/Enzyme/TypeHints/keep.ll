; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="type-hints-to-values" -S | FileCheck %s; fi

; Hints that stay: the alloca escapes, the hinted bytes are copied out by a
; memory intrinsic, or a load from the alloca has a variable offset.

declare void @__enzyme_integer(ptr, i64)
declare void @use(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

define i64 @escapes(i64 %n) {
entry:
  %a = alloca i64, align 8
  store i64 %n, ptr %a, align 8
  call void @__enzyme_integer(ptr %a, i64 8)
  call void @use(ptr %a)
  %v = load i64, ptr %a, align 8
  ret i64 %v
}

; CHECK-LABEL: define i64 @escapes(
; CHECK: call void @__enzyme_integer(ptr %a, i64 8)
; CHECK-NOT: __enzyme_typed_value
; CHECK: ret i64 %v

define void @copied_out(i64 %n, ptr %out) {
entry:
  %a = alloca [2 x i64], align 8
  %hi = getelementptr inbounds i8, ptr %a, i64 8
  store i64 %n, ptr %hi, align 8
  call void @__enzyme_integer(ptr %hi, i64 8)
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %a, i64 16, i1 false)
  ret void
}

; CHECK-LABEL: define void @copied_out(
; CHECK: call void @__enzyme_integer(ptr %hi, i64 8)

; Copying into the alloca, or out of bytes the hint does not cover, is fine.
define i64 @copied_other(ptr %in, ptr %out) {
entry:
  %a = alloca [2 x i64], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %in, i64 16, i1 false)
  %hi = getelementptr inbounds i8, ptr %a, i64 8
  call void @__enzyme_integer(ptr %hi, i64 8)
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %a, i64 8, i1 false)
  %v = load i64, ptr %hi, align 8
  ret i64 %v
}

; CHECK-LABEL: define i64 @copied_other(
; CHECK-NOT: @__enzyme_integer(
; CHECK: %[[T:.+]] = call i64 @__enzyme_typed_value.i64(i64 %v)
; CHECK: ret i64 %[[T]]

define i64 @variable_offset(i64 %n, i64 %j) {
entry:
  %a = alloca [4 x i64], align 8
  store i64 %n, ptr %a, align 8
  call void @__enzyme_integer(ptr %a, i64 8)
  %p = getelementptr inbounds i64, ptr %a, i64 %j
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

; CHECK-LABEL: define i64 @variable_offset(
; CHECK: call void @__enzyme_integer(ptr %a, i64 8)
; CHECK-NOT: __enzyme_typed_value
; CHECK: ret i64 %v
