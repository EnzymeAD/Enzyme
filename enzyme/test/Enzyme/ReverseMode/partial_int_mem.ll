; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | FileCheck %s; fi

declare void @__enzyme_autodiff(...)
@enzyme_dup = global i32 0

; Partial integer memory (bytes 0..3 Integer, bytes 4..7 Unknown) must not
; treat the 8-byte store into %slot as all-integral and drop %din.
define void @f_partial_int(ptr %in, ptr %out, i64 %idx) {
entry:
  %buf = alloca [2 x i64], align 8
  %slot = getelementptr inbounds [2 x i64], ptr %buf, i64 0, i64 %idx
  store i32 42, ptr %slot, align 4
  %v = load i64, ptr %in, align 8
  store i64 %v, ptr %slot, align 8
  %v2 = load i64, ptr %slot, align 8
  store i64 %v2, ptr %out, align 8
  ret void
}

; Integer + Anything (from float 0.0) must not treat the 8-byte store as
; all-integral and drop %din.
define void @f_int_float0(ptr %in, ptr %out, i64 %idx) {
entry:
  %buf = alloca [2 x { i32, float }], align 8
  %slot = getelementptr inbounds [2 x { i32, float }], ptr %buf, i64 0, i64 %idx
  store { i32, float } { i32 42, float 0.0 }, ptr %slot, align 8
  %v = load i64, ptr %in, align 8
  store i64 %v, ptr %slot, align 8
  %v2 = load i64, ptr %slot, align 8
  store i64 %v2, ptr %out, align 8
  ret void
}

define void @df_partial_int(ptr %in, ptr %din, ptr %out, ptr %dout, i64 %idx) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f_partial_int, ptr @enzyme_dup, ptr %in, ptr %din, ptr @enzyme_dup, ptr %out, ptr %dout, i64 %idx)
  ret void
}

define void @df_int_float0(ptr %in, ptr %din, ptr %out, ptr %dout, i64 %idx) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f_int_float0, ptr @enzyme_dup, ptr %in, ptr %din, ptr @enzyme_dup, ptr %out, ptr %dout, i64 %idx)
  ret void
}

; CHECK-LABEL: define void @df_partial_int(
; CHECK: %[[DV:[^ ]+]] = load i64, ptr %din, align 8
; CHECK: store i64 %[[DV]], ptr %dout, align 8

; CHECK-LABEL: define void @df_int_float0(
; CHECK: %[[DV2:[^ ]+]] = load i64, ptr %din, align 8
; CHECK: store i64 %[[DV2]], ptr %dout, align 8
