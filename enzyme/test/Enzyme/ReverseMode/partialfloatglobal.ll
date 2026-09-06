; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

; @tab holds two packed doubles as raw bytes, so its ConstantDataArray elements
; are i8. A single byte overlaps a double without covering all of it, so
; inverting that element used to take the generic partially-float path in
; invertPointerM, which emits an alloca/store/load and hands back a LoadInst.
; The ConstantDataArray case then did cast<Constant> on it and asserted with
; "cast<Ty>() argument of incompatible type!".

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@tab = global [16 x i8] c"UUUUUU\D5?\00\00\00\00\00\00\D0?"
@enzyme_const = external global ptr
@enzyme_dup = external global ptr

define void @f(ptr %a, i64 %n, ptr %b, i64 %m, i32 %c) {
  store i32 0, ptr @tab, align 4
  ret void
}

define {} @entry() {
  %r = tail call {} (...) @__enzyme_autodiff(ptr @f, ptr @enzyme_dup, ptr null, ptr null, ptr @enzyme_const, i64 0, ptr @enzyme_dup, ptr null, ptr null, ptr @enzyme_const, i64 0, ptr @enzyme_const, i32 0)
  ret {} %r
}

declare {} @__enzyme_autodiff(...)

; Every byte of @tab is part of a double, so the whole shadow folds to zero and
; stays a constant initializer.
; CHECK: @tab_shadow = global [16 x i8] zeroinitializer
