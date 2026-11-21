; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -enzyme -enzyme-preopt=false -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme,function(early-cse)" -enzyme-preopt=false -S | FileCheck %s

source_filename = "customfwd_int.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__enzyme_register_derivative_square = dso_local local_unnamed_addr global [2 x i8*] [i8* bitcast (i32 (i32)* @square to i8*), i8* bitcast (i32 (i32, i32)* @derivative_square to i8*)], align 16

; Function Attrs: norecurse nounwind readnone uwtable willreturn
define i32 @square(i32 %x) #0 {
entry:
  %mul = mul i32 %x, %x
  ret i32 %mul
}

define i32 @derivative_square(i32 %x, i32 %dx) #0 {
entry:
  ret i32 100
}

; Function Attrs: nounwind uwtable
define i32 @caller(i32 %x, i32 %dx) {
entry:
  %call = call i32 (i8*, ...) @__enzyme_fwddiff(i8* bitcast (i32 (i32)* @square to i8*), metadata !"enzyme_dup", i32 %x, i32 %dx)
  ret i32 %call
}

declare dso_local i32 @__enzyme_fwddiff(i8*, ...)

attributes #0 = { norecurse nounwind readnone }

; CHECK: define internal i32 @fwddiffesquare(i32 %x, i32 %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call i32 @derivative_square(i32 %x, i32 %"x'")
; CHECK-NEXT:   ret i32 %0
; CHECK-NEXT: }
