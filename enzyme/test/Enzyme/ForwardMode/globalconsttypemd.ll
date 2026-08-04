; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Constant globals have their type analysis result cached as !enzyme_type
; metadata. Offsets within a TypeTree may be negative (-1 denotes "any
; offset"), so the offsets must be serialized as signed constants.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@str = private unnamed_addr constant [12 x i8] c"src/main.rs\00", align 1
@loc = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @str, [16 x i8] c"\0B\00\00\00\00\00\00\00\08\00\00\00\05\00\00\00" }>, align 8

define float @mul(ptr align 4 %a, ptr align 4 %b, ptr align 8 %loc) {
  %la = load float, ptr %a, align 4
  %lb = load float, ptr %b, align 4
  %res = fmul float %la, %lb
  ret float %res
}

define float @f(ptr align 4 %a, ptr align 4 %b) {
  %res = call float @mul(ptr align 4 %a, ptr align 4 %b, ptr align 8 @loc)
  ret float %res
}

define float @df(ptr %a, ptr %da, ptr %b, ptr %db) {
  %res = call float (...) @__enzyme_fwddiff(ptr @f, ptr %a, ptr %da, ptr %b, ptr %db)
  ret float %res
}

declare float @__enzyme_fwddiff(...)

; CHECK: @str = private unnamed_addr constant [12 x i8] c"src/main.rs\00", align 1, !enzyme_type ![[STRTY:[0-9]+]]
; CHECK: @loc = {{.*}}, !enzyme_type ![[LOCTY:[0-9]+]]

; CHECK-DAG: ![[STRTY]] = !{!"Unknown", i32 -1, ![[ANYPTR:[0-9]+]]}
; CHECK-DAG: ![[ANYPTR]] = !{!"Pointer", i32 -1, !{{[0-9]+}}}
; CHECK-DAG: ![[LOCTY]] = !{!"Unknown", i32 -1, !{{[0-9]+}}}
