; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

source_filename = "partial_int_window"
target triple = "x86_64-pc-linux-gnu"

define void @tester(i64* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %src, i64* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %dst) {
entry:
  %val = load i64, i64* %src, align 8
  store i64 %val, i64* %dst, align 8
  ret void
}

define void @test_derivative(i64* %src, i64* %dsrc, i64* %dst, i64* %ddst) {
entry:
  call void (void (i64*, i64*)*, ...) @__enzyme_autodiff(void (i64*, i64*)* @tester, metadata !"enzyme_dup", i64* %src, i64* %dsrc, metadata !"enzyme_dup", i64* %dst, i64* %ddst)
  ret void
}

declare void @__enzyme_autodiff(void (i64*, i64*)*, ...)

; CHECK: define internal void @diffetester(i64* nocapture readonly "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %src, i64* nocapture "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %"src'", i64* nocapture writeonly "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %dst, i64* nocapture "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %"dst'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"val'ipl" = load i64, i64* %"src'", align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %val = load i64, i64* %src, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %.sroa.1.0.extract.shift = lshr i64 %"val'ipl", 32
; CHECK-NEXT:   %.sroa.1.0.extract.trunc = trunc i64 %.sroa.1.0.extract.shift to i32
; CHECK-NEXT:   %0 = bitcast i64* %"dst'" to i8*
; CHECK-NEXT:   %1 = getelementptr inbounds i8, i8* %0, i64 4
; CHECK-NEXT:   %2 = bitcast i8* %1 to i32*
; CHECK-NEXT:   store i32 %.sroa.1.0.extract.trunc, i32* %2, align 1, !alias.scope !5, !noalias !8
; CHECK-NEXT:   store i64 %val, i64* %dst, align 8, !alias.scope !8, !noalias !5
; CHECK-NEXT:   %3 = load i64, i64* %"dst'", align 8, !alias.scope !5, !noalias !8
; CHECK-NEXT:   %4 = bitcast i64* %"dst'" to i32*
; CHECK-NEXT:   store i32 0, i32* %4, align 8, !alias.scope !5, !noalias !8
; CHECK-NEXT:   %.4.insert.mask = and i64 %3, 4294967295
; CHECK-NEXT:   %5 = bitcast i64* %"src'" to float*
; CHECK-NEXT:   %cast.alloca.sroa.0.0.extract.trunc = trunc i64 %.4.insert.mask to i32
; CHECK-NEXT:   %6 = bitcast i32 %cast.alloca.sroa.0.0.extract.trunc to float
; CHECK-NEXT:   %7 = load float, float* %5, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %8 = fadd fast float %7, %6
; CHECK-NEXT:   store float %8, float* %5, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
