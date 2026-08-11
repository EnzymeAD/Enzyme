; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

source_filename = "partial_int_window"
target triple = "x86_64-pc-linux-gnu"

define void @tester(i64* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %src, i64* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %dst) {
entry:
  %val = load i64, i64* %src, align 8
  store i64 %val, i64* %dst, align 8
  ret void
}

define void @test_derivative(i64* %src, i64* %dsrc1, i64* %dsrc2, i64* %dst, i64* %ddst1, i64* %ddst2) {
entry:
  call void (void (i64*, i64*)*, ...) @__enzyme_autodiff(void (i64*, i64*)* @tester, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", i64* %src, i64* %dsrc1, i64* %dsrc2, metadata !"enzyme_dup", i64* %dst, i64* %ddst1, i64* %ddst2)
  ret void
}

declare void @__enzyme_autodiff(void (i64*, i64*)*, ...)

; CHECK: define internal void @diffe2tester(i64* nocapture readonly "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %src, [2 x i64*] "enzyme_type_v"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %"src'", i64* nocapture writeonly "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %dst, [2 x i64*] "enzyme_type_v"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %"dst'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x i64*] %"src'", 0
; CHECK-NEXT:   %"val'ipl" = load i64, i64* %0, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %1 = extractvalue [2 x i64*] %"src'", 1
; CHECK-NEXT:   %"val'ipl1" = load i64, i64* %1, align 8, !alias.scope !6, !noalias !7
; CHECK-NEXT:   %val = load i64, i64* %src, align 8, !alias.scope !8, !noalias !9
; CHECK-NEXT:   %2 = extractvalue [2 x i64*] %"dst'", 0
; CHECK-NEXT:   %.sroa.15.0.extract.shift = lshr i64 %"val'ipl", 32
; CHECK-NEXT:   %.sroa.15.0.extract.trunc = trunc i64 %.sroa.15.0.extract.shift to i32
; CHECK-NEXT:   %3 = bitcast i64* %2 to i8*
; CHECK-NEXT:   %4 = getelementptr inbounds i8, i8* %3, i64 4
; CHECK-NEXT:   %5 = bitcast i8* %4 to i32*
; CHECK-NEXT:   store i32 %.sroa.15.0.extract.trunc, i32* %5, align 1, !alias.scope !10, !noalias !13
; CHECK-NEXT:   %6 = extractvalue [2 x i64*] %"dst'", 1
; CHECK-NEXT:   %.sroa.1.0.extract.shift = lshr i64 %"val'ipl1", 32
; CHECK-NEXT:   %.sroa.1.0.extract.trunc = trunc i64 %.sroa.1.0.extract.shift to i32
; CHECK-NEXT:   %7 = bitcast i64* %6 to i8*
; CHECK-NEXT:   %8 = getelementptr inbounds i8, i8* %7, i64 4
; CHECK-NEXT:   %9 = bitcast i8* %8 to i32*
; CHECK-NEXT:   store i32 %.sroa.1.0.extract.trunc, i32* %9, align 1, !alias.scope !16, !noalias !17
; CHECK-NEXT:   store i64 %val, i64* %dst, align 8, !alias.scope !18, !noalias !19
; CHECK-NEXT:   %10 = load i64, i64* %2, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   %11 = load i64, i64* %6, align 8, !alias.scope !16, !noalias !17
; CHECK-NEXT:   %12 = bitcast i64* %2 to i32*
; CHECK-NEXT:   store i32 0, i32* %12, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   %13 = bitcast i64* %6 to i32*
; CHECK-NEXT:   store i32 0, i32* %13, align 8, !alias.scope !16, !noalias !17
; CHECK-NEXT:   %.sroa.0.4.insert.mask = and i64 %10, 4294967295
; CHECK-NEXT:   %.sroa.3.12.insert.mask = and i64 %11, 4294967295
; CHECK-NEXT:   %"val'de.sroa.0.0.extract.trunc" = trunc i64 %.sroa.0.4.insert.mask to i8
; CHECK-NEXT:   %"val'de.sroa.5.0.extract.shift" = lshr i64 %.sroa.0.4.insert.mask, 8
; CHECK-NEXT:   %"val'de.sroa.5.0.extract.trunc" = trunc i64 %"val'de.sroa.5.0.extract.shift" to i8
; CHECK-NEXT:   %"val'de.sroa.6.0.extract.shift" = lshr i64 %.sroa.0.4.insert.mask, 16
; CHECK-NEXT:   %"val'de.sroa.6.0.extract.trunc" = trunc i64 %"val'de.sroa.6.0.extract.shift" to i8
; CHECK-NEXT:   %"val'de.sroa.7.0.extract.shift" = lshr i64 %.sroa.0.4.insert.mask, 24
; CHECK-NEXT:   %"val'de.sroa.7.0.extract.trunc" = trunc i64 %"val'de.sroa.7.0.extract.shift" to i8
; CHECK-NEXT:   %"val'de.sroa.8.0.extract.shift" = lshr i64 %.sroa.0.4.insert.mask, 32
; CHECK-NEXT:   %"val'de.sroa.8.0.extract.trunc" = trunc i64 %"val'de.sroa.8.0.extract.shift" to i32
; CHECK-NEXT:   %"val'de.sroa.8.0.insert.ext29" = zext i32 %"val'de.sroa.8.0.extract.trunc" to i64
; CHECK-NEXT:   %"val'de.sroa.8.0.insert.shift30" = shl i64 %"val'de.sroa.8.0.insert.ext29", 32
; CHECK-NEXT:   %"val'de.sroa.7.0.insert.ext22" = zext i8 %"val'de.sroa.7.0.extract.trunc" to i64
; CHECK-NEXT:   %"val'de.sroa.7.0.insert.shift23" = shl i64 %"val'de.sroa.7.0.insert.ext22", 24
; CHECK-NEXT:   %"val'de.sroa.7.0.insert.insert25" = or i64 %"val'de.sroa.8.0.insert.shift30", %"val'de.sroa.7.0.insert.shift23"
; CHECK-NEXT:   %"val'de.sroa.6.0.insert.ext18" = zext i8 %"val'de.sroa.6.0.extract.trunc" to i64
; CHECK-NEXT:   %"val'de.sroa.6.0.insert.shift19" = shl i64 %"val'de.sroa.6.0.insert.ext18", 16
; CHECK-NEXT:   %"val'de.sroa.6.0.insert.insert21" = or i64 %"val'de.sroa.7.0.insert.insert25", %"val'de.sroa.6.0.insert.shift19"
; CHECK-NEXT:   %"val'de.sroa.5.0.insert.ext14" = zext i8 %"val'de.sroa.5.0.extract.trunc" to i64
; CHECK-NEXT:   %"val'de.sroa.5.0.insert.shift15" = shl i64 %"val'de.sroa.5.0.insert.ext14", 8
; CHECK-NEXT:   %"val'de.sroa.5.0.insert.insert17" = or i64 %"val'de.sroa.6.0.insert.insert21", %"val'de.sroa.5.0.insert.shift15"
; CHECK-NEXT:   %"val'de.sroa.0.0.insert.ext11" = zext i8 %"val'de.sroa.0.0.extract.trunc" to i64
; CHECK-NEXT:   %"val'de.sroa.0.0.insert.insert13" = or i64 %"val'de.sroa.5.0.insert.insert17", %"val'de.sroa.0.0.insert.ext11"
; CHECK-NEXT:   %14 = bitcast i64* %0 to float*
; CHECK-NEXT:   %15 = bitcast i64* %1 to float*
; CHECK-NEXT:   %cast.alloca.sroa.0.0.extract.trunc = trunc i64 %"val'de.sroa.0.0.insert.insert13" to i32
; CHECK-NEXT:   %16 = bitcast i32 %cast.alloca.sroa.0.0.extract.trunc to float
; CHECK-NEXT:   %cast.alloca2.sroa.0.0.extract.trunc = trunc i64 %.sroa.3.12.insert.mask to i32
; CHECK-NEXT:   %17 = bitcast i32 %cast.alloca2.sroa.0.0.extract.trunc to float
; CHECK-NEXT:   %18 = load float, float* %14, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %19 = fadd fast float %18, %16
; CHECK-NEXT:   store float %19, float* %14, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %20 = load float, float* %15, align 8, !alias.scope !6, !noalias !7
; CHECK-NEXT:   %21 = fadd fast float %20, %17
; CHECK-NEXT:   store float %21, float* %15, align 8, !alias.scope !6, !noalias !7
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
