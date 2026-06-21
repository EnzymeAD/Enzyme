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

; CHECK: define internal void @diffetester(ptr nocapture readonly "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %src, ptr nocapture "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %"src'", ptr nocapture writeonly "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %dst, ptr nocapture "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Integer}" %"dst'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"val'ipl" = load i64, ptr %"src'", align 8
; CHECK-NEXT:   %val = load i64, ptr %src, align 8
; CHECK-NEXT:   %.sroa.1.0.extract.shift = lshr i64 %"val'ipl", 32
; CHECK-NEXT:   %.sroa.1.0.extract.trunc = trunc i64 %.sroa.1.0.extract.shift to i32
; CHECK-NEXT:   %0 = getelementptr inbounds i8, ptr %"dst'", i64 4
; CHECK-NEXT:   store i32 %.sroa.1.0.extract.trunc, ptr %0, align 1
; CHECK-NEXT:   store i64 %val, ptr %dst, align 8
; CHECK-NEXT:   store i32 0, ptr %"dst'", align 8
; CHECK-NEXT:   store float poison, ptr %"src'", align 8
; CHECK-NEXT:   ret void
