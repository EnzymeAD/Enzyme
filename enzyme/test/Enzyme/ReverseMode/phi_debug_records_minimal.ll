; Test for fixing assertion failure when inserting PHI nodes after debug records
; This test ensures PHI nodes are inserted correctly in blocks with debug records
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @__enzyme_autodiff(...) 

; Simple function with a loop that has dynamic bounds
define void @loop_fn(double* %data, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %ptr = getelementptr inbounds double, double* %data, i64 %i
  %val = load double, double* %ptr, align 8
  %val2 = fmul double %val, 2.0
  store double %val2, double* %ptr, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !dbg !10

exit:
  ret void, !dbg !11
}

; Enzyme autodiff call
define void @test_derivative(double* %data, double* %data_grad, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(void (double*, i64)* @loop_fn, metadata !"enzyme_dup", double* %data, double* %data_grad, metadata !"enzyme_const", i64 %n)
  ret void
}

; CHECK: define internal void @diffeloop_fn(double* %data, double* %"data'", i64 %n)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "loop_fn", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocation(line: 1, column: 1, scope: !4)
!10 = !DILocation(line: 5, column: 3, scope: !4)
!11 = !DILocation(line: 6, column: 1, scope: !4)
