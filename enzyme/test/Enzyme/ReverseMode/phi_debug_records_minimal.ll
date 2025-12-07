; Test for fixing assertion failure when inserting PHI nodes with debug records
; This ensures PHI nodes are inserted at the correct position with LLVM's new debug record format
; RUN: if [ %llvmver -ge 18 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @__enzyme_autodiff(...) 

; Function with a loop that has dynamic bounds to trigger getDynamicLoopLimit
define void @loop_fn(ptr %data, i64 %n) #0 !dbg !4 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
    #dbg_value(i64 %i, !8, !DIExpression(), !7)
  %ptr = getelementptr inbounds double, ptr %data, i64 %i
  %val = load double, ptr %ptr, align 8
  %val2 = fmul double %val, 2.0
  store double %val2, ptr %ptr, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !dbg !10

exit:
  ret void, !dbg !11
}

; Enzyme autodiff call
define void @test_derivative(ptr %data, ptr %data_grad, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @loop_fn, metadata !"enzyme_dup", ptr %data, ptr %data_grad, metadata !"enzyme_const", i64 %n)
  ret void
}

; CHECK: define internal void @diffeloop_fn(ptr %data, ptr %"data'", i64 %n)

attributes #0 = { noinline }

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
!8 = !DILocalVariable(name: "i", scope: !4, file: !1, line: 2, type: !9)
!9 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!10 = !DILocation(line: 5, column: 3, scope: !4)
!11 = !DILocation(line: 6, column: 1, scope: !4)
