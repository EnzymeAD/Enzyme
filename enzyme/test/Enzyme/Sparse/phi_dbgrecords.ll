; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -eq 18 ]; then %opt < %s %OPnewLoadEnzyme --try-experimental-debuginfo-iterators=true -passes="enzyme,verify" -enzyme-auto-sparsity=1 -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare ptr @__enzyme_todense(...)
declare void @__enzyme_autodiff(...)
declare void @accumulate(i64, i64, double, ptr) #0

define internal double @seed_load(i64 %off, i64 %col) #1 {
  %idx = lshr exact i64 %off, 3
  %match = icmp eq i64 %idx, %col
  %res = uitofp i1 %match to double
  ret double %res
}

define internal void @nostore(double %val, i64 %off, i64 %col) #1 {
  ret void
}

define internal double @zero_load(i64 %off, i64 %col, ptr %acc) #1 {
  ret double 0.000000e+00
}

define internal void @acc_store(double %val, i64 %off, i64 %col, ptr %acc) #1 {
entry:
  %nz = fcmp une double %val, 0.000000e+00
  br i1 %nz, label %do, label %end

do:
  call void @accumulate(i64 %off, i64 %col, double %val, ptr %acc)
  br label %end

end:
  ret void
}

; CHECK-LABEL: define void @sparse_jac(
; CHECK: outer:
; CHECK:   call void @sparse_jac.inner(
define void @sparse_jac(i64 %n, ptr %acc) !dbg !4 {
entry:
  br label %outer

outer:
  %col = phi i64 [ 0, %entry ], [ %col.next, %outer.latch ]
  %dx = call ptr (...) @__enzyme_todense(ptr @seed_load, ptr @nostore, i64 %col)
  %dy = call ptr (...) @__enzyme_todense(ptr @zero_load, ptr @acc_store, i64 %col, ptr %acc)
  %init_off = add nuw nsw i64 %col, 0
  call void @llvm.dbg.value(metadata i64 %col, metadata !8, metadata !DIExpression()), !dbg !7
  br label %inner, !dbg !7

inner:
  %iv = phi i64 [ 0, %outer ], [ %iv.next, %inner ]
  %shifted = phi i64 [ %init_off, %outer ], [ %shifted.next, %inner ]
  call void @llvm.dbg.value(metadata i64 %iv, metadata !8, metadata !DIExpression()), !dbg !7
  call void @llvm.dbg.value(metadata i64 %shifted, metadata !8, metadata !DIExpression()), !dbg !7
  %iv.next = add nuw nsw i64 %iv, 1
  %shifted.next = add nuw nsw i64 %iv.next, %col
  %px = getelementptr inbounds double, ptr %dx, i64 %iv
  %vx = load double, ptr %px, align 8
  %scaled = fmul fast double %vx, 3.000000e+00
  %py = getelementptr inbounds double, ptr %dy, i64 %iv
  store double %scaled, ptr %py, align 8
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %outer.latch, label %inner, !dbg !10

outer.latch:
  %col.next = add nuw nsw i64 %col, 1
  %cdone = icmp eq i64 %col.next, %n
  br i1 %cdone, label %exit, label %outer

exit:
  ret void, !dbg !11
}

define void @ptr_phi_kernel(ptr %left, ptr %right, i1 %pick_left, i64 %n) #2 !dbg !12 {
entry:
  br i1 %pick_left, label %from_left, label %from_right

from_left:
  store double 1.000000e+00, ptr %left, align 8
  br label %merge

from_right:
  br label %merge

merge:
  %chosen = phi ptr [ %left, %from_left ], [ %right, %from_right ]
  call void @llvm.dbg.value(metadata ptr %chosen, metadata !13, metadata !DIExpression()), !dbg !14
  br label %loop

loop:
  %idx = phi i64 [ 0, %merge ], [ %idx.next, %loop ]
  call void @llvm.dbg.value(metadata i64 %idx, metadata !15, metadata !DIExpression()), !dbg !14
  %slot = getelementptr inbounds double, ptr %chosen, i64 %idx
  %cur = load double, ptr %slot, align 8
  %twice = fmul double %cur, 2.000000e+00
  store double %twice, ptr %slot, align 8
  %idx.next = add nuw nsw i64 %idx, 1
  %cont = icmp ult i64 %idx.next, %n
  br i1 %cont, label %loop, label %done, !dbg !14

done:
  ret void, !dbg !14
}

; CHECK-LABEL: define void @run_ptr_phi_grad(
; CHECK: call void @diffeptr_phi_kernel(
define void @run_ptr_phi_grad(ptr %left, ptr %dleft, ptr %right, ptr %dright, i1 %pick_left, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @ptr_phi_kernel, metadata !"enzyme_dup", ptr %left, ptr %dleft, metadata !"enzyme_dup", ptr %right, ptr %dright, metadata !"enzyme_const", i1 %pick_left, metadata !"enzyme_const", i64 %n)
  ret void
}

; CHECK-LABEL: define internal void @diffeptr_phi_kernel(
; CHECK: merge:
; CHECK-NEXT: %[[SHADOW_PHI:.+]] = phi ptr
; CHECK-NEXT: %chosen = phi ptr

attributes #0 = { noinline "enzyme_sparse_accumulate" }
attributes #1 = { alwaysinline }
attributes #2 = { noinline }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "phi_dbgrecords.c", directory: "/tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "sparse_jac", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocation(line: 2, column: 3, scope: !4)
!8 = !DILocalVariable(name: "iv", scope: !4, file: !1, line: 2, type: !9)
!9 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!10 = !DILocation(line: 8, column: 3, scope: !4)
!11 = !DILocation(line: 10, column: 1, scope: !4)
!12 = distinct !DISubprogram(name: "ptr_phi_kernel", scope: !1, file: !1, line: 12, type: !5, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocalVariable(name: "chosen", scope: !12, file: !1, line: 13, type: !9)
!14 = !DILocation(line: 14, column: 3, scope: !12)
!15 = !DILocalVariable(name: "idx", scope: !12, file: !1, line: 15, type: !9)
