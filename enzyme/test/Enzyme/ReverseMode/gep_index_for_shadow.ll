; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EXEC

; The pointer walked by the loop advances by a stride loaded from memory, so
; %curnext depends on a value that is not available in the reverse pass. The
; min cut can satisfy the primal need by caching %curnext itself, but the
; shadow of %curnext is rebuilt in the reverse pass as a getelementptr of the
; shadow base with the primal index, which therefore has to be kept as well.

declare void @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
@fmt = private constant [16 x i8] c"%f %f %f %f %f\0A\00"

define internal void @precalc(ptr %diag, ptr %rows, ptr %vals, i32 %n) noinline {
entry:
  %pos = icmp sgt i32 %n, 0
  br i1 %pos, label %loop, label %exit

loop:
  %i = phi i32 [ %inext, %loop ], [ 0, %entry ]
  %row = phi ptr [ %rownext, %loop ], [ %rows, %entry ]
  %out = phi ptr [ %outnext, %loop ], [ %diag, %entry ]
  %cur = phi ptr [ %curnext, %loop ], [ %vals, %entry ]
  %a = load double, ptr %cur, align 8
  %p1 = getelementptr inbounds double, ptr %cur, i64 1
  %b = load double, ptr %p1, align 8
  %ab = fadd double %a, %b
  %inv = fdiv double 1.000000e+00, %ab
  store double %inv, ptr %out, align 8
  %outnext = getelementptr inbounds double, ptr %out, i64 1
  %rownext = getelementptr inbounds i32, ptr %row, i64 1
  %stride = load i32, ptr %row, align 4
  %off = sext i32 %stride to i64
  %curnext = getelementptr inbounds double, ptr %cur, i64 %off
  %inext = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @outer(ptr %diag, ptr %rows, ptr %vals, i32 %n) {
  call void @precalc(ptr %diag, ptr %rows, ptr %vals, i32 %n)
  %d0 = load double, ptr %diag, align 8
  %d1p = getelementptr inbounds double, ptr %diag, i64 1
  %d1 = load double, ptr %d1p, align 8
  %s = fadd double %d0, %d1
  store double %s, ptr %diag, align 8
  ret void
}

define i32 @main() {
  %diag = alloca [2 x double], align 8
  %ddiag = alloca [2 x double], align 8
  %rows = alloca [2 x i32], align 4
  %vals = alloca [5 x double], align 8
  %dvals = alloca [5 x double], align 8
  store [2 x double] zeroinitializer, ptr %diag
  store [2 x double] [double 1.0, double 0.0], ptr %ddiag
  store [2 x i32] [i32 3, i32 1], ptr %rows
  store [5 x double] [double 1.0, double 1.0, double 9.0, double 2.0, double 2.0], ptr %vals
  store [5 x double] zeroinitializer, ptr %dvals
  call void (...) @__enzyme_autodiff(ptr @outer, metadata !"enzyme_dup", ptr %diag, ptr %ddiag, metadata !"enzyme_const", ptr %rows, metadata !"enzyme_dup", ptr %vals, ptr %dvals, metadata !"enzyme_const", i32 2)
  %v0 = load double, ptr %dvals
  %v1p = getelementptr double, ptr %dvals, i64 1
  %v1 = load double, ptr %v1p
  %v3p = getelementptr double, ptr %dvals, i64 3
  %v3 = load double, ptr %v3p
  %v4p = getelementptr double, ptr %dvals, i64 4
  %v4 = load double, ptr %v4p
  %r = load double, ptr %diag
  call i32 (ptr, ...) @printf(ptr @fmt, double %r, double %v0, double %v1, double %v3, double %v4)
  ret i32 0
}

; EXEC: 0.750000 -0.250000 -0.250000 -0.062500 -0.062500

; CHECK-LABEL: define internal void @diffeprecalc(
; CHECK: %[[STRIDE_PTR:.+]] = getelementptr inbounds i32, ptr %{{.+}}, i64 %iv
; CHECK-NEXT: %stride = load i32, ptr %[[STRIDE_PTR]]
; CHECK: %off = sext i32 %stride to i64
; CHECK: %"curnext'ipg" = getelementptr inbounds double, ptr %{{.+}}, i64 %off
