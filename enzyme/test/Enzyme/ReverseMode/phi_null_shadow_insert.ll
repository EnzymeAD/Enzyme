; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-loose-types=1 -enzyme-strict-aliasing=0 -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-loose-types=1 -enzyme-strict-aliasing=0 -S | FileCheck %s

declare double @__enzyme_autodiff(...)

define void @leaf(i32 %val) {
entry:
  store double 0.000000e+00, ptr null, align 8
  ret void
}

define void @loops(ptr %arg, i1 %cond) {
entry:
  %init = load i32, ptr %arg, align 4
  br label %header

header:
  %cur = phi i32 [ %next, %latch ], [ %init, %entry ]
  %prev = phi i32 [ %cur, %latch ], [ 0, %entry ]
  %carried = phi i32 [ %merged, %latch ], [ 0, %entry ]
  br i1 %cond, label %latch, label %body

latch:
  %merged = phi i32 [ %init, %body ], [ %carried, %header ]
  %next = add i32 %cur, 1
  br label %header

body:
  call void @leaf(i32 %carried)
  call void @leaf(i32 %prev)
  br label %latch
}

define double @outer(ptr %arg) {
entry:
  call void @loops(ptr %arg, i1 false)
  ret double 0.000000e+00
}

define double @caller() {
entry:
  %res = tail call double (...) @__enzyme_autodiff(ptr @outer, ptr null, ptr null)
  ret double %res
}

; CHECK-LABEL: define internal void @diffeloops(
; CHECK: header:
; CHECK-NEXT:   %iv = phi i64
; CHECK-NEXT:   %[[P1:.+]] = phi i32
; CHECK-NEXT:   %[[P2:.+]] = phi i32
; CHECK-NEXT:   %[[P3:.+]] = phi i32
; CHECK-NEXT:   %[[P4:.+]] = phi i32
; CHECK-NEXT:   %[[ALLOC:.+]] = alloca i64, align 8
; CHECK-NEXT:   store i64 %iv, ptr %[[ALLOC]]
