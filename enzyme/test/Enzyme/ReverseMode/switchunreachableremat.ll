; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -opaque-pointers -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @lstm_predict(ptr %0) {
  br label %2

2:                                                ; preds = %6, %1
  %3 = tail call i8 @func()
  switch i8 %3, label %4 [
    i8 1, label %5
    i8 0, label %6
  ]

4:                                                ; preds = %2
  unreachable

5:                                                ; preds = %2
  br label %6

6:                                                ; preds = %5, %2
  %7 = phi i32 [ 0, %5 ], [ 0, %2 ]
  %8 = phi i1 [ false, %5 ], [ true, %2 ]
  %9 = call ptr @calloc(i64 0)
  tail call void @free(ptr %9)
  br label %2
}

define fastcc void @_ZN2ad23d_lstm_unsafe_objective17hfdae8b2443372ae4E() {
  tail call void (...) @__enzyme_autodiff_ZN2ad23d_lstm_unsafe_objective17hfdae8b2443372ae4E(ptr @lstm_predict, metadata !"enzyme_const", ptr null)
  ret void
}

declare i8 @func()

declare void @free(ptr nocapture)

declare ptr @calloc(i64)

declare void @__enzyme_autodiff_ZN2ad23d_lstm_unsafe_objective17hfdae8b2443372ae4E(...)

; CHECK: define internal void @diffelstm_predict
