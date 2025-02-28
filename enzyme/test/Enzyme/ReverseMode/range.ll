; RUN: if [ %llvmver -ge 19 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5) personality ptr null {
"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$15try_allocate_in17hd1f0620d66c47f1bE.exit":
  %6 = call i32 @bar(ptr %0)
  ret void
}

define void @caller() {
  tail call void (...) @__enzyme_autodiff(ptr @foo, metadata !"enzyme_dup", ptr null, ptr null, metadata !"enzyme_const", ptr null, metadata !"enzyme_const", ptr null, metadata !"enzyme_const", ptr null, metadata !"enzyme_const", ptr null, metadata !"enzyme_dup", ptr null, ptr null)
  ret void
}

declare void @__enzyme_autodiff(...)

define range(i32 0, 2) i32 @baz(ptr %0) personality ptr null {
  ret i32 0
}

define i32 @bar(ptr %0) personality ptr null {
  %2 = call i32 @baz(ptr %0)
  ret i32 0
}

; uselistorder directives
uselistorder ptr null, { 11, 12, 0, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 10 }

; We verify that the range(i32 0, 2) metadata is dropped.

; CHECK: define internal {{(dso_local )?}}void @diffefoo(

; CHECK: define internal {{(dso_local )?}}void @diffebar(

; CHECK: define internal {{(dso_local )?}}void @diffebaz(

