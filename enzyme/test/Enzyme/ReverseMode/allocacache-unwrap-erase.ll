; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-global-activity -S | FileCheck %s; fi

; The reverse pass looks up %ch, a load from the alloca %name that is cached
; by copying the whole alloca in %scan.ph, while unwrapping a phi. That
; unwrap creates blocks which it erases again when it fails. The cache
; pointer computed in such a block was recorded among the instructions that
; create the cache, so erasing it tripped an AssertingVH ("An asserting value
; handle still pointed to this value!"). Reduced from MITgcm.

@enzyme_const = external global i32
@fld = global [16 x double] zeroinitializer, !enzyme_shadow !0
@dfld = global [16 x double] zeroinitializer

declare i8 @llvm.ucmp.i8.i8(i8, i8)
declare void @__enzyme_autodiff(...)

define void @f(i64 %step, ptr %unused) {
entry:
  %name = alloca [16 x i8], align 1
  %skip = icmp eq i64 %step, 0
  br i1 %skip, label %ret, label %scan.ph

scan.ph:
  %cmp0 = call i8 @llvm.ucmp.i8.i8(i8 0, i8 0)
  br label %scan

scan:
  %state = phi i8 [ %cmp0, %scan.ph ], [ 0, %latch ]
  %i = phi i64 [ 0, %scan.ph ], [ %i.next, %latch ]
  %blank = icmp eq i8 %state, 0
  br i1 %blank, label %check, label %latch

check:
  %p = getelementptr [1 x i8], ptr %name, i64 %i
  %ch = load i8, ptr %p, align 1
  %cmp = call i8 @llvm.ucmp.i8.i8(i8 %ch, i8 0)
  br label %latch

latch:
  %last = phi i8 [ %ch, %check ], [ 0, %scan ]
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i, 15
  br i1 %done, label %exit, label %scan

exit:
  %empty = icmp eq i8 %last, 0
  br i1 %empty, label %merge, label %clobber

clobber:
  store i32 0, ptr %name, align 4
  br label %merge

merge:
  %n = phi i64 [ 0, %clobber ], [ 1, %exit ]
  br label %loop

loop:
  %k = phi i64 [ %n, %merge ], [ %k.next, %loop ]
  %fp = getelementptr [8 x i8], ptr @fld, i64 %n
  store double 0.000000e+00, ptr %fp, align 8
  %k.next = add i64 %k, -1
  %more = icmp sgt i64 %k, 0
  br i1 %more, label %loop, label %ret

ret:
  ret void
}

define void @test() {
  call void (...) @__enzyme_autodiff(ptr @f, ptr @enzyme_const, i64 0, ptr @enzyme_const, ptr null)
  ret void
}

!0 = !{ptr @dfld}

; CHECK-LABEL: define internal void @diffef(
; CHECK: scan.ph:
; CHECK: %ch_malloccache = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) ptr @malloc(i64 16)
; CHECK: invertscan.ph:
