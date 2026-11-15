; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s

; When zeroing the float bytes of a mixed-type induction PHI in
; GradientUtils::invertPointerM, or caching a PHI in
; CacheUtility::storeInstructionInCache, the insertion point must advance past
; the entire leading PHI group, including zero-incoming shadow placeholder PHIs
; that have not been populated yet.

@enzyme_dup = external global i32

declare ptr @__enzyme_augmentfwd(...)

define i32 @eig_step(ptr %data, i1 %keep_going) {
entry:
  %initial = load i32, ptr %data, align 4
  br label %loop

loop:
  %cur = phi i32 [ %next, %latch ], [ %initial, %entry ]
  %prev = phi i32 [ %mux, %latch ], [ 0, %entry ]
  %mux = select i1 false, i32 %prev, i32 %cur
  br i1 %keep_going, label %latch, label %exit

exit:
  call fastcc void @sink(i32 %prev)
  ret i32 0

latch:
  %next = add i32 %cur, 1
  br label %loop
}

define fastcc void @sink(i32 %val) {
entry:
  store double 0.000000e+00, ptr null, align 8
  ret void
}

define void @wrap(ptr %a, ptr %out) {
entry:
  %step = call i32 @eig_step(ptr %a, i1 false)
  %x = load double, ptr %a, align 8
  %sq = fmul double %x, %x
  store double %sq, ptr %out, align 8
  ret void
}

define ptr @caller(ptr %a, ptr %da, ptr %out, ptr %dout) {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @wrap, i32 %dup, ptr %a, ptr %da, i32 %dup, ptr %out, ptr %dout)
  ret ptr %tape
}

; CHECK-LABEL: define internal {{.*}}@augmented_eig_step(
; CHECK-NEXT: entry:
; CHECK: %[[ALLOC:[0-9]+]] = alloca i64, align 8
; CHECK: loop:
; CHECK-NEXT:   %iv = phi i64
; CHECK-NEXT:   phi i32
; CHECK-NEXT:   %prev = phi i32
; CHECK-NEXT:   store i64 %iv, ptr %[[ALLOC]]
; CHECK: store i32 %prev, ptr
; CHECK-NOT: = alloca 
; CHECK: ret
