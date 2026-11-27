; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=step_narrow -o /dev/null | FileCheck %s --check-prefix=NARROW; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=step_narrow -S -o /dev/null | FileCheck %s --check-prefix=NARROW
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=step_loop -o /dev/null | FileCheck %s --check-prefix=LOOP; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=step_loop -S -o /dev/null | FileCheck %s --check-prefix=LOOP
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=nextafter_like -o /dev/null | FileCheck %s --check-prefix=BITHACK; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=nextafter_like -S -o /dev/null | FileCheck %s --check-prefix=BITHACK
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=copy_i32 -o /dev/null | FileCheck %s --check-prefix=COPY; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=copy_i32 -S -o /dev/null | FileCheck %s --check-prefix=COPY

target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"

declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.abs.i32(i32, i1)
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

define void @step_narrow(ptr %state, ptr %out) {
entry:
  %p = getelementptr inbounds { double, i32 }, ptr %state, i64 0, i32 1
  %nfe = load i32, ptr %p, align 4
  %inc = add i32 %nfe, 1
  %q = sdiv i32 100, %inc
  %m1 = call i32 @llvm.smax.i32(i32 %q, i32 0)
  %m2 = call i32 @llvm.smin.i32(i32 %m1, i32 50)
  %m3 = call i32 @llvm.umax.i32(i32 %m2, i32 1)
  %m4 = call i32 @llvm.umin.i32(i32 %m3, i32 49)
  %ab = call i32 @llvm.abs.i32(i32 %m4, i1 false)
  %ov = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %ab, i32 1)
  %v = extractvalue { i32, i1 } %ov, 0
  store i32 %v, ptr %p, align 4
  %x = load double, ptr %state, align 8
  %y = fmul double %x, %x
  store double %y, ptr %out, align 8
  ret void
}

define void @step_loop(ptr %state, ptr %out) {
entry:
  %p = getelementptr inbounds { double, i32 }, ptr %state, i64 0, i32 1
  %nfe0 = load i32, ptr %p, align 4
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %cnt = phi i32 [ %nfe0, %entry ], [ %inc, %loop ]
  %inc = add i32 %cnt, 1
  %inext = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inext, 3
  br i1 %done, label %exit, label %loop

exit:
  store i32 %inc, ptr %p, align 4
  %x = load double, ptr %state, align 8
  %y = fmul double %x, %x
  store double %y, ptr %out, align 8
  ret void
}

define float @nextafter_like(float %x) {
entry:
  %i = bitcast float %x to i32
  %j = add i32 %i, 1
  %y = bitcast i32 %j to float
  ret float %y
}

define void @copy_i32(ptr %src, ptr %dst) {
entry:
  %v = load i32, ptr %src, align 4
  store i32 %v, ptr %dst, align 4
  ret void
}

; NARROW: entry
; NARROW-NEXT:   %p = getelementptr inbounds { double, i32 }, ptr %state, i64 0, i32 1: icv:0 ici:1
; NARROW-NEXT:   %nfe = load i32, ptr %p, align 4: icv:1 ici:1
; NARROW-NEXT:   %inc = add i32 %nfe, 1: icv:1 ici:1
; NARROW-NEXT:   %q = sdiv i32 100, %inc: icv:1 ici:1
; NARROW-NEXT:   %m1 = call i32 @llvm.smax.i32(i32 %q, i32 0): icv:1 ici:1
; NARROW-NEXT:   %m2 = call i32 @llvm.smin.i32(i32 %m1, i32 50): icv:1 ici:1
; NARROW-NEXT:   %m3 = call i32 @llvm.umax.i32(i32 %m2, i32 1): icv:1 ici:1
; NARROW-NEXT:   %m4 = call i32 @llvm.umin.i32(i32 %m3, i32 49): icv:1 ici:1
; NARROW-NEXT:   %ab = call i32 @llvm.abs.i32(i32 %m4, i1 false): icv:1 ici:1
; NARROW-NEXT:   %ov = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %ab, i32 1): icv:1 ici:1
; NARROW-NEXT:   %v = extractvalue { i32, i1 } %ov, 0: icv:1 ici:1
; NARROW-NEXT:   store i32 %v, ptr %p, align 4: icv:1 ici:1
; NARROW-NEXT:   %x = load double, ptr %state, align 8: icv:0 ici:0
; NARROW-NEXT:   %y = fmul double %x, %x: icv:0 ici:0
; NARROW-NEXT:   store double %y, ptr %out, align 8: icv:1 ici:0

; LOOP: entry
; LOOP-NEXT:   %p = getelementptr inbounds { double, i32 }, ptr %state, i64 0, i32 1: icv:0 ici:1
; LOOP-NEXT:   %nfe0 = load i32, ptr %p, align 4: icv:1 ici:1
; LOOP-NEXT:   br label %loop: icv:1 ici:1
; LOOP-NEXT: loop
; LOOP-NEXT:   %i = phi i32 [ 0, %entry ], [ %inext, %loop ]: icv:1 ici:1
; LOOP-NEXT:   %cnt = phi i32 [ %nfe0, %entry ], [ %inc, %loop ]: icv:1 ici:1
; LOOP-NEXT:   %inc = add i32 %cnt, 1: icv:1 ici:1
; LOOP-NEXT:   %inext = add nuw nsw i32 %i, 1: icv:1 ici:1
; LOOP-NEXT:   %done = icmp eq i32 %inext, 3: icv:1 ici:1
; LOOP-NEXT:   br i1 %done, label %exit, label %loop: icv:1 ici:1
; LOOP-NEXT: exit
; LOOP-NEXT:   store i32 %inc, ptr %p, align 4: icv:1 ici:1
; LOOP-NEXT:   %x = load double, ptr %state, align 8: icv:0 ici:0

; BITHACK: entry
; BITHACK-NEXT:   %i = bitcast float %x to i32: icv:0 ici:0
; BITHACK-NEXT:   %j = add i32 %i, 1: icv:0 ici:0
; BITHACK-NEXT:   %y = bitcast i32 %j to float: icv:0 ici:0

; COPY: entry
; COPY-NEXT:   %v = load i32, ptr %src, align 4: icv:0 ici:0
; COPY-NEXT:   store i32 %v, ptr %dst, align 4: icv:1 ici:0
