; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=halo -S -o /dev/null | FileCheck %s

; A copy of 10000 REAL*8 values (a halo exchange or buffer move) has no
; floating-point operation to infer the type from; the run of the dummies'
; 80000 bytes types every element, far beyond -enzyme-max-type-offset.

define void @halo(ptr "enzyme_type"="{[-1]:Pointer, [-1,0+1*80000]:Float@double}" %dst, ptr "enzyme_type"="{[-1]:Pointer, [-1,0+1*80000]:Float@double}" %src) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sp = getelementptr inbounds i64, ptr %src, i64 %i
  %dp = getelementptr inbounds i64, ptr %dst, i64 %i
  %v = load i64, ptr %sp
  store i64 %v, ptr %dp
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 10000
  br i1 %done, label %exit, label %loop

exit:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 80000, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

; CHECK: halo - {} |{[-1]:Pointer, [-1,0+1*80000]:Float@double}:{} {[-1]:Pointer, [-1,0+1*80000]:Float@double}:{}
; CHECK: %sp = getelementptr inbounds i64, ptr %src, i64 %i: {[-1]:Pointer, [-1,0+1*8]:Float@double}
; CHECK: %dp = getelementptr inbounds i64, ptr %dst, i64 %i: {[-1]:Pointer, [-1,0+1*8]:Float@double}
; CHECK: %v = load i64, ptr %sp, align 4: {[-1]:Float@double}
