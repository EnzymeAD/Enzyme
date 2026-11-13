; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare i8* @__enzyme_augmentfwd(i8*, ...)
declare void @__enzyme_reverse(i8*, ...)

; Exercises typed-pointer pack/unpack in cacheForReverse when invariant loop
; pruning collapses a 2-chunk cache (double**) to a flat 1-chunk cache (double*).
define void @triangular_attn_typed(double* noalias %out, double* noalias %q, double* noalias %k) {
entry:
  br label %loop.t

loop.t:
  %t = phi i64 [ 0, %entry ], [ %t.next, %loop.t.latch ]
  %t.h = mul nuw nsw i64 %t, 3
  %t.T = mul nuw nsw i64 %t, 4
  %t.lim = add nuw nsw i64 %t, 1
  br label %loop.t2

loop.t2:
  %t2 = phi i64 [ 0, %loop.t ], [ %t2.next, %loop.t2.latch ]
  %t2.h = mul nuw nsw i64 %t2, 3
  br label %loop.h

loop.h:
  %h = phi i64 [ 0, %loop.t2 ], [ %h.next, %loop.h ]
  %acc = phi double [ 0.0, %loop.t2 ], [ %acc.next, %loop.h ]
  %q.idx = add nuw nsw i64 %t.h, %h
  %q.ptr = getelementptr inbounds double, double* %q, i64 %q.idx
  %q.val = load double, double* %q.ptr, align 8
  %k.idx = add nuw nsw i64 %t2.h, %h
  %k.ptr = getelementptr inbounds double, double* %k, i64 %k.idx
  %k.val = load double, double* %k.ptr, align 8
  %prod = fmul double %q.val, %k.val
  %acc.next = fadd double %acc, %prod
  %h.next = add nuw nsw i64 %h, 1
  %h.cmp = icmp eq i64 %h.next, 3
  br i1 %h.cmp, label %loop.t2.latch, label %loop.h

loop.t2.latch:
  %out.idx = add nuw nsw i64 %t.T, %t2
  %out.ptr = getelementptr inbounds double, double* %out, i64 %out.idx
  store double %acc.next, double* %out.ptr, align 8
  %t2.next = add nuw nsw i64 %t2, 1
  %t2.cmp = icmp eq i64 %t2.next, %t.lim
  br i1 %t2.cmp, label %loop.t.latch, label %loop.t2

loop.t.latch:
  %t.next = add nuw nsw i64 %t, 1
  %t.cmp = icmp eq i64 %t.next, 4
  br i1 %t.cmp, label %exit, label %loop.t

exit:
  ret void
}

define void @caller(double* %out, double* %dout, double* %q, double* %dq, double* %k, double* %dk) {
entry:
  %tape = call i8* (i8*, ...) @__enzyme_augmentfwd(i8* bitcast (void (double*, double*, double*)* @triangular_attn_typed to i8*), metadata !"enzyme_dup", double* %out, double* %dout, metadata !"enzyme_dup", double* %q, double* %dq, metadata !"enzyme_dup", double* %k, double* %dk)
  call void (i8*, ...) @__enzyme_reverse(i8* bitcast (void (double*, double*, double*)* @triangular_attn_typed to i8*), metadata !"enzyme_dup", double* %out, double* %dout, metadata !"enzyme_dup", double* %q, double* %dq, metadata !"enzyme_dup", double* %k, double* %dk, i8* %tape)
  ret void
}

; CHECK-LABEL: define internal i8* @augmented_triangular_attn_typed(
; CHECK: entry:
; CHECK-DAG: @malloc(i64 96)
; CHECK-DAG: @malloc(i64 32)
; CHECK-NOT: @malloc(i64 32)
; CHECK: loop.t:
; CHECK: @malloc(i64
; CHECK-NOT: @malloc(i64
; CHECK: ret i8*

; CHECK-LABEL: define internal void @diffetriangular_attn_typed(
; CHECK-NOT: @malloc(
; CHECK: ret void
