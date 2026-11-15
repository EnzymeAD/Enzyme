; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s; fi

; A loop counting down while a signed `i >= m` holds. The bound %m may be
; negative, so the trip count is only known when the comparison is treated
; as signed.

define double @prod(ptr %x, i64 %n, i64 %m) {
entry:
  %guard = icmp sge i64 %n, %m
  br i1 %guard, label %loop, label %exit

loop:
  %i = phi i64 [ %n, %entry ], [ %i.next, %loop ]
  %acc = phi double [ 1.000000e+00, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %p
  %acc.next = fmul double %acc, %v
  %i.next = add nsw i64 %i, -1
  %cmp = icmp sge i64 %i.next, %m
  br i1 %cmp, label %loop, label %exit

exit:
  %r = phi double [ 1.000000e+00, %entry ], [ %acc.next, %loop ]
  ret double %r
}

declare void @__enzyme_autodiff(...)

define void @dprod(ptr %x, ptr %dx, i64 %n, i64 %m) {
entry:
  call void (...) @__enzyme_autodiff(ptr @prod, ptr %x, ptr %dx, i64 %n, i64 %m)
  ret void
}

; The cache holds one entry per iteration, n - m + 1 of them, and the
; reverse loop starts from the last one. Treating the comparison as unsigned
; gave a trip count of 0: an 8-byte cache and a single reverse iteration.

; CHECK: define internal void @diffeprod(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n, i64 %m, double %differeturn)
; CHECK: loop.preheader:
; CHECK-NEXT:   %[[NM1:.+]] = add {{.*}}i64 %n, -1
; CHECK-NEXT:   %[[MM1:.+]] = add {{.*}}i64 %m, -1
; CHECK-NEXT:   %[[SMIN:.+]] = call i64 @llvm.smin.i64(i64 %[[MM1]], i64 %[[NM1]])
; CHECK-NEXT:   %[[CNT:.+]] = sub {{.*}}i64 %[[NM1]], %[[SMIN]]
; CHECK-NEXT:   %[[TRIPS:.+]] = add {{.*}}i64 %[[CNT]], 1
; CHECK-NEXT:   %[[mallocsize:.+]] = mul {{.*}}i64 %[[TRIPS]], 8
; CHECK-NEXT:   %[[acc_malloccache:.+]] = tail call noalias nonnull ptr @malloc(i64 %[[mallocsize]])

; CHECK: invertloop:
; CHECK:   %"iv'ac.0" = phi i64 [ %[[START:.+]], %invertexit.loopexit ], [ %{{.+}}, %incinvertloop ]

; CHECK: invertexit.loopexit:
; CHECK-NEXT:   %[[UN1:.+]] = add {{.*}}i64 %n, -1
; CHECK-NEXT:   %[[UM1:.+]] = add {{.*}}i64 %m, -1
; CHECK-NEXT:   %[[USMIN:.+]] = call i64 @llvm.smin.i64(i64 %[[UM1]], i64 %[[UN1]])
; CHECK-NEXT:   %[[START]] = sub {{.*}}i64 %[[UN1]], %[[USMIN]]
