; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S | FileCheck %s

declare noalias double** @ijl_alloc_array_1d() inaccessiblememonly 

define double @f(double** %i2, double %arg) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb3, %bb
  %i5 = phi i64 [ 0, %bb ], [ %i16, %bb3 ]
  %i4 = phi double** [ %i2, %bb ], [ %i6, %bb3 ]
  %i6 = call noalias nonnull double** @ijl_alloc_array_1d()
  %i9 = load double*, double** %i4, align 8
  %i10 = load double, double* %i9, align 8
  %i14 = load double*, double** %i6, align 8
  store double %arg, double* %i14, align 8
  %i16 = add i64 %i5, 1
  %i15 = icmp eq i64 %i5, 2
  br i1 %i15, label %bb17, label %bb3

bb17:                                             ; preds = %bb3
  ret double %i10
}

; CHECK: double** %i2: icv:0
; CHECK-NEXT: double %arg: icv:0
; CHECK-NEXT: bb
; CHECK-NEXT:   br label %bb3: icv:1 ici:1
; CHECK-NEXT: bb3
; CHECK-NEXT:   %i5 = phi i64 [ 0, %bb ], [ %i16, %bb3 ]: icv:1 ici:1
; CHECK-NEXT:   %i4 = phi double** [ %i2, %bb ], [ %i6, %bb3 ]: icv:0 ici:1
; CHECK-NEXT:   %i6 = call noalias nonnull double** @ijl_alloc_array_1d(): icv:0 ici:1
; CHECK-NEXT:   %i9 = load double*, double** %i4, align 8: icv:0 ici:1
; CHECK-NEXT:   %i10 = load double, double* %i9, align 8: icv:0 ici:0
; CHECK-NEXT:   %i14 = load double*, double** %i6, align 8: icv:0 ici:1
; CHECK-NEXT:   store double %arg, double* %i14, align 8: icv:1 ici:0
; CHECK-NEXT:   %i16 = add i64 %i5, 1: icv:1 ici:1
; CHECK-NEXT:   %i15 = icmp eq i64 %i5, 2: icv:1 ici:1
; CHECK-NEXT:   br i1 %i15, label %bb17, label %bb3: icv:1 ici:1
; CHECK-NEXT: bb17
; CHECK-NEXT:   ret double %i10: icv:1 ici:1
