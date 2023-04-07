; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -activity-analysis-inactive-args -o /dev/null | FileCheck %s

declare noalias double** @ijl_alloc_array_1d() 

define double @f(double %arg) {
bb:
  %i2 = call noalias nonnull double** @ijl_alloc_array_1d()
  br label %bb3

bb3:                                              ; preds = %bb3, %bb
  %i5 = phi i64 [ 0, %bb ], [ %i16, %bb3 ]
  %i4 = phi double** [ %i2, %bb ], [ %i6, %bb3 ]
  %i9 = load double*, double** %i4, align 8
  %i10 = load double, double* %i9, align 8
  %i11 = fadd double %i10, %arg
  %i6 = call noalias nonnull double** @ijl_alloc_array_1d()
  %i14 = load double*, double** %i6, align 8
  store double %i11, double* %i14, align 8
  %i16 = add i64 %i5, 1
  %i15 = icmp eq i64 %i5, 2
  br i1 %i15, label %bb17, label %bb3

bb17:                                             ; preds = %bb3
  ret double %i11
}
