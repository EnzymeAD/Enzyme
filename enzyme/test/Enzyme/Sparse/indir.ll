; RUN: %opt < %s %loadEnzyme -enzyme -instcombine -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,instcombine" -S | FileCheck %s

declare dso_local double @_Z5loadel(i64 %l)

declare dso_local void @_Z6storerld(i64 %l, double %v)

; Function Attrs: uwtable mustprogress
define dso_local double @_Z4testv() {
  %1 = call double* @_Z16__enzyme_todensePvS_(i8* bitcast (double (i64)* @_Z5loadel to i8*), i8* bitcast (void (i64, double)* @_Z6storerld to i8*))
  br label %3

2:                                                ; preds = %3
  ret double %12

3:                                                ; preds = %0, %3
  %4 = phi i64 [ 0, %0 ], [ %13, %3 ]
  %5 = phi double [ 0.000000e+00, %0 ], [ %12, %3 ]
  %6 = mul nuw nsw i64 %4, 1000
  %7 = getelementptr inbounds double, double* %1, i64 %6
  %8 = load double, double* %7, align 8
  %9 = fadd double %5, %8
  %10 = getelementptr inbounds double, double* %7, i64 1
  %11 = load double, double* %10, align 8
  %12 = fadd double %9, %11
  %13 = add nuw nsw i64 %4, 1
  %14 = icmp eq i64 %13, 10000
  br i1 %14, label %2, label %3
}

declare dso_local double* @_Z16__enzyme_todensePvS_(i8*, i8*) 

; CHECK: define dso_local double @_Z4testv() {
; CHECK:   %3 = phi i64 [ 0, %0 ], [ %11, %2 ]
; CHECK-NEXT:   %4 = phi double [ 0.000000e+00, %0 ], [ %10, %2 ]
; CHECK-NEXT:   %5 = mul nuw i64 %3, 8000
; CHECK-NEXT:   %6 = call double @_Z5loadel(i64 %5)
; CHECK-NEXT:   %7 = fadd double %4, %6
; CHECK-NEXT:   %8 = or i64 %5, 8
; CHECK-NEXT:   %9 = call double @_Z5loadel(i64 %8)
; CHECK-NEXT:   %10 = fadd double %7, %9
; CHECK-NEXT:   %11 = add nuw nsw i64 %3, 1
; CHECK-NEXT:   %12 = icmp eq i64 %11, 10000
; CHECK-NEXT:   br i1 %12, label %1, label %2
