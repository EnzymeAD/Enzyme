; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -S | FileCheck %s

declare dso_local double @_Z5loadel(i64 %l)

declare dso_local void @_Z6storerld(i64 %l, double %v)

; Function Attrs: uwtable mustprogress
define dso_local double @_Z4testv() {
entry:
  %td = call double* @_Z16__enzyme_todensePvS_(i8* bitcast (double (i64)* @_Z5loadel to i8*), i8* bitcast (void (i64, double)* @_Z6storerld to i8*))
  br label %loop

rb:                                                ; preds = %3
  ret double %a12

loop:                                                ; preds = %0, %3
  %a4 = phi i64 [ 0, %entry ], [ %a13, %loop ]
  %a5 = phi double [ 0.000000e+00, %entry ], [ %a12, %loop ]
  %a6 = mul nuw nsw i64 %a4, 1000
  %a7 = getelementptr inbounds double, double* %td, i64 %a6
  %a8 = load double, double* %a7, align 8
  %a9 = fadd double %a5, %a8
  %a10 = getelementptr inbounds double, double* %a7, i64 1
  %a11 = load double, double* %a10, align 8
  %a12 = fadd double %a9, %a11
  %a13 = add nuw nsw i64 %a4, 1
  %a14 = icmp eq i64 %a13, 10000
  br i1 %a14, label %rb, label %loop
}

declare dso_local double* @_Z16__enzyme_todensePvS_(i8*, i8*) 

; CHECK: define dso_local double @_Z4testv() {
; CHECK:   %a4 = phi i64 [ 0, %entry ], [ %a13, %loop ]
; CHECK-NEXT:   %a5 = phi double [ 0.000000e+00, %entry ], [ %a12, %loop ]
; CHECK-NEXT:   %a6 = mul nuw nsw i64 %a4, 1000
; CHECK-NEXT:   %0 = mul nuw nsw i64 %a6, 8
; CHECK-NEXT:   %1 = call double @_Z5loadel(i64 %0)
; CHECK-NEXT:   %a9 = fadd double %a5, %1
; CHECK-NEXT:   %2 = add i64 %0, 8
; CHECK-NEXT:   %3 = call double @_Z5loadel(i64 %2)
; CHECK-NEXT:   %a12 = fadd double %a9, %3
; CHECK-NEXT:   %a13 = add nuw nsw i64 %a4, 1
; CHECK-NEXT:   %a14 = icmp eq i64 %a13, 10000
; CHECK-NEXT:   br i1 %a14, label %rb, label %loop
