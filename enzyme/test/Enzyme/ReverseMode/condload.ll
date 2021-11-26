; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -instsimplify -correlated-propagation -simplifycfg -adce -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)

; Function Attrs: norecurse nounwind uwtable
define double @alldiv(double* %a, i1 %cmp, i1 %cmp2) {
entry:
  br i1 %cmp, label %mid, label %fin

mid:
  br i1 %cmp2, label %b1, label %b2

b1: 
  %g1 = getelementptr inbounds double, double* %a, i32 32
  %l1 = load double, double* %g1, align 8
  br label %end

b2: 
  %g2 = getelementptr inbounds double, double* %a, i32 64
  %l2 = load double, double* %g2, align 8
  br label %end

end:
  %p = phi double [ %l1, %b1 ], [ %l2, %b2 ]
  %sq = fmul double %p, %p
  br label %fin

fin:
  %res = phi double [ 0.000000e+00, %entry ], [ %sq, %end ]
  ret double %res
}

define void @main(double* %a, double* %da, i1 %N, i1 %N2) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, i1, i1)* @alldiv to i8*), double* nonnull %a, double* nonnull %da, i1 %N, i1 %N2)
  ret void
}

; CHECK: define internal void @diffealldiv(double* %a, double* %"a'", i1 %cmp, i1 %cmp2, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = select {{(fast )?}}i1 %cmp, double %differeturn, double 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %invertend, label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertb1:                                         ; preds = %invertend_phimerge
; CHECK-NEXT:   %"g1'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i32 32
; CHECK-NEXT:   %1 = load double, double* %"g1'ipg_unwrap", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %8
; CHECK-NEXT:   store double %2, double* %"g1'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertb2:                                         ; preds = %invertend_phimerge
; CHECK-NEXT:   %"g2'ipg_unwrap" = getelementptr inbounds double, double* %"a'", i32 64
; CHECK-NEXT:   %3 = load double, double* %"g2'ipg_unwrap", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %7
; CHECK-NEXT:   store double %4, double* %"g2'ipg_unwrap", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertend:                                        ; preds = %entry
; CHECK-NEXT:   br i1 %cmp2, label %invertend_phirc, label %invertend_phirc1

; CHECK: invertend_phirc:                                  ; preds = %invertend
; CHECK-NEXT:   %g1_unwrap = getelementptr inbounds double, double* %a, i32 32
; CHECK-NEXT:   %l1_unwrap = load double, double* %g1_unwrap, align 8
; CHECK-NEXT:   br label %invertend_phimerge

; CHECK: invertend_phirc1:                                 ; preds = %invertend
; CHECK-NEXT:   %g2_unwrap = getelementptr inbounds double, double* %a, i32 64
; CHECK-NEXT:   %l2_unwrap = load double, double* %g2_unwrap, align 8
; CHECK-NEXT:   br label %invertend_phimerge

; CHECK: invertend_phimerge:                               ; preds = %invertend_phirc1, %invertend_phirc
; CHECK-NEXT:   %5 = phi {{(fast )?}}double [ %l1_unwrap, %invertend_phirc ], [ %l2_unwrap, %invertend_phirc1 ]
; CHECK-NEXT:   %m0diffep = fmul fast double %0, %5
; CHECK-NEXT:   %6 = fadd fast double %m0diffep, %m0diffep
; CHECK-NEXT:   %7 = select {{(fast )?}}i1 %cmp2, double 0.000000e+00, double %6
; CHECK-NEXT:   %8 = select {{(fast )?}}i1 %cmp2, double %6, double 0.000000e+00
; CHECK-NEXT:   br i1 %cmp2, label %invertb1, label %invertb2
; CHECK-NEXT: }

