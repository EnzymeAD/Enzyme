; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false  -enzyme -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg)" -S | FileCheck %s

define internal fastcc nonnull double* @julia_mygetindex_2341(double* %a12) {
top:
  %a13 = load double, double* %a12, align 8
  %a22 = call noalias nonnull double* @malloc(i64 24) 
  br label %L8

L8:                                               ; preds = %idxend2, %idxend
  %a21 = phi i64 [ 0, %top ], [ %a24, %L8 ]
  %a23 = getelementptr inbounds double, double* %a22, i64 %a21
  store double %a13, double* %a23, align 8
  %a24 = add nuw nsw i64 %a21, 1
  %.not5 = icmp ne i64 1, %a24
  br i1 %.not5, label %L8, label %L28

L28:                                              ; preds = %idxend2
  ret double* %a22
}

define internal fastcc nonnull double* @mydiag(double* %a12) {
top:
  %a4 = call fastcc nonnull double* @julia_mygetindex_2341(double* %a12)
  ret double* %a4
}

declare void @__enzyme_autodiff(...) 

define void @caller(double* %x, double* %dx) {
entry:
    call void (...) @__enzyme_autodiff(double (double*)* @f, metadata !"enzyme_dup", double* %x, double* %dx)
    ret void
}

define double @f(double* %a13) {
top:
  %b13 = call fastcc nonnull double* @mydiag(double* %a13)
  %a14 = load double, double* %b13, align 8
  ret double %a14
}

declare double* @malloc(i64) local_unnamed_addr

; CHECK: define internal fastcc void @diffejulia_mygetindex_2341(double* %a12, double* %"a12'", { double*, double* } %tapeArg)
; CHECK-NEXT: top:
; CHECK-NEXT:   %"a22'mi" = extractvalue { double*, double* } %tapeArg, 0
; CHECK-NEXT:   %a22 = extractvalue { double*, double* } %tapeArg, 1
; CHECK-NEXT:   br label %L8

; CHECK: L8:                                               ; preds = %L8, %top
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %L8 ], [ 0, %top ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %.not5 = icmp ne i64 %iv.next, 1
; CHECK-NEXT:   br i1 %.not5, label %L8, label %invertL8

; CHECK: inverttop:                                        ; preds = %invertL8
; CHECK-NEXT:   %[[a0:.+]] = bitcast double* %"a22'mi" to i8*
; CHECK-NEXT:   call void @free(i8* nonnull %[[a0]])
; CHECK-NEXT:   %[[a1:.+]] = bitcast double* %a22 to i8*
; CHECK-NEXT:   call void @free(i8* %[[a1]])
; CHECK-NEXT:   %[[a2:.+]] = load double, double* %"a12'", align 8, !alias.scope !5, !noalias !8
; CHECK-NEXT:   %[[a3:.+]] = fadd fast double %[[a2]], %[[a5:.+]]
; CHECK-NEXT:   store double %[[a3]], double* %"a12'", align 8, !alias.scope !5, !noalias !8
; CHECK-NEXT:   ret void

; CHECK: invertL8:                                         ; preds = %L8, %incinvertL8
; CHECK-NEXT:   %"a13'de.0" = phi double [ %[[a5]], %incinvertL8 ], [ 0.000000e+00, %L8 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[a7:.+]], %incinvertL8 ], [ 0, %L8 ]
; CHECK-NEXT:   %"a23'ipg_unwrap" = getelementptr inbounds double, double* %"a22'mi", i64 %"iv'ac.0"
; CHECK-NEXT:   %[[a4:.+]] = load double, double* %"a23'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a23'ipg_unwrap", align 8
; CHECK-NEXT:   %[[a5]] = fadd fast double %"a13'de.0", %[[a4]]
; CHECK-NEXT:   %[[a6:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[a6]], label %inverttop, label %incinvertL8

; CHECK: incinvertL8:                                      ; preds = %invertL8
; CHECK-NEXT:   %[[a7]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertL8
; CHECK-NEXT: }
