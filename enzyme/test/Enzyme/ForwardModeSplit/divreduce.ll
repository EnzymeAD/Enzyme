; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse-memssa -instsimplify -correlated-propagation -adce -S | FileCheck %s

; Function Attrs: norecurse nounwind readonly uwtable
define double @alldiv(double* nocapture readonly %A, i64 %N, double %start) {
entry:
  br label %loop

loop:                                                ; preds = %9, %5
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %reduce = phi double [ %start, %entry ], [ %div, %loop ]
  %gep = getelementptr inbounds double, double* %A, i64 %i
  %ld = load double, double* %gep, align 8, !tbaa !2
  %div = fdiv double %reduce, %ld
  %next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %next, %N
  br i1 %cmp, label %end, label %loop

end:                                                ; preds = %9, %3
  ret double %div
}

define double @alldiv2(double* nocapture readonly %A, i64 %N) {
entry:
  br label %loop

loop:                                                ; preds = %9, %5
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %reduce = phi double [ 2.000000e+00, %entry ], [ %div, %loop ]
  %gep = getelementptr inbounds double, double* %A, i64 %i
  %ld = load double, double* %gep, align 8, !tbaa !2
  %div = fdiv double %reduce, %ld
  %next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %next, %N
  br i1 %cmp, label %end, label %loop

end:                                                ; preds = %9, %3
  ret double %div
}

; Function Attrs: nounwind uwtable
define double @main(double* %A, double* %dA, i64 %N, double %start) {
  %r = call double (...) @__enzyme_fwdsplit(i8* bitcast (double (double*, i64, double)* @alldiv to i8*), metadata !"enzyme_nofree", double* %A, double* %dA, i64 %N, double %start, double 1.0, i8* null)
  %r2 = call double (...) @__enzyme_fwdsplit2(i8* bitcast (double (double*, i64)* @alldiv2 to i8*), metadata !"enzyme_nofree", double* %A, double* %dA, i64 %N, i8* null)
  ret double %r
}

declare double @__enzyme_fwdsplit(...)
declare double @__enzyme_fwdsplit2(...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Ubuntu clang version 10.0.1-++20200809072545+ef32c611aa2-1~exp1~20200809173142.193"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}


; CHECK: define internal double @fwddiffealldiv(double* nocapture readonly %A, double* nocapture %"A'", i64 %N, double %start, double %"start'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { double*, double* }*
; CHECK-NEXT:   %truetape = load { double*, double* }, { double*, double* }* %0
; CHECK-DAG:   %[[i1:.+]] = extractvalue { double*, double* } %truetape, 0
; CHECK-DAG:   %[[i2:.+]] = extractvalue { double*, double* } %truetape, 1
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-DAG:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-DAG:   %[[dreduce:.+]] = phi {{(fast )?}}double [ %"start'", %entry ], [ %[[i10:.+]], %loop ]
; CHECK-NEXT:   %[[i3:.+]] = getelementptr inbounds double, double* %[[i1]], i64 %iv
; CHECK-NEXT:   %reduce = load double, double* %[[i3]], align 8, !invariant.group !{{[0-9]+}}
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %"gep'ipg" = getelementptr inbounds double, double* %"A'", i64 %iv
; CHECK-NEXT:   %[[i5:.+]] = load double, double* %"gep'ipg", align 8
; CHECK-NEXT:   %[[i4:.+]] = getelementptr inbounds double, double* %[[i2]], i64 %iv
; CHECK-NEXT:   %ld = load double, double* %[[i4]], align 8, !invariant.group !{{[0-9]+}}
; CHECK-NEXT:   %[[i6:.+]] = fmul fast double %[[dreduce]], %ld
; CHECK-NEXT:   %[[i7:.+]] = fmul fast double %[[i5]], %reduce 
; CHECK-NEXT:   %[[i8:.+]] = fsub fast double %[[i6]], %[[i7]]
; CHECK-NEXT:   %[[i9:.+]] = fmul fast double %ld, %ld
; CHECK-NEXT:   %[[i10]] = fdiv fast double %[[i8]], %[[i9]]
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, %N
; CHECK-NEXT:   br i1 %cmp, label %end, label %loop

; CHECK: end:                                              ; preds = %loop
; CHECK-NEXT:   ret double %[[i10]]
; CHECK-NEXT: }


; CHECK: define internal double @fwddiffealldiv2(double* nocapture readonly %A, double* nocapture %"A'", i64 %N, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { double*, double* }*
; CHECK-NEXT:   %truetape = load { double*, double* }, { double*, double* }* %0
; CHECK-DAG:   %[[i1:.+]] = extractvalue { double*, double* } %truetape, 0
; CHECK-DAG:   %[[i2:.+]] = extractvalue { double*, double* } %truetape, 1
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-DAg:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-DAG:   %[[dreduce:.+]] = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %[[i10:.+]], %loop ]
; CHECK-NEXT:   %[[i3:.+]] = getelementptr inbounds double, double* %[[i1]], i64 %iv
; CHECK-NEXT:   %reduce = load double, double* %[[i3]], align 8, !invariant.group !{{[0-9]+}}
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %"gep'ipg" = getelementptr inbounds double, double* %"A'", i64 %iv
; CHECK-NEXT:   %[[i5:.+]] = load double, double* %"gep'ipg", align 8
; CHECK-NEXT:   %[[i4:.+]] = getelementptr inbounds double, double* %[[i2]], i64 %iv
; CHECK-NEXT:   %ld = load double, double* %[[i4]], align 8, !invariant.group !{{[0-9]+}}
; CHECK-NEXT:   %[[i6:.+]] = fmul fast double %[[dreduce]], %ld
; CHECK-NEXT:   %[[i7:.+]] = fmul fast double %[[i5]], %reduce 
; CHECK-NEXT:   %[[i8:.+]] = fsub fast double %[[i6]], %[[i7]]
; CHECK-NEXT:   %[[i9:.+]] = fmul fast double %ld, %ld
; CHECK-NEXT:   %[[i10]] = fdiv fast double %[[i8]], %[[i9]]
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, %N
; CHECK-NEXT:   br i1 %cmp, label %end, label %loop

; CHECK: end:                                              ; preds = %loop
; CHECK-NEXT:   ret double %[[i10]]
; CHECK-NEXT: }
