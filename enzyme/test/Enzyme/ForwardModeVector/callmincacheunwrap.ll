; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -S | FileCheck %s

source_filename = "/mnt/pci4/wmdata/Enzyme2/enzyme/test/Integration/ReverseMode/eigensumsqdyn.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZNK5Eigen9EigenBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE4colsEv = comdat any

define void @caller(i8* %a, i8* %b1, i8* %b2, i8* %b3, i8* %c) local_unnamed_addr {
entry:
  call void (i8*, ...) @__enzyme_fwddiff(i8* bitcast (void (double**, i64*)* @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_ to i8*), metadata !"enzyme_width", i64 3 , i8* %a, i8* %b1, i8* %b2, i8* %b3, i8* %c)
  ret void
}

declare void @__enzyme_fwddiff(i8*, ...) local_unnamed_addr

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6matvecPKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEES3_(double** noalias %m_data.i.i.i, i64* noalias %m_rows) #0 {
entry:
  call void @subcall(double** nonnull %m_data.i.i.i, i64* nonnull %m_rows) #3
  store i64 0, i64* %m_rows, align 8
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE4colsEv(i64* %m_cols) local_unnamed_addr #1 comdat align 2 {
entry:
  %tmp.i.i = load i64, i64* %m_cols, align 8, !tbaa !2
  ret i64 %tmp.i.i
}

; Function Attrs: nounwind uwtable
define void @subcall(double** %m_data.i.i.i, i64* %tmp7) local_unnamed_addr #2 {
entry:
  %mat = load double*, double** %m_data.i.i.i, align 8, !tbaa !8
  %cols = call i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE4colsEv(i64* %tmp7) #3
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %call = getelementptr inbounds double, double* %mat, i64 %i
  %ld = load double, double* %call, align 8
  %fmul = fmul double %ld, %ld
  store double %fmul, double* %call, align 8, !tbaa !11
  %inc = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %inc, %cols
  br i1 %exitcond, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret void
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !7, i64 16}
!3 = !{!"_ZTSN5Eigen12DenseStorageIdLin1ELin1ELin1ELi0EEE", !4, i64 0, !7, i64 8, !7, i64 16}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"long", !5, i64 0}
!8 = !{!9, !4, i64 0}
!9 = !{!"_ZTSN5Eigen8internal9evaluatorINS_15PlainObjectBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEEEEE", !4, i64 0, !10, i64 8}
!10 = !{!"_ZTSN5Eigen8internal19variable_if_dynamicIlLin1EEE", !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !5, i64 0}


; CHECK: define internal void @fwddiffe3subcall(double** %m_data.i.i.i, [3 x double**] %"m_data.i.i.i'", i64* %tmp7)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double**] %"m_data.i.i.i'", 0
; CHECK-NEXT:   %"mat'ipl" = load double*, double** %0, align 8
; CHECK-NEXT:   %1 = extractvalue [3 x double**] %"m_data.i.i.i'", 1
; CHECK-NEXT:   %"mat'ipl1" = load double*, double** %1, align 8
; CHECK-NEXT:   %2 = extractvalue [3 x double**] %"m_data.i.i.i'", 2
; CHECK-NEXT:   %"mat'ipl2" = load double*, double** %2, align 8
; CHECK-NEXT:   %mat = load double*, double** %m_data.i.i.i, align 8
; CHECK-NEXT:   %cols = call i64 @_ZNK5Eigen9EigenBaseINS_6MatrixIdLin1ELin1ELi0ELin1ELin1EEEE4colsEv(i64* %tmp7)
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %"call'ipg" = getelementptr inbounds double, double* %"mat'ipl", i64 %iv
; CHECK-NEXT:   %"call'ipg3" = getelementptr inbounds double, double* %"mat'ipl1", i64 %iv
; CHECK-NEXT:   %"call'ipg4" = getelementptr inbounds double, double* %"mat'ipl2", i64 %iv
; CHECK-NEXT:   %call = getelementptr inbounds double, double* %mat, i64 %iv
; CHECK-NEXT:   %ld = load double, double* %call, align 8
; CHECK-NEXT:   %3 = load double, double* %"call'ipg", align 8
; CHECK-NEXT:   %4 = load double, double* %"call'ipg3", align 8
; CHECK-NEXT:   %5 = load double, double* %"call'ipg4", align 8
; CHECK-NEXT:   %fmul = fmul double %ld, %ld
; CHECK-NEXT:   %6 = fmul fast double %3, %ld
; CHECK-NEXT:   %7 = fmul fast double %3, %ld
; CHECK-NEXT:   %8 = fadd fast double %6, %7
; CHECK-NEXT:   %9 = fmul fast double %4, %ld
; CHECK-NEXT:   %10 = fmul fast double %4, %ld
; CHECK-NEXT:   %11 = fadd fast double %9, %10
; CHECK-NEXT:   %12 = fmul fast double %5, %ld
; CHECK-NEXT:   %13 = fmul fast double %5, %ld
; CHECK-NEXT:   %14 = fadd fast double %12, %13
; CHECK-NEXT:   store double %fmul, double* %call, align 8
; CHECK-NEXT:   store double %8, double* %"call'ipg", align 8
; CHECK-NEXT:   store double %11, double* %"call'ipg3", align 8
; CHECK-NEXT:   store double %14, double* %"call'ipg4", align 8
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next, %cols
; CHECK-NEXT:   br i1 %exitcond, label %exit, label %for.body

; CHECK: exit:                                             ; preds = %for.body
; CHECK-NEXT:   ret void
; CHECK-NEXT: }