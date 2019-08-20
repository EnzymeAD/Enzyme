; RUN: opt < %s -lower-autodiff -inline -mem2reg -adce -aggressive-instcombine -instsimplify -early-cse-memssa -simplifycfg -correlated-propagation -adce -S | FileCheck %s

%struct.n = type { double*, %struct.n* }

@.str = private unnamed_addr constant [6 x i8] c"x=%f\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"xp=%f\0A\00", align 1

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @sum_list(%struct.n* noalias readonly %node, i64 %times) local_unnamed_addr #0 {
entry:
  %cmp18 = icmp eq %struct.n* %node, null
  br i1 %cmp18, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup4, %entry
  %val.020 = phi %struct.n* [ %1, %for.cond.cleanup4 ], [ %node, %entry ]
  %sum.019 = phi double [ %add, %for.cond.cleanup4 ], [ 0.000000e+00, %entry ]
  %values = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 0
  %0 = load double*, double** %values, align 8, !tbaa !2
  br label %for.body5

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4, %entry
  %sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.cond.cleanup4 ]
  ret double %sum.0.lcssa

for.cond.cleanup4:                                ; preds = %for.body5
  %next = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 1
  %1 = load %struct.n*, %struct.n** %next, align 8, !tbaa !7
  %cmp = icmp eq %struct.n* %1, null
  br i1 %cmp, label %for.cond.cleanup, label %for.cond1.preheader

for.body5:                                        ; preds = %for.body5, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body5 ]
  %sum.116 = phi double [ %sum.019, %for.cond1.preheader ], [ %add, %for.body5 ]
  %arrayidx = getelementptr inbounds double, double* %0, i64 %indvars.iv
  %2 = load double, double* %arrayidx, align 8, !tbaa !8
  %add = fadd fast double %2, %sum.116
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %times
  br i1 %exitcond, label %for.cond.cleanup4, label %for.body5
}

; Function Attrs: nounwind uwtable
define dso_local double @list_creator(double %x, i64 %n, i64 %times) #1 {
entry:
  %add = shl i64 %times, 3
  %mul = add i64 %add, 8
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup7
  %call13 = tail call fast double @sum_list(%struct.n* %2, i64 %times)
  ret double %call13

for.body:                                         ; preds = %for.cond.cleanup7, %entry
  %indvars.iv30 = phi i64 [ 0, %entry ], [ %indvars.iv.next31, %for.cond.cleanup7 ]
  %list.029 = phi %struct.n* [ null, %entry ], [ %2, %for.cond.cleanup7 ]
  %call = tail call noalias i8* @malloc(i64 16) #4
  %next = getelementptr inbounds i8, i8* %call, i64 8
  %0 = bitcast i8* %next to %struct.n**
  store %struct.n* %list.029, %struct.n** %0, align 8, !tbaa !7
  %call2 = tail call noalias i8* @malloc(i64 %mul) #4
  %1 = bitcast i8* %call to i8**
  store i8* %call2, i8** %1, align 8, !tbaa !2
  %.cast = bitcast i8* %call2 to double*
  br label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8
  %2 = bitcast i8* %call to %struct.n*
  %indvars.iv.next31 = add nuw i64 %indvars.iv30, 1
  %exitcond32 = icmp eq i64 %indvars.iv30, %n
  br i1 %exitcond32, label %for.cond.cleanup, label %for.body

for.body8:                                        ; preds = %for.body8, %for.body
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx = getelementptr inbounds double, double* %.cast, i64 %indvars.iv
  store double %x, double* %arrayidx, align 8, !tbaa !8
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %times
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x, i64 %n, i64 %times) local_unnamed_addr #3 {
entry:
  %0 = tail call double (double (double, i64, i64)*, ...) @llvm.autodiff.p0f_f64f64i64i64f(double (double, i64, i64)* nonnull @list_creator, double %x, i64 %n, i64 %times)
  ret double %0
}

; Function Attrs: nounwind
declare double @llvm.autodiff.p0f_f64f64i64i64f(double (double, i64, i64)*, ...) #4

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #1 {
entry:
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !10
  %call.i = tail call fast double @strtod(i8* nocapture nonnull %0, i8** null) #4
  %arrayidx1 = getelementptr inbounds i8*, i8** %argv, i64 2
  %1 = load i8*, i8** %arrayidx1, align 8, !tbaa !10
  %call.i16 = tail call i64 @strtol(i8* nocapture nonnull %1, i8** null, i32 10) #4
  %sext = shl i64 %call.i16, 32
  %conv = ashr exact i64 %sext, 32
  %arrayidx3 = getelementptr inbounds i8*, i8** %argv, i64 3
  %2 = load i8*, i8** %arrayidx3, align 8, !tbaa !10
  %call.i17 = tail call i64 @strtol(i8* nocapture nonnull %2, i8** null, i32 10) #4
  %sext19 = shl i64 %call.i17, 32
  %conv5 = ashr exact i64 %sext19, 32
  %call6 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i64 0, i64 0), double %call.i)
  %call7 = tail call fast double @derivative(double %call.i, i64 %conv, i64 %conv5)
  %call8 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i64 0, i64 0), double %call7)
  ret i32 0
}

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local double @strtod(i8* readonly, i8** nocapture) local_unnamed_addr #2

; Function Attrs: nounwind
declare dso_local i64 @strtol(i8* readonly, i8** nocapture, i32) local_unnamed_addr #2

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"n", !4, i64 0, !4, i64 8}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !5, i64 0}
!10 = !{!4, !4, i64 0}


; CHECK: define dso_local double @derivative(double %x, i64 %n, i64 %times)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %add.i = shl i64 %times, 3
; CHECK-NEXT:   %mul.i = add i64 %add.i, 8
; CHECK-NEXT:   %0 = add nuw i64 %n, 1
; CHECK-NEXT:   %mallocsize.i = mul i64 %0, 8
; CHECK-NEXT:   %malloccall.i = call i8* @malloc(i64 %mallocsize.i) #4
; CHECK-NEXT:   %"call'mi_malloccache.i" = bitcast i8* %malloccall.i to i8**
; CHECK-NEXT:   %malloccall2.i = call i8* @malloc(i64 %mallocsize.i) #4
; CHECK-NEXT:   %"call2'mi_malloccache.i" = bitcast i8* %malloccall2.i to i8**
; CHECK-NEXT:   %malloccall5.i = call i8* @malloc(i64 %mallocsize.i) #4
; CHECK-NEXT:   %call2_malloccache.i = bitcast i8* %malloccall5.i to i8**
; CHECK-NEXT:   %[[mcall2:.+]] = call i8* @malloc(i64 %mallocsize.i) #4
; CHECK-NEXT:   %call_malloccache.i = bitcast i8* %[[mcall2]] to i8**
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.cond.cleanup7.i, %entry
; CHECK-NEXT:   %indvars.iv30.i = phi i64 [ 0, %entry ], [ %indvars.iv.next31.i, %for.cond.cleanup7.i ]
; CHECK-NEXT:   %1 = phi %struct.n* [ null, %entry ], [ %"'ipc.i", %for.cond.cleanup7.i ]
; CHECK-NEXT:   %list.029.i = phi %struct.n* [ null, %entry ], [ %9, %for.cond.cleanup7.i ]
; CHECK-NEXT:   %"call'mi.i" = call noalias i8* @malloc(i64 16) #4
; CHECK-NEXT:   %2 = getelementptr i8*, i8** %"call'mi_malloccache.i", i64 %indvars.iv30.i
; CHECK-NEXT:   store i8* %"call'mi.i", i8** %2
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"call'mi.i", i8 0, i64 16, i1 false) #4
; CHECK-NEXT:   %call.i = call noalias i8* @malloc(i64 16) #4
; CHECK-NEXT:   %3 = getelementptr i8*, i8** %call_malloccache.i, i64 %indvars.iv30.i
; CHECK-NEXT:   store i8* %call.i, i8** %3
; CHECK-NEXT:   %next.i = getelementptr inbounds i8, i8* %call.i, i64 8
; CHECK-NEXT:   %4 = bitcast i8* %next.i to %struct.n**
; CHECK-NEXT:   %"next'ipg.i" = getelementptr i8, i8* %"call'mi.i", i64 8
; CHECK-NEXT:   %"'ipc7.i" = bitcast i8* %"next'ipg.i" to %struct.n**
; CHECK-NEXT:   store %struct.n* %1, %struct.n** %"'ipc7.i"
; CHECK-NEXT:   store %struct.n* %list.029.i, %struct.n** %4, align 8, !tbaa !7
; CHECK-NEXT:   %"call2'mi.i" = call noalias i8* @malloc(i64 %mul.i) #4
; CHECK-NEXT:   %5 = getelementptr i8*, i8** %"call2'mi_malloccache.i", i64 %indvars.iv30.i
; CHECK-NEXT:   store i8* %"call2'mi.i", i8** %5
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"call2'mi.i", i8 0, i64 %mul.i, i1 false) #4
; CHECK-NEXT:   %call2.i = call noalias i8* @malloc(i64 %mul.i) #4
; CHECK-NEXT:   %6 = getelementptr i8*, i8** %call2_malloccache.i, i64 %indvars.iv30.i
; CHECK-NEXT:   store i8* %call2.i, i8** %6
; CHECK-NEXT:   %7 = bitcast i8* %call.i to i8**
; CHECK-NEXT:   %"'ipc3.i" = bitcast i8* %"call'mi.i" to i8**
; CHECK-NEXT:   store i8* %"call2'mi.i", i8** %"'ipc3.i"
; CHECK-NEXT:   store i8* %call2.i, i8** %7, align 8, !tbaa !2
; CHECK-NEXT:   %.cast.i = bitcast i8* %call2.i to double*
; CHECK-NEXT:   br label %for.body8.i

; CHECK: for.cond.cleanup7.i:                              ; preds = %for.body8.i
; CHECK-NEXT:   %8 = icmp ult i64 %indvars.iv30.i, %n
; CHECK-NEXT:   %9 = bitcast i8* %call.i to %struct.n*
; CHECK-NEXT:   %indvars.iv.next31.i = add nuw i64 %indvars.iv30.i, 1
; CHECK-NEXT:   %"'ipc.i" = bitcast i8* %"call'mi.i" to %struct.n*
; CHECK-NEXT:   br i1 %8, label %for.body.i, label %invertfor.cond.cleanup.i

; CHECK: for.body8.i:                                      ; preds = %for.body8.i, %for.body.i
; CHECK-NEXT:   %indvars.iv.i = phi i64 [ 0, %for.body.i ], [ %indvars.iv.next.i, %for.body8.i ]
; CHECK-NEXT:   %10 = icmp ult i64 %indvars.iv.i, %times
; CHECK-NEXT:   %arrayidx.i = getelementptr inbounds double, double* %.cast.i, i64 %indvars.iv.i
; CHECK-NEXT:   store double %x, double* %arrayidx.i, align 8, !tbaa !8
; CHECK-NEXT:   %indvars.iv.next.i = add nuw i64 %indvars.iv.i, 1
; CHECK-NEXT:   br i1 %10, label %for.body8.i, label %for.cond.cleanup7.i

; CHECK: invertfor.cond.cleanup.i:                         ; preds = %for.cond.cleanup7.i
; CHECK-NEXT:   %11 = call {} @diffesum_list(%struct.n* %9, %struct.n* %"'ipc.i", i64 %times, double 1.000000e+00) #4
; CHECK-NEXT:   br label %invertfor.cond.cleanup7.i

; CHECK: invertfor.body.i:                                 ; preds = %invertfor.body8.i
; CHECK-NEXT:   %12 = getelementptr i8*, i8** %call2_malloccache.i, i64 %"indvars.iv30'phi.i"
; CHECK-NEXT:   %13 = load i8*, i8** %12
; CHECK-NEXT:   call void @free(i8* %13) #4
; CHECK-NEXT:   %14 = getelementptr i8*, i8** %call_malloccache.i, i64 %"indvars.iv30'phi.i"
; CHECK-NEXT:   %15 = load i8*, i8** %14
; CHECK-NEXT:   call void @free(i8* %15) #4
; CHECK-NEXT:   %16 = getelementptr i8*, i8** %"call'mi_malloccache.i", i64 %"indvars.iv30'phi.i"
; CHECK-NEXT:   %17 = load i8*, i8** %16
; CHECK-NEXT:   %18 = load i8*, i8** %22
; CHECK-NEXT:   %19 = icmp ne i64 %"indvars.iv30'phi.i", 0
; CHECK-NEXT:   call void @free(i8* %17) #4
; CHECK-NEXT:   call void @free(i8* %18) #4
; CHECK-NEXT:   br i1 %19, label %invertfor.cond.cleanup7.i, label %diffelist_creator.exit

; CHECK: invertfor.cond.cleanup7.i:                        ; preds = %invertfor.body.i, %invertfor.cond.cleanup.i
; CHECK-NEXT:   %"x'de.0.i" = phi double [ 0.000000e+00, %invertfor.cond.cleanup.i ], [ %25, %invertfor.body.i ]
; CHECK-NEXT:   %"indvars.iv30'phi.i" = phi i64 [ %n, %invertfor.cond.cleanup.i ], [ %20, %invertfor.body.i ]
; CHECK-NEXT:   %20 = sub i64 %"indvars.iv30'phi.i", 1
; CHECK-NEXT:   br label %invertfor.body8.i

; CHECK: invertfor.body8.i:                                ; preds = %invertfor.body8.i, %invertfor.cond.cleanup7.i
; CHECK-NEXT:   %"x'de.1.i" = phi double [ %"x'de.0.i", %invertfor.cond.cleanup7.i ], [ %25, %invertfor.body8.i ]
; CHECK-NEXT:   %"indvars.iv'phi.i" = phi i64 [ %times, %invertfor.cond.cleanup7.i ], [ %21, %invertfor.body8.i ]
; CHECK-NEXT:   %21 = sub i64 %"indvars.iv'phi.i", 1
; CHECK-NEXT:   %22 = getelementptr i8*, i8** %"call2'mi_malloccache.i", i64 %"indvars.iv30'phi.i"
; CHECK-NEXT:   %23 = load i8*, i8** %22
; CHECK-NEXT:   %".cast'ipc.i" = bitcast i8* %23 to double*
; CHECK-NEXT:   %"arrayidx'ipg.i" = getelementptr double, double* %".cast'ipc.i", i64 %"indvars.iv'phi.i"
; CHECK-NEXT:   %24 = load double, double* %"arrayidx'ipg.i"
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx'ipg.i"
; CHECK-NEXT:   %25 = fadd fast double %"x'de.1.i", %24
; CHECK-NEXT:   %26 = icmp ne i64 %"indvars.iv'phi.i", 0
; CHECK-NEXT:   br i1 %26, label %invertfor.body8.i, label %invertfor.body.i

; CHECK: diffelist_creator.exit:                           ; preds = %invertfor.body.i
; CHECK-NEXT:   call void @free(i8* nonnull %[[mcall2]]) #4
; CHECK-NEXT:   call void @free(i8* nonnull %malloccall5.i) #4
; CHECK-NEXT:   call void @free(i8* nonnull %malloccall2.i) #4
; CHECK-NEXT:   call void @free(i8* nonnull %malloccall.i) #4
; CHECK-NEXT:   ret double %25
; CHECK-NEXT: }



; CHECK: define internal {} @diffesum_list(%struct.n* noalias readonly %node, %struct.n* %"node'", i64 %times, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp18 = icmp eq %struct.n* %node, null
; CHECK-NEXT:   br i1 %cmp18, label %invertfor.cond.cleanup, label %for.cond1.preheader.preheader

; CHECK: for.cond1.preheader.preheader:                    ; preds = %entry
; CHECK-NEXT:   %malloccall = tail call i8* @malloc(i64 8)
; CHECK-NEXT:   %[[castmalloc:.+]] = bitcast i8* %malloccall to double**
; CHECK-NEXT:   br label %for.cond1.preheader

; CHECK: for.cond1.preheader:
; CHECK-NEXT:   %[[phirealloc:.+]] = phi double** [ %[[castmalloc]], %for.cond1.preheader.preheader ], [ %5, %for.cond.cleanup4 ]
; CHECK-NEXT:   %0 = phi i64 [ %2, %for.cond.cleanup4 ], [ 0, %for.cond1.preheader.preheader ]
; CHECK-NEXT:   %1 = phi %struct.n* [ %"'ipl", %for.cond.cleanup4 ], [ %"node'", %for.cond1.preheader.preheader ]
; CHECK-NEXT:   %val.020 = phi %struct.n* [ %7, %for.cond.cleanup4 ], [ %node, %for.cond1.preheader.preheader ]
; CHECK-NEXT:   %"values'ipg" = getelementptr %struct.n, %struct.n* %1, i64 0, i32 0
; CHECK-NEXT:   %"'ipl1" = load double*, double** %"values'ipg", align 8
; CHECK-NEXT:   %2 = add nuw i64 %0, 1
; CHECK-NEXT:   %3 = bitcast double** %[[phirealloc]] to i8*
; CHECK-NEXT:   %4 = mul nuw i64 8, %2
; CHECK-NEXT:   %[[postrealloc:.+]] = call i8* @realloc(i8* %3, i64 %4)
; CHECK-NEXT:   %5 = bitcast i8* %[[postrealloc]] to double**
; CHECK-NEXT:   %6 = getelementptr double*, double** %5, i64 %0
; CHECK-NEXT:   store double* %"'ipl1", double** %6
; CHECK-NEXT:   br label %for.body5

; CHECK: for.cond.cleanup4:                                ; preds = %for.body5
; CHECK-NEXT:   %next = getelementptr inbounds %struct.n, %struct.n* %val.020, i64 0, i32 1
; CHECK-NEXT:   %"next'ipg" = getelementptr %struct.n, %struct.n* %1, i64 0, i32 1
; CHECK-NEXT:   %"'ipl" = load %struct.n*, %struct.n** %"next'ipg", align 8
; CHECK-NEXT:   %7 = load %struct.n*, %struct.n** %next, align 8, !tbaa !7
; CHECK-NEXT:   %cmp = icmp eq %struct.n* %7, null
; CHECK-NEXT:   br i1 %cmp, label %invertfor.cond.cleanup, label %for.cond1.preheader

; CHECK: for.body5:                                        ; preds = %for.body5, %for.cond1.preheader
; CHECK-NEXT:   %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body5 ]
; CHECK-NEXT:   %8 = icmp ult i64 %indvars.iv, %times
; CHECK-NEXT:   %indvars.iv.next = add nuw i64 %indvars.iv, 1
; CHECK-NEXT:   br i1 %8, label %for.body5, label %for.cond.cleanup4

; CHECK: invertentry:                                      ; preds = %invertfor.cond.cleanup, %invertfor.cond1.preheader.preheader
; CHECK-NEXT:   ret {} undef

; CHECK: invertfor.cond1.preheader.preheader:              ; preds = %invertfor.cond1.preheader
; CHECK-NEXT:   %9 = bitcast double** %[[invertcache:.+]] to i8*
; CHECK-NEXT:   tail call void @free(i8* %9)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertfor.cond1.preheader:                        ; preds = %invertfor.body5
; CHECK-NEXT:   %10 = icmp ne i64 %"'phi", 0
; CHECK-NEXT:   br i1 %10, label %invertfor.cond.cleanup4, label %invertfor.cond1.preheader.preheader
; CHECK: invertfor.cond.cleanup:                           ; preds = %entry, %for.cond.cleanup4
; CHECK-NEXT:   %_cache.0 = phi i64 [ undef, %entry ], [ %0, %for.cond.cleanup4 ]
; CHECK-NEXT:   %[[invertcache]] = phi double** [ undef, %entry ], [ %5, %for.cond.cleanup4 ]
; CHECK-NEXT:   br i1 %cmp18, label %invertentry, label %invertfor.cond.cleanup4

; CHECK: invertfor.cond.cleanup4:                          ; preds = %invertfor.cond.cleanup, %invertfor.cond1.preheader
; CHECK-NEXT:   %"add.lcssa'de.0" = phi double [ %20, %invertfor.cond1.preheader ], [ %differeturn, %invertfor.cond.cleanup ]
; CHECK-NEXT:   %"'phi" = phi i64 [ %11, %invertfor.cond1.preheader ], [ %_cache.0, %invertfor.cond.cleanup ]
; CHECK-NEXT:   %11 = sub i64 %"'phi", 1
; CHECK-NEXT:   %12 = fadd fast double 0.000000e+00, %"add.lcssa'de.0"
; CHECK-NEXT:   br label %invertfor.body5

; CHECK: invertfor.body5:                                  ; preds = %invertfor.body5, %invertfor.cond.cleanup4
; CHECK-NEXT:   %"sum.019'de.1" = phi double [ 0.000000e+00, %invertfor.cond.cleanup4 ], [ %[[seladd:.+]], %invertfor.body5 ]
; CHECK-NEXT:   %"indvars.iv'phi" = phi i64 [ %times, %invertfor.cond.cleanup4 ], [ %[[idxsub:.+]], %invertfor.body5 ]
; CHECK-NEXT:   %[[idxsub]] = sub i64 %"indvars.iv'phi", 1
; CHECK-NEXT:   %14 = getelementptr double*, double** %[[invertcache]], i64 %"'phi"
; CHECK-NEXT:   %15 = load double*, double** %14
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr double, double* %15, i64 %"indvars.iv'phi"
; CHECK-NEXT:   %[[arrayload:.+]] = load double, double* %"arrayidx'ipg"
; CHECK-NEXT:   %[[arraytostore:.+]] = fadd fast double %[[arrayload]], %12
; CHECK-NEXT:   store double %[[arraytostore]], double* %"arrayidx'ipg"
; CHECK-NEXT:   %[[endcond:.+]] = icmp ne i64 %"indvars.iv'phi", 0
; CHECK-NEXT:   %[[selected:.+]] = select i1 %[[endcond]], double 0.000000e+00, double %12
; CHECK-NEXT:   %[[seladd]] = fadd fast double %"sum.019'de.1", %[[selected]]
; CHECK-NEXT:   br i1 %[[endcond]], label %invertfor.body5, label %invertfor.cond1.preheader
; CHECK-NEXT: }
