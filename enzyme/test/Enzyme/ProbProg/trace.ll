; RUN: %opt < %s %loadEnzyme -enzyme -O1 -S | FileCheck %s

@.str = private unnamed_addr constant [11 x i8] c"predict, 0\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"m\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@enzyme_interface = dso_local local_unnamed_addr global i32 0, align 4

define dso_local double @normal(double %mean, double %var) #0 {
entry:
  ret double 4.000000e+00 ; chosen by fair dice roll.
                          ; guaranteed to be random.
}

define dso_local double @normal_pdf(double noundef %mean, double noundef %var, double noundef %x) local_unnamed_addr #1 {
entry:
  %div = fdiv double 1.000000e+00, %var
  %mul = fmul double %div, 0x40040D931FF62705
  %div1 = fdiv double %mean, %var
  %sub = fsub double %x, %div1
  %square = fmul double %sub, %sub
  %mul3 = fmul double %square, -5.000000e-01
  %call4 = tail call double @exp(double noundef %mul3) #6
  %mul5 = fmul double %mul, %call4
  ret double %mul5
}

declare double @exp(double noundef) local_unnamed_addr #2

define dso_local double @normal_logpdf(double noundef %mean, double noundef %var, double noundef %x) #1 {
entry:
  %div.i = fdiv double 1.000000e+00, %var
  %mul.i = fmul double %div.i, 0x40040D931FF62705
  %div1.i = fdiv double %mean, %var
  %sub.i = fsub double %x, %div1.i
  %square.i = fmul double %sub.i, %sub.i
  %mul3.i = fmul double %square.i, -5.000000e-01
  %call4.i = tail call double @exp(double noundef %mul3.i) #6
  %mul5.i = fmul double %mul.i, %call4.i
  %call1 = tail call double @log(double noundef %mul5.i) #6
  ret double %call1
}

declare double @log(double noundef) local_unnamed_addr #2

define dso_local double @calculate_loss(double noundef %m, double noundef %b, double* nocapture noundef readonly %data, i32 noundef %n) local_unnamed_addr #3 {
entry:
  %cmp19 = icmp sgt i32 %n, 0
  br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %loss.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %3, %for.body ]
  ret double %loss.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %loss.021 = phi double [ 0.000000e+00, %for.body.preheader ], [ %3, %for.body ]
  %0 = trunc i64 %indvars.iv to i32
  %conv2 = sitofp i32 %0 to double
  %1 = tail call double @llvm.fmuladd.f64(double %conv2, double %m, double %b)
  %call = tail call double @__enzyme_sample(double (double, double)* noundef nonnull @normal, double (double, double, double)* noundef nonnull @normal_logpdf, i8* noundef getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double noundef %1, double noundef 1.000000e+00) #6
  %arrayidx3 = getelementptr inbounds double, double* %data, i64 %indvars.iv
  %2 = load double, double* %arrayidx3, align 8
  %sub = fsub double %call, %2
  %3 = tail call double @llvm.fmuladd.f64(double %sub, double %sub, double %loss.021)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare double @__enzyme_sample(double (double, double)* noundef, double (double, double, double)* noundef, i8* noundef, double noundef, double noundef) local_unnamed_addr #4

declare double @llvm.fmuladd.f64(double, double, double) #5

define dso_local double @loss(double* nocapture noundef readonly %data, i32 noundef %n) #3 {
entry:
  %call = tail call double @__enzyme_sample(double (double, double)* noundef nonnull @normal, double (double, double, double)* noundef nonnull @normal_logpdf, i8* noundef getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double noundef 0.000000e+00, double noundef 1.000000e+00) #6
  %call1 = tail call double @__enzyme_sample(double (double, double)* noundef nonnull @normal, double (double, double, double)* noundef nonnull @normal_logpdf, i8* noundef getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double noundef 0.000000e+00, double noundef 1.000000e+00) #6
  %cmp19.i = icmp sgt i32 %n, 0
  br i1 %cmp19.i, label %for.body.preheader.i, label %calculate_loss.exit

for.body.preheader.i:                             ; preds = %entry
  %wide.trip.count.i = zext i32 %n to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i = phi i64 [ 0, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
  %loss.021.i = phi double [ 0.000000e+00, %for.body.preheader.i ], [ %3, %for.body.i ]
  %0 = trunc i64 %indvars.iv.i to i32
  %conv2.i = sitofp i32 %0 to double
  %1 = tail call double @llvm.fmuladd.f64(double %conv2.i, double %call, double %call1)
  %call.i = tail call double @__enzyme_sample(double (double, double)* noundef nonnull @normal, double (double, double, double)* noundef nonnull @normal_logpdf, i8* noundef getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double noundef %1, double noundef 1.000000e+00) #6
  %arrayidx3.i = getelementptr inbounds double, double* %data, i64 %indvars.iv.i
  %2 = load double, double* %arrayidx3.i, align 8
  %sub.i = fsub double %call.i, %2
  %3 = tail call double @llvm.fmuladd.f64(double %sub.i, double %sub.i, double %loss.021.i)
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %calculate_loss.exit, label %for.body.i

calculate_loss.exit:                              ; preds = %for.body.i, %entry
  %loss.0.lcssa.i = phi double [ 0.000000e+00, %entry ], [ %3, %for.body.i ]
  ret double %loss.0.lcssa.i
}

define dso_local i8* @generate(double* noundef %data, i32 noundef %n, i8** noundef %interface) local_unnamed_addr #3 {
entry:
  %0 = load i32, i32* @enzyme_interface, align 4
  %call = tail call i8* @__enzyme_trace(double (double*, i32)* noundef nonnull @loss, double* noundef %data, i32 noundef %n, i32 noundef %0, i8** noundef %interface) #6
  ret i8* %call
}

declare i8* @__enzyme_trace(double (double*, i32)* noundef, double* noundef, i32 noundef, i32 noundef, i8** noundef) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nofree nounwind willreturn writeonly uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nofree nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { noinline nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #6 = { nounwind }


; CHECK: define internal fastcc i8* @trace_loss(double* nocapture noundef readonly %data, i32 noundef %n, i8** nocapture readonly %0)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = getelementptr inbounds i8*, i8** %0, i64 4
; CHECK-NEXT:   %2 = bitcast i8** %1 to i8* ()**
; CHECK-NEXT:   %3 = load i8* ()*, i8* ()** %2, align 8
; CHECK-NEXT:   %trace = call i8* %3() #6
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double 4.000000e+00)
; CHECK-NEXT:   %4 = alloca double, align 8
; CHECK-NEXT:   store double 4.000000e+00, double* %4, align 8
; CHECK-NEXT:   %.0..0..sroa_cast3 = bitcast double* %4 to i8**
; CHECK-NEXT:   %.0..0.2 = load i8*, i8** %.0..0..sroa_cast3, align 8
; CHECK-NEXT:   %5 = getelementptr inbounds i8*, i8** %0, i64 3
; CHECK-NEXT:   %6 = bitcast i8** %5 to void (i8*, i8*, double, i8*, i64)**
; CHECK-NEXT:   %7 = load void (i8*, i8*, double, i8*, i64)*, void (i8*, i8*, double, i8*, i64)** %6, align 8
; CHECK-NEXT:   call void %7(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.call, i8* %.0..0.2, i64 8) #6
; CHECK-NEXT:   %likelihood.call1 = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double 4.000000e+00)
; CHECK-NEXT:   %8 = alloca double, align 8
; CHECK-NEXT:   store double 4.000000e+00, double* %8, align 8
; CHECK-NEXT:   %.0..0..sroa_cast = bitcast double* %8 to i8**
; CHECK-NEXT:   %.0..0. = load i8*, i8** %.0..0..sroa_cast, align 8
; CHECK-NEXT:   %9 = load void (i8*, i8*, double, i8*, i64)*, void (i8*, i8*, double, i8*, i64)** %6, align 8
; CHECK-NEXT:   call void %9(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double %likelihood.call1, i8* %.0..0., i64 8) #6
; CHECK-NEXT:   %cmp19.i = icmp sgt i32 %n, 0
; CHECK-NEXT:   br i1 %cmp19.i, label %for.body.preheader.i, label %calculate_loss.exit

; CHECK: for.body.preheader.i:                             ; preds = %entry
; CHECK-NEXT:   %wide.trip.count.i = zext i32 %n to i64
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
; CHECK-NEXT:   %indvars.iv.i = phi i64 [ 0, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
; CHECK-NEXT:   %10 = trunc i64 %indvars.iv.i to i32
; CHECK-NEXT:   %conv2.i = sitofp i32 %10 to double
; CHECK-NEXT:   %11 = tail call double @llvm.fmuladd.f64(double %conv2.i, double 4.000000e+00, double 4.000000e+00)
; CHECK-NEXT:   %likelihood.call.i = call double @normal_logpdf(double %11, double 1.000000e+00, double 4.000000e+00)
; CHECK-NEXT:   %12 = alloca double, align 8
; CHECK-NEXT:   store double 4.000000e+00, double* %12, align 8
; CHECK-NEXT:   %13 = bitcast double* %12 to i8**
; CHECK-NEXT:   %14 = load i8*, i8** %13, align 8
; CHECK-NEXT:   %15 = load void (i8*, i8*, double, i8*, i64)*, void (i8*, i8*, double, i8*, i64)** %6, align 8
; CHECK-NEXT:   call void %15(i8* %trace, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %likelihood.call.i, i8* %14, i64 8) #6
; CHECK-NEXT:   %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
; CHECK-NEXT:   %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
; CHECK-NEXT:   br i1 %exitcond.not.i, label %calculate_loss.exit, label %for.body.i

; CHECK: calculate_loss.exit:                              ; preds = %for.body.i, %entry
; CHECK-NEXT:   ret i8* %trace
; CHECK-NEXT: }