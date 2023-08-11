; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

@.str = private constant [11 x i8] c"predict, 0\00"
@.str.1 = private constant [2 x i8] c"m\00"
@.str.2 = private constant [2 x i8] c"b\00"

@enzyme_interface = global i32 0
@enzyme_const = global i32 0
@enzyme_trace = global i32 0

declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)

declare i8* @__enzyme_newtrace()

declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare double @__enzyme_trace(double (double*, i32)*, i32, double*, i32, i32, i8*, i32, i8**)

define double @calculate_loss(double %m, double %b, double* %data, i32 %n) {
entry:
  %cmp19 = icmp sgt i32 %n, 0
  br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %loss.0.lcssa = phi double [ 0.0, %entry ], [ %3, %for.body ]
  ret double %loss.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %loss.021 = phi double [ 0.0, %for.body.preheader ], [ %3, %for.body ]
  %0 = trunc i64 %indvars.iv to i32
  %conv2 = sitofp i32 %0 to double
  %mul1 = fmul double %conv2, %m
  %1 = fadd double %mul1, %b 
  %call = tail call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %1, double 1.0)
  %arrayidx3 = getelementptr inbounds double, double* %data, i64 %indvars.iv
  %2 = load double, double* %arrayidx3
  %sub = fsub double %call, %2
  %mul2 = fmul double %sub, %sub
  %3 = fadd double %mul2, %loss.021
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define double @loss(double* %data, i32 %n) {
entry:
  %call = tail call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double 0.0, double 1.0)
  %call1 = tail call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double 0.0, double 1.0)
  %cmp19.i = icmp sgt i32 %n, 0
  br i1 %cmp19.i, label %for.body.preheader.i, label %calculate_loss.exit

for.body.preheader.i:                             ; preds = %entry
  %wide.trip.count.i = zext i32 %n to i64
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
  %indvars.iv.i = phi i64 [ 0, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
  %loss.021.i = phi double [ 0.0, %for.body.preheader.i ], [ %3, %for.body.i ]
  %0 = trunc i64 %indvars.iv.i to i32
  %conv2.i = sitofp i32 %0 to double
  %mul1 = fmul double %conv2.i, %call
  %1 = fadd double %mul1, %call1
  %call.i = tail call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %1, double 1.0)
  %arrayidx3.i = getelementptr inbounds double, double* %data, i64 %indvars.iv.i
  %2 = load double, double* %arrayidx3.i
  %sub.i = fsub double %call.i, %2
  %mul2 = fmul double %sub.i, %sub.i
  %3 = fadd double %mul2, %loss.021.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.not.i, label %calculate_loss.exit, label %for.body.i

calculate_loss.exit:                              ; preds = %for.body.i, %entry
  %loss.0.lcssa.i = phi double [ 0.0, %entry ], [ %3, %for.body.i ]
  ret double %loss.0.lcssa.i
}

define i8* @generate(double* %data, i32 %n, i8** %interface) {
entry:
  %0 = load i32, i32* @enzyme_interface
  %1 = load i32, i32* @enzyme_const
  %2 = load i32, i32* @enzyme_trace
  %trace = call i8* @__enzyme_newtrace()
  %call = tail call double @__enzyme_trace(double (double*, i32)* @loss, i32 %1, double* %data, i32 %n, i32 %2, i8* %trace, i32 %0, i8** %interface)
  ret i8* %trace
}


; CHECK: define internal double @trace_loss(double* %data, i32 %n, double* "enzyme_likelihood" %likelihood, i8* "enzyme_trace" %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %insert_function.i = load void (i8*, i8*)*, void (i8*, i8*)** @insert_function_ptr
; CHECK-NEXT:   call void %insert_function.i(i8* %trace, i8* bitcast (double (double*, i32, double*, i8*)* @trace_loss to i8*))
; CHECK-NEXT:   %0 = bitcast double* %data to i8*
; CHECK-NEXT:   call void @insert_argument(i8* %trace, i8* nocapture readonly getelementptr inbounds ([5 x i8], [5 x i8]* @0, i32 0, i32 0), i8* %0, i64 0)
; CHECK-NEXT:   %1 = zext i32 %n to i64
; CHECK-NEXT:   %2 = inttoptr i64 %1 to i8*
; CHECK-NEXT:   call void @insert_argument(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @1, i32 0, i32 0), i8* %2, i64 4)
; CHECK-NEXT:   %3 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %3)
; CHECK-NEXT:   %log_prob_sum = load double, double* %likelihood
; CHECK-NEXT:   %4 = fadd double %log_prob_sum, %likelihood.call
; CHECK-NEXT:   store double %4, double* %likelihood
; CHECK-NEXT:   %5 = bitcast double %3 to i64
; CHECK-NEXT:   %6 = inttoptr i64 %5 to i8*
; CHECK-NEXT:   call void @insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.call, i8* %6, i64 8)
; CHECK-NEXT:   %7 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.call1 = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %7)
; CHECK-NEXT:   %log_prob_sum1 = load double, double* %likelihood
; CHECK-NEXT:   %8 = fadd double %log_prob_sum1, %likelihood.call1
; CHECK-NEXT:   store double %8, double* %likelihood
; CHECK-NEXT:   %9 = bitcast double %7 to i64
; CHECK-NEXT:   %10 = inttoptr i64 %9 to i8*
; CHECK-NEXT:   call void @insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double %likelihood.call1, i8* %10, i64 8)
; CHECK-NEXT:   %cmp19.i = icmp sgt i32 %n, 0
; CHECK-NEXT:   br i1 %cmp19.i, label %for.body.preheader.i, label %calculate_loss.exit

; CHECK: for.body.preheader.i:                             ; preds = %entry
; CHECK-NEXT:   %wide.trip.count.i = zext i32 %n to i64
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %for.body.preheader.i
; CHECK-NEXT:   %indvars.iv.i = phi i64 [ 0, %for.body.preheader.i ], [ %indvars.iv.next.i, %for.body.i ]
; CHECK-NEXT:   %loss.021.i = phi double [ 0.000000e+00, %for.body.preheader.i ], [ %18, %for.body.i ]
; CHECK-NEXT:   %11 = trunc i64 %indvars.iv.i to i32
; CHECK-NEXT:   %conv2.i = sitofp i32 %11 to double
; CHECK-NEXT:   %mul1 = fmul double %conv2.i, %3
; CHECK-NEXT:   %12 = fadd double %mul1, %7
; CHECK-NEXT:   %13 = call double @normal(double %12, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.call.i = call double @normal_logpdf(double %12, double 1.000000e+00, double %13)
; CHECK-NEXT:   %log_prob_sum2 = load double, double* %likelihood
; CHECK-NEXT:   %14 = fadd double %log_prob_sum2, %likelihood.call.i
; CHECK-NEXT:   store double %14, double* %likelihood
; CHECK-NEXT:   %15 = bitcast double %13 to i64
; CHECK-NEXT:   %16 = inttoptr i64 %15 to i8*
; CHECK-NEXT:   call void @insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %likelihood.call.i, i8* %16, i64 8)
; CHECK-NEXT:   %arrayidx3.i = getelementptr inbounds double, double* %data, i64 %indvars.iv.i
; CHECK-NEXT:   %17 = load double, double* %arrayidx3.i
; CHECK-NEXT:   %sub.i = fsub double %13, %17
; CHECK-NEXT:   %mul2 = fmul double %sub.i, %sub.i
; CHECK-NEXT:   %18 = fadd double %mul2, %loss.021.i
; CHECK-NEXT:   %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
; CHECK-NEXT:   %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
; CHECK-NEXT:   br i1 %exitcond.not.i, label %calculate_loss.exit, label %for.body.i

; CHECK: calculate_loss.exit:                              ; preds = %for.body.i, %entry
; CHECK-NEXT:   %loss.0.lcssa.i = phi double [ 0.000000e+00, %entry ], [ %18, %for.body.i ]
; CHECK-NEXT:   %19 = bitcast double %loss.0.lcssa.i to i64
; CHECK-NEXT:   %20 = inttoptr i64 %19 to i8*
; CHECK-NEXT:   %insert_return.i = load void (i8*, i8*, i64)*, void (i8*, i8*, i64)** @insert_return_ptr
; CHECK-NEXT:   call void %insert_return.i(i8* %trace, i8* %20, i64 8)
; CHECK-NEXT:   ret double %loss.0.lcssa.i
; CHECK-NEXT: }