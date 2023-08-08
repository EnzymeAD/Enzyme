; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

@.str = private constant [11 x i8] c"predict, 0\00"
@.str.1 = private constant [2 x i8] c"m\00"
@.str.2 = private constant [2 x i8] c"b\00"

@enzyme_observations = global i32 0
@enzyme_const = global i32 0
@enzyme_trace = global i32 0

declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)

declare i8* @__enzyme_newtrace()
declare void @__enzyme_freetrace(i8*)
declare i8* @__enzyme_get_trace(i8*, i8*)
declare i64 @__enzyme_get_choice(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_call(i8*, i8*, i8*)
declare void @__enzyme_insert_choice(i8*, i8*, double, i8*, i64)
declare void @__enzyme_insert_argument(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_return(i8*, i8*, i64)
declare void @__enzyme_insert_function(i8*, i8*)
declare void @__enzyme_insert_gradient_choice(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_gradient_argument(i8*, i8*, i8*, i64)
declare i1 @__enzyme_has_call(i8*, i8*)
declare i1 @__enzyme_has_choice(i8*, i8*)

declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare double @__enzyme_condition(double (double*, i32)*, i32, double*, i32, i32, i8*, i32, i8*)


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
  %call2 = tail call double @calculate_loss(double %call, double %call1, double* %data, i32 %n)
  ret double %call2
}

define i8* @condition(double* %data, i32 %n, i8* %observations) {
entry:
  %0 = load i32, i32* @enzyme_observations
  %1 = load i32, i32* @enzyme_trace
  %2 = load i32, i32* @enzyme_const
  %trace = call i8* @__enzyme_newtrace()
  %call = tail call double @__enzyme_condition(double (double*, i32)* @loss, i32 %2, double* %data, i32 %n, i32 %0, i8* %observations, i32 %1, i8* %trace)
  ret i8* %trace
}


; CHECK: define internal double @condition_loss(double* %data, i32 %n, double* "enzyme_likelihood" %likelihood, i8* "enzyme_observations" %observations, i8* "enzyme_trace" %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %normal.ptr.i4 = alloca double
; CHECK-NEXT:   %normal.ptr.i = alloca double
; CHECK-NEXT:   call void @__enzyme_insert_function(i8* %trace, i8* bitcast (double (double*, i32, double*, i8*, i8*)* @condition_loss to i8*))
; CHECK-NEXT:   %0 = bitcast double* %data to i8*
; CHECK-NEXT:   call void @__enzyme_insert_argument(i8* %trace, i8* nocapture readonly getelementptr inbounds ([5 x i8], [5 x i8]* @0, i32 0, i32 0), i8* %0, i64 0)
; CHECK-NEXT:   %1 = zext i32 %n to i64
; CHECK-NEXT:   %2 = inttoptr i64 %1 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_argument(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @1, i32 0, i32 0), i8* %2, i64 4)
; CHECK-NEXT:   %3 = bitcast double* %normal.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %3)
; CHECK-NEXT:   %has.choice.normal.i = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.normal.i, label %condition.normal.with.trace.i, label %condition.normal.without.trace.i

; CHECK: condition.normal.with.trace.i:                    ; preds = %entry
; CHECK-NEXT:   %4 = bitcast double* %normal.ptr.i to i8*
; CHECK-NEXT:   %normal.size.i = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %4, i64 8)
; CHECK-NEXT:   %from.trace.normal.i = load double, double* %normal.ptr.i
; CHECK-NEXT:   br label %condition_normal.exit

; CHECK: condition.normal.without.trace.i:                 ; preds = %entry
; CHECK-NEXT:   %sample.normal.i = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %condition_normal.exit

; CHECK: condition_normal.exit:                            ; preds = %condition.normal.with.trace.i, %condition.normal.without.trace.i
; CHECK-NEXT:   %5 = phi double [ %from.trace.normal.i, %condition.normal.with.trace.i ], [ %sample.normal.i, %condition.normal.without.trace.i ]
; CHECK-NEXT:   %6 = bitcast double* %normal.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %6)
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %5)
; CHECK-NEXT:   %log_prob_sum = load double, double* %likelihood
; CHECK-NEXT:   %7 = fadd double %log_prob_sum, %likelihood.call
; CHECK-NEXT:   store double %7, double* %likelihood
; CHECK-NEXT:   %8 = bitcast double %5 to i64
; CHECK-NEXT:   %9 = inttoptr i64 %8 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.call, i8* %9, i64 8)
; CHECK-NEXT:   %10 = bitcast double* %normal.ptr.i4 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %10)
; CHECK-NEXT:   %has.choice.normal.i5 = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.normal.i5, label %condition.normal.with.trace.i8, label %condition.normal.without.trace.i10

; CHECK: condition.normal.with.trace.i8:                   ; preds = %condition_normal.exit
; CHECK-NEXT:   %11 = bitcast double* %normal.ptr.i4 to i8*
; CHECK-NEXT:   %normal.size.i6 = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), i8* %11, i64 8)
; CHECK-NEXT:   %from.trace.normal.i7 = load double, double* %normal.ptr.i4
; CHECK-NEXT:   br label %condition_normal.2.exit

; CHECK: condition.normal.without.trace.i10:               ; preds = %condition_normal.exit
; CHECK-NEXT:   %sample.normal.i9 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %condition_normal.2.exit

; CHECK: condition_normal.2.exit:                          ; preds = %condition.normal.with.trace.i8, %condition.normal.without.trace.i10
; CHECK-NEXT:   %12 = phi double [ %from.trace.normal.i7, %condition.normal.with.trace.i8 ], [ %sample.normal.i9, %condition.normal.without.trace.i10 ]
; CHECK-NEXT:   %13 = bitcast double* %normal.ptr.i4 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %13)
; CHECK-NEXT:   %likelihood.call1 = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %12)
; CHECK-NEXT:   %log_prob_sum1 = load double, double* %likelihood
; CHECK-NEXT:   %14 = fadd double %log_prob_sum1, %likelihood.call1
; CHECK-NEXT:   store double %14, double* %likelihood
; CHECK-NEXT:   %15 = bitcast double %12 to i64
; CHECK-NEXT:   %16 = inttoptr i64 %15 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double %likelihood.call1, i8* %16, i64 8)
; CHECK-NEXT:   %trace2 = call i8* @__enzyme_newtrace()
; CHECK-NEXT:   %has.call.call2 = call i1 @__enzyme_has_call(i8* %observations, i8* nocapture readonly getelementptr inbounds ([21 x i8], [21 x i8]* @6, i32 0, i32 0))
; CHECK-NEXT:   br i1 %has.call.call2, label %condition.call2.with.trace, label %condition.call2.without.trace

; CHECK: condition.call2.with.trace:                       ; preds = %condition_normal.2.exit
; CHECK-NEXT:   %calculate_loss.subtrace = call i8* @__enzyme_get_trace(i8* %observations, i8* nocapture readonly getelementptr inbounds ([21 x i8], [21 x i8]* @6, i32 0, i32 0))
; CHECK-NEXT:   %condition.calculate_loss = call double @condition_calculate_loss(double %5, double %12, double* %data, i32 %n, double* %likelihood, i8* %calculate_loss.subtrace, i8* %trace2)
; CHECK-NEXT:   br label %entry.cntd

; CHECK: condition.call2.without.trace:                    ; preds = %condition_normal.2.exit
; CHECK-NEXT:   %trace.calculate_loss = call double @condition_calculate_loss(double %5, double %12, double* %data, i32 %n, double* %likelihood, i8* null, i8* %trace2)
; CHECK-NEXT:   br label %entry.cntd

; CHECK: entry.cntd:                                       ; preds = %condition.call2.without.trace, %condition.call2.with.trace
; CHECK-NEXT:   %call2 = phi double [ %condition.calculate_loss, %condition.call2.with.trace ], [ %trace.calculate_loss, %condition.call2.without.trace ]
; CHECK-NEXT:   call void @__enzyme_insert_call(i8* %trace, i8* nocapture readonly getelementptr inbounds ([21 x i8], [21 x i8]* @6, i32 0, i32 0), i8* %trace2)
; CHECK-NEXT:   %17 = bitcast double %call2 to i64
; CHECK-NEXT:   %18 = inttoptr i64 %17 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_return(i8* %trace, i8* %18, i64 8)
; CHECK-NEXT:   ret double %call2
; CHECK-NEXT: }

; CHECK: define internal double @condition_calculate_loss(double %m, double %b, double* %data, i32 %n, double* "enzyme_likelihood" %likelihood, i8* "enzyme_observations" %observations, i8* "enzyme_trace" %trace) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %normal.ptr.i = alloca double
; CHECK-NEXT:   call void @__enzyme_insert_function(i8* %trace, i8* bitcast (double (double, double, double*, i32, double*, i8*, i8*)* @condition_calculate_loss to i8*))
; CHECK-NEXT:   %0 = bitcast double %m to i64
; CHECK-NEXT:   %1 = inttoptr i64 %0 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_argument(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @2, i32 0, i32 0), i8* %1, i64 8)
; CHECK-NEXT:   %2 = bitcast double %b to i64
; CHECK-NEXT:   %3 = inttoptr i64 %2 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_argument(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @3, i32 0, i32 0), i8* %3, i64 8)
; CHECK-NEXT:   %4 = bitcast double* %data to i8*
; CHECK-NEXT:   call void @__enzyme_insert_argument(i8* %trace, i8* nocapture readonly getelementptr inbounds ([5 x i8], [5 x i8]* @4, i32 0, i32 0), i8* %4, i64 0)
; CHECK-NEXT:   %5 = zext i32 %n to i64
; CHECK-NEXT:   %6 = inttoptr i64 %5 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_argument(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @5, i32 0, i32 0), i8* %6, i64 4)
; CHECK-NEXT:   %cmp19 = icmp sgt i32 %n, 0
; CHECK-NEXT:   br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

; CHECK: for.body.preheader:                               ; preds = %entry
; CHECK-NEXT:   %wide.trip.count = zext i32 %n to i64
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %condition_normal.8.exit, %entry
; CHECK-NEXT:   %loss.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %19, %condition_normal.8.exit ]
; CHECK-NEXT:   %7 = bitcast double %loss.0.lcssa to i64
; CHECK-NEXT:   %8 = inttoptr i64 %7 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_return(i8* %trace, i8* %8, i64 8)
; CHECK-NEXT:   ret double %loss.0.lcssa

; CHECK: for.body:                                         ; preds = %condition_normal.8.exit, %for.body.preheader
; CHECK-NEXT:   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %condition_normal.8.exit ]
; CHECK-NEXT:   %loss.021 = phi double [ 0.000000e+00, %for.body.preheader ], [ %19, %condition_normal.8.exit ]
; CHECK-NEXT:   %9 = trunc i64 %indvars.iv to i32
; CHECK-NEXT:   %conv2 = sitofp i32 %9 to double
; CHECK-NEXT:   %mul1 = fmul double %conv2, %m
; CHECK-NEXT:   %10 = fadd double %mul1, %b
; CHECK-NEXT:   %11 = bitcast double* %normal.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %11)
; CHECK-NEXT:   %has.choice.normal.i = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.normal.i, label %condition.normal.with.trace.i, label %condition.normal.without.trace.i

; CHECK: condition.normal.with.trace.i:                    ; preds = %for.body
; CHECK-NEXT:   %12 = bitcast double* %normal.ptr.i to i8*
; CHECK-NEXT:   %normal.size.i = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i8* %12, i64 8)
; CHECK-NEXT:   %from.trace.normal.i = load double, double* %normal.ptr.i
; CHECK-NEXT:   br label %condition_normal.8.exit

; CHECK: condition.normal.without.trace.i:                 ; preds = %for.body
; CHECK-NEXT:   %sample.normal.i = call double @normal(double %10, double 1.000000e+00)
; CHECK-NEXT:   br label %condition_normal.8.exit

; CHECK: condition_normal.8.exit:                          ; preds = %condition.normal.with.trace.i, %condition.normal.without.trace.i
; CHECK-NEXT:   %13 = phi double [ %from.trace.normal.i, %condition.normal.with.trace.i ], [ %sample.normal.i, %condition.normal.without.trace.i ]
; CHECK-NEXT:   %14 = bitcast double* %normal.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %14)
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double %10, double 1.000000e+00, double %13)
; CHECK-NEXT:   %log_prob_sum = load double, double* %likelihood
; CHECK-NEXT:   %15 = fadd double %log_prob_sum, %likelihood.call
; CHECK-NEXT:   store double %15, double* %likelihood
; CHECK-NEXT:   %16 = bitcast double %13 to i64
; CHECK-NEXT:   %17 = inttoptr i64 %16 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %likelihood.call, i8* %17, i64 8)
; CHECK-NEXT:   %arrayidx3 = getelementptr inbounds double, double* %data, i64 %indvars.iv
; CHECK-NEXT:   %18 = load double, double* %arrayidx3
; CHECK-NEXT:   %sub = fsub double %13, %18
; CHECK-NEXT:   %mul2 = fmul double %sub, %sub
; CHECK-NEXT:   %19 = fadd double %mul2, %loss.021
; CHECK-NEXT:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT:   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
; CHECK-NEXT:   br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
; CHECK-NEXT: }