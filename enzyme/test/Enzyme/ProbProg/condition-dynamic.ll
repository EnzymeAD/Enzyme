; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s

@.str = private constant [11 x i8] c"predict, 0\00"
@.str.1 = private constant [2 x i8] c"m\00"
@.str.2 = private constant [2 x i8] c"b\00"

@enzyme_condition = global i32 0
@enzyme_interface = global i32 0

declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)

declare double @exp(double)
declare double @log(double)

declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare i8* @__enzyme_condition(double (double*, i32)*, double*, i32, i32, i8*, i32, i8**)


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

define i8* @condition(double* %data, i32 %n, i8* %trace, i8** %interface) {
entry:
  %0 = load i32, i32* @enzyme_condition
  %1 = load i32, i32* @enzyme_interface
  %call = tail call i8* @__enzyme_condition(double (double*, i32)* @loss, double* %data, i32 %n, i32 %0, i8* %trace, i32 %1, i8** %interface)
  ret i8* %call
}


; CHECK: define i8* @condition(double* %data, i32 %n, i8* %trace, i8** %interface)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i32, i32* @enzyme_condition
; CHECK-NEXT:   %1 = load i32, i32* @enzyme_interface
; CHECK-NEXT:   %2 = call { double, i8* } @condition_loss(double* %data, i32 %n, i8** %interface, i8* %trace)
; CHECK-NEXT:   %3 = extractvalue { double, i8* } %2, 1
; CHECK-NEXT:   ret i8* %3
; CHECK-NEXT: }


; CHECK: define internal { double, i8* } @condition_loss(double* %data, i32 %n, i8** %interface, i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = getelementptr inbounds i8*, i8** %interface, i32 4
; CHECK-NEXT:   %1 = load i8*, i8** %0
; CHECK-NEXT:   %new_trace = bitcast i8* %1 to i8* ()*
; CHECK-NEXT:   %trace1 = call i8* %new_trace()
; CHECK-NEXT:   %2 = getelementptr inbounds i8*, i8** %interface, i32 7
; CHECK-NEXT:   %3 = load i8*, i8** %2
; CHECK-NEXT:   %has_choice = bitcast i8* %3 to i1 (i8*, i8*)*
; CHECK-NEXT:   %4 = call i1 %has_choice(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   br i1 %4, label %condition.call.with.trace, label %condition.call.without.trace

; CHECK: condition.call.with.trace:                        ; preds = %entry
; CHECK-NEXT:   %5 = alloca double
; CHECK-NEXT:   %6 = bitcast double* %5 to i8*
; CHECK-NEXT:   %7 = getelementptr inbounds i8*, i8** %interface, i32 1
; CHECK-NEXT:   %8 = load i8*, i8** %7
; CHECK-NEXT:   %get_choice = bitcast i8* %8 to i64 (i8*, i8*, i8*, i64)*
; CHECK-NEXT:   %9 = call i64 %get_choice(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %6, i64 8)
; CHECK-NEXT:   %10 = load double, double* %5
; CHECK-NEXT:   br label %entry.cntd

; CHECK: condition.call.without.trace:                     ; preds = %entry
; CHECK-NEXT:   %sample.call = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %entry.cntd

; CHECK: entry.cntd:                                       ; preds = %condition.call.without.trace, %condition.call.with.trace
; CHECK-NEXT:   %11 = phi double [ %10, %condition.call.with.trace ], [ %sample.call, %condition.call.without.trace ]
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %11)
; CHECK-NEXT:   %12 = alloca double
; CHECK-NEXT:   store double %11, double* %12
; CHECK-NEXT:   %13 = bitcast double* %12 to i8**
; CHECK-NEXT:   %14 = load i8*, i8** %13
; CHECK-NEXT:   %15 = getelementptr inbounds i8*, i8** %interface, i32 3
; CHECK-NEXT:   %16 = load i8*, i8** %15
; CHECK-NEXT:   %insert_choice = bitcast i8* %16 to void (i8*, i8*, double, i8*, i64)*
; CHECK-NEXT:   call void %insert_choice(i8* %trace1, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.call, i8* %14, i64 8)
; CHECK-NEXT:   %17 = getelementptr inbounds i8*, i8** %interface, i32 7
; CHECK-NEXT:   %18 = load i8*, i8** %17
; CHECK-NEXT:   %has_choice3 = bitcast i8* %18 to i1 (i8*, i8*)*
; CHECK-NEXT:   %19 = call i1 %has_choice3(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0))
; CHECK-NEXT:   br i1 %19, label %condition.call1.with.trace, label %condition.call1.without.trace

; CHECK: condition.call1.with.trace:                       ; preds = %entry.cntd
; CHECK-NEXT:   %20 = alloca double
; CHECK-NEXT:   %21 = bitcast double* %20 to i8*
; CHECK-NEXT:   %22 = getelementptr inbounds i8*, i8** %interface, i32 1
; CHECK-NEXT:   %23 = load i8*, i8** %22
; CHECK-NEXT:   %get_choice4 = bitcast i8* %23 to i64 (i8*, i8*, i8*, i64)*
; CHECK-NEXT:   %24 = call i64 %get_choice4(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), i8* %21, i64 8)
; CHECK-NEXT:   %25 = load double, double* %20
; CHECK-NEXT:   br label %entry.cntd.cntd

; CHECK: condition.call1.without.trace:                    ; preds = %entry.cntd
; CHECK-NEXT:   %sample.call1 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %entry.cntd.cntd

; CHECK: entry.cntd.cntd:                                  ; preds = %condition.call1.without.trace, %condition.call1.with.trace
; CHECK-NEXT:   %26 = phi double [ %25, %condition.call1.with.trace ], [ %sample.call1, %condition.call1.without.trace ]
; CHECK-NEXT:   %likelihood.call1 = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %26)
; CHECK-NEXT:   %27 = alloca double
; CHECK-NEXT:   store double %26, double* %27
; CHECK-NEXT:   %28 = bitcast double* %27 to i8**
; CHECK-NEXT:   %29 = load i8*, i8** %28
; CHECK-NEXT:   %30 = getelementptr inbounds i8*, i8** %interface, i32 3
; CHECK-NEXT:   %31 = load i8*, i8** %30
; CHECK-NEXT:   %insert_choice5 = bitcast i8* %31 to void (i8*, i8*, double, i8*, i64)*
; CHECK-NEXT:   call void %insert_choice5(i8* %trace1, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0), double %likelihood.call1, i8* %29, i64 8)
; CHECK-NEXT:   %32 = getelementptr inbounds i8*, i8** %interface, i32 6
; CHECK-NEXT:   %33 = load i8*, i8** %32
; CHECK-NEXT:   %has_call = bitcast i8* %33 to i1 (i8*, i8*)*
; CHECK-NEXT:   %34 = call i1 %has_call(i8* %trace, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @0, i32 0, i32 0))
; CHECK-NEXT:   br i1 %34, label %condition.call2.with.trace, label %condition.call2.without.trace

; CHECK: condition.call2.with.trace:                       ; preds = %entry.cntd.cntd
; CHECK-NEXT:   %35 = getelementptr inbounds i8*, i8** %interface, i32 0
; CHECK-NEXT:   %36 = load i8*, i8** %35
; CHECK-NEXT:   %get_trace = bitcast i8* %36 to i8* (i8*, i8*)*
; CHECK-NEXT:   %37 = call i8* %get_trace(i8* %trace, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @0, i32 0, i32 0))
; CHECK-NEXT:   %call26 = call { double, i8* } @condition_calculate_loss(double %11, double %26, double* %data, i32 %n, i8** %interface, i8* %37)
; CHECK-NEXT:   br label %entry.cntd.cntd.cntd

; CHECK: condition.call2.without.trace:                    ; preds = %entry.cntd.cntd
; CHECK-NEXT:   %call27 = call { double, i8* } @condition_calculate_loss(double %11, double %26, double* %data, i32 %n, i8** %interface, i8* null)
; CHECK-NEXT:   br label %entry.cntd.cntd.cntd

; CHECK: entry.cntd.cntd.cntd:                             ; preds = %condition.call2.without.trace, %condition.call2.with.trace
; CHECK-NEXT:   %38 = phi { double, i8* } [ %call26, %condition.call2.with.trace ], [ %call27, %condition.call2.without.trace ]
; CHECK-NEXT:   %39 = extractvalue { double, i8* } %38, 0
; CHECK-NEXT:   %40 = extractvalue { double, i8* } %38, 1
; CHECK-NEXT:   %41 = getelementptr inbounds i8*, i8** %interface, i32 2
; CHECK-NEXT:   %42 = load i8*, i8** %41
; CHECK-NEXT:   %insert_call = bitcast i8* %42 to void (i8*, i8*, i8*)*
; CHECK-NEXT:   call void %insert_call(i8* %trace1, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @0, i32 0, i32 0), i8* %40)
; CHECK-NEXT:   %mrv = insertvalue { double, i8* } undef, double %39, 0
; CHECK-NEXT:   %mrv2 = insertvalue { double, i8* } %mrv, i8* %trace1, 1
; CHECK-NEXT:   ret { double, i8* } %mrv2
; CHECK-NEXT: }


; CHECK: define internal { double, i8* } @condition_calculate_loss(double %m, double %b, double* %data, i32 %n, i8** %interface, i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = getelementptr inbounds i8*, i8** %interface, i32 4
; CHECK-NEXT:   %1 = load i8*, i8** %0
; CHECK-NEXT:   %new_trace = bitcast i8* %1 to i8* ()*
; CHECK-NEXT:   %trace1 = call i8* %new_trace()
; CHECK-NEXT:   %cmp19 = icmp sgt i32 %n, 0
; CHECK-NEXT:   br i1 %cmp19, label %for.body.preheader, label %for.cond.cleanup

; CHECK: for.body.preheader:                               ; preds = %entry
; CHECK-NEXT:   %wide.trip.count = zext i32 %n to i64
; CHECK-NEXT:   br label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.body.cntd, %entry
; CHECK-NEXT:   %loss.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %20, %for.body.cntd ]
; CHECK-NEXT:   %mrv = insertvalue { double, i8* } undef, double %loss.0.lcssa, 0
; CHECK-NEXT:   %mrv2 = insertvalue { double, i8* } %mrv, i8* %trace1, 1
; CHECK-NEXT:   ret { double, i8* } %mrv2

; CHECK: for.body:                                         ; preds = %for.body.cntd, %for.body.preheader
; CHECK-NEXT:   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body.cntd ]
; CHECK-NEXT:   %loss.021 = phi double [ 0.000000e+00, %for.body.preheader ], [ %20, %for.body.cntd ]
; CHECK-NEXT:   %2 = trunc i64 %indvars.iv to i32
; CHECK-NEXT:   %conv2 = sitofp i32 %2 to double
; CHECK-NEXT:   %mul1 = fmul double %conv2, %m
; CHECK-NEXT:   %3 = fadd double %mul1, %b
; CHECK-NEXT:   %4 = getelementptr inbounds i8*, i8** %interface, i32 7
; CHECK-NEXT:   %5 = load i8*, i8** %4
; CHECK-NEXT:   %has_choice = bitcast i8* %5 to i1 (i8*, i8*)*
; CHECK-NEXT:   %6 = call i1 %has_choice(i8* %trace, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   br i1 %6, label %condition.call.with.trace, label %condition.call.without.trace

; CHECK: condition.call.with.trace:                        ; preds = %for.body
; CHECK-NEXT:   %7 = alloca double
; CHECK-NEXT:   %8 = bitcast double* %7 to i8*
; CHECK-NEXT:   %9 = getelementptr inbounds i8*, i8** %interface, i32 1
; CHECK-NEXT:   %10 = load i8*, i8** %9
; CHECK-NEXT:   %get_choice = bitcast i8* %10 to i64 (i8*, i8*, i8*, i64)*
; CHECK-NEXT:   %11 = call i64 %get_choice(i8* %trace, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i8* %8, i64 8)
; CHECK-NEXT:   %12 = load double, double* %7
; CHECK-NEXT:   br label %for.body.cntd

; CHECK: condition.call.without.trace:                     ; preds = %for.body
; CHECK-NEXT:   %sample.call = call double @normal(double %3, double 1.000000e+00)
; CHECK-NEXT:   br label %for.body.cntd

; CHECK: for.body.cntd:                                    ; preds = %condition.call.without.trace, %condition.call.with.trace
; CHECK-NEXT:   %13 = phi double [ %12, %condition.call.with.trace ], [ %sample.call, %condition.call.without.trace ]
; CHECK-NEXT:   %likelihood.call = call double @normal_logpdf(double %3, double 1.000000e+00, double %13)
; CHECK-NEXT:   %14 = alloca double
; CHECK-NEXT:   store double %13, double* %14
; CHECK-NEXT:   %15 = bitcast double* %14 to i8**
; CHECK-NEXT:   %16 = load i8*, i8** %15
; CHECK-NEXT:   %17 = getelementptr inbounds i8*, i8** %interface, i32 3
; CHECK-NEXT:   %18 = load i8*, i8** %17
; CHECK-NEXT:   %insert_choice = bitcast i8* %18 to void (i8*, i8*, double, i8*, i64)*
; CHECK-NEXT:   call void %insert_choice(i8* %trace1, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), double %likelihood.call, i8* %16, i64 8)
; CHECK-NEXT:   %arrayidx3 = getelementptr inbounds double, double* %data, i64 %indvars.iv
; CHECK-NEXT:   %19 = load double, double* %arrayidx3
; CHECK-NEXT:   %sub = fsub double %13, %19
; CHECK-NEXT:   %mul2 = fmul double %sub, %sub
; CHECK-NEXT:   %20 = fadd double %mul2, %loss.021
; CHECK-NEXT:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT:   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
; CHECK-NEXT:   br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
; CHECK-NEXT: }