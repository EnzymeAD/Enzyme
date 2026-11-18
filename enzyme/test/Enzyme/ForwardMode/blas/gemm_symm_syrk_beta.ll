; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: PASS

@main.A = internal constant [6 x double] [double 1.200000e+00, double -5.000000e-01, double 9.900000e+01, double 8.000000e-01, double 2.100000e+00, double 9.900000e+01], align 16
@main.dA = internal constant [6 x double] [double 3.000000e-01, double 4.000000e-01, double 0.000000e+00, double -2.000000e-01, double 9.000000e-01, double 0.000000e+00], align 16
@main.B = internal constant [9 x double] [double -6.000000e-01, double 1.400000e+00, double 9.900000e+01, double 5.000000e-01, double -1.100000e+00, double 9.900000e+01, double 2.000000e+00, double 3.000000e-01, double 9.900000e+01], align 16
@main.dB = internal constant [9 x double] [double 2.000000e-01, double 0xBFE6666666666666, double 0.000000e+00, double 1.100000e+00, double 6.000000e-01, double 0.000000e+00, double -4.000000e-01, double 8.000000e-01, double 0.000000e+00], align 16
@main.C_init = internal constant [12 x double] [double 3.000000e+00, double -2.000000e+00, double 9.990000e+02, double 9.990000e+02, double 1.500000e+00, double 4.000000e+00, double 9.990000e+02, double 9.990000e+02, double -5.000000e-01, double 2.500000e+00, double 9.990000e+02, double 9.990000e+02], align 16
@main.dC_init = internal constant [12 x double] [double 6.000000e-01, double -1.200000e+00, double 7.770000e+02, double 7.770000e+02, double 1.300000e+00, double 4.000000e-01, double 7.770000e+02, double 7.770000e+02, double -9.000000e-01, double 1.700000e+00, double 7.770000e+02, double 7.770000e+02], align 16
@enzyme_dup = global i32 0, align 4
@.str = private unnamed_addr constant [6 x i8] c"dgemm\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"dgemm_pad\00", align 1
@main.A.2 = internal constant [4 x double] [double 1.000000e+00, double 2.000000e+00, double -1.000000e+00, double 5.000000e-01], align 16
@main.B.3 = internal constant [6 x double] [double 4.000000e-01, double -8.000000e-01, double 1.200000e+00, double 1.500000e+00, double 3.000000e-01, double 0xBFE6666666666666], align 16
@main.C_init.4 = internal constant [10 x double] [double 2.000000e+00, double -3.000000e+00, double 4.000000e+00, double 8.880000e+02, double 8.880000e+02, double -1.500000e+00, double 2.500000e+00, double 5.000000e-01, double 8.880000e+02, double 8.880000e+02], align 16
@main.dC_init.5 = internal constant [10 x double] [double 5.000000e-01, double 1.000000e+00, double -2.000000e+00, double 7.770000e+02, double 7.770000e+02, double -4.000000e-01, double 8.000000e-01, double 1.200000e+00, double 7.770000e+02, double 7.770000e+02], align 16
@enzyme_const = global i32 0, align 4
@.str.6 = private unnamed_addr constant [15 x i8] c"cblas_rowmajor\00", align 1
@.str.7 = private unnamed_addr constant [19 x i8] c"cblas_rowmajor_pad\00", align 1
@main.A.8 = internal constant [6 x double] [double 2.000000e+00, double -9.999000e+03, double 9.900000e+01, double 0xBFE6666666666666, double 1.500000e+00, double 9.900000e+01], align 16
@main.dA.9 = internal constant [6 x double] [double 5.000000e-01, double -8.888000e+03, double 0.000000e+00, double 3.000000e-01, double -6.000000e-01, double 0.000000e+00], align 16
@main.B.10 = internal constant [9 x double] [double 1.100000e+00, double -4.000000e-01, double 9.900000e+01, double -8.000000e-01, double 2.200000e+00, double 9.900000e+01, double 6.000000e-01, double 1.300000e+00, double 9.900000e+01], align 16
@main.dB.11 = internal constant [9 x double] [double -3.000000e-01, double 9.000000e-01, double 0.000000e+00, double 0x3FE6666666666666, double -5.000000e-01, double 0.000000e+00, double 1.200000e+00, double -2.000000e-01, double 0.000000e+00], align 16
@main.C_init.12 = internal constant [12 x double] [double 1.000000e+00, double 2.000000e+00, double 9.990000e+02, double 9.990000e+02, double -3.000000e+00, double 5.000000e-01, double 9.990000e+02, double 9.990000e+02, double 4.000000e+00, double -1.500000e+00, double 9.990000e+02, double 9.990000e+02], align 16
@main.dC_init.13 = internal constant [12 x double] [double -5.000000e-01, double 1.500000e+00, double 7.770000e+02, double 7.770000e+02, double 8.000000e-01, double -1.100000e+00, double 7.770000e+02, double 7.770000e+02, double 3.000000e-01, double 2.000000e+00, double 7.770000e+02, double 7.770000e+02], align 16
@.str.14 = private unnamed_addr constant [6 x i8] c"dsymm\00", align 1
@.str.15 = private unnamed_addr constant [10 x i8] c"dsymm_pad\00", align 1
@main.A.16 = internal constant [8 x double] [double 1.200000e+00, double 0xBFE6666666666666, double 9.000000e-01, double 9.900000e+01, double -4.000000e-01, double 1.800000e+00, double 5.000000e-01, double 9.900000e+01], align 16
@main.dA.17 = internal constant [8 x double] [double 3.000000e-01, double 5.000000e-01, double -2.000000e-01, double 0.000000e+00, double -8.000000e-01, double 4.000000e-01, double 0x3FE6666666666666, double 0.000000e+00], align 16
@.str.18 = private unnamed_addr constant [10 x i8] c"dsyrk_pad\00", align 1
@.str.19 = private unnamed_addr constant [10 x i8] c"dsyrk_opp\00", align 1
@.str.20 = private unnamed_addr constant [10 x i8] c"dsyrk_tri\00", align 1
@main.A.21 = internal constant [6 x double] [double 1.000000e+00, double -5.000000e-01, double 0x3FE6666666666666, double 1.200000e+00, double -3.000000e-01, double 9.000000e-01], align 16
@.str.22 = private unnamed_addr constant [16 x i8] c"cblas_dsyrk_pad\00", align 1
@.str.23 = private unnamed_addr constant [16 x i8] c"cblas_dsyrk_opp\00", align 1
@.str.24 = private unnamed_addr constant [16 x i8] c"cblas_dsyrk_tri\00", align 1
@.str.25 = private unnamed_addr constant [6 x i8] c"PASS\0A\00", align 1
@.str.26 = private unnamed_addr constant [43 x i8] c"MISMATCH %s[%d]: got=%.15f expected=%.15f\0A\00", align 1

; Function Attrs: noinline nounwind uwtable
define void @daxpy_(ptr noundef %np, ptr noundef %alphap, ptr noundef %x, ptr noundef %incxp, ptr noundef %y, ptr noundef %incyp) noinline {
entry:
  %0 = load i32, ptr %np, align 4
  %1 = load double, ptr %alphap, align 8
  %2 = load i32, ptr %incxp, align 4
  %3 = load i32, ptr %incyp, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %mul = mul nsw i32 %i.0, %2
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds double, ptr %x, i64 %idxprom
  %4 = load double, ptr %arrayidx, align 8
  %mul1 = fmul double %1, %4
  %mul2 = mul nsw i32 %i.0, %3
  %idxprom3 = sext i32 %mul2 to i64
  %arrayidx4 = getelementptr inbounds double, ptr %y, i64 %idxprom3
  %5 = load double, ptr %arrayidx4, align 8
  %add = fadd double %5, %mul1
  store double %add, ptr %arrayidx4, align 8
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @cblas_daxpy(i32 noundef %n, double noundef %alpha, ptr noundef %x, i32 noundef %incx, ptr noundef %y, i32 noundef %incy) noinline {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %mul = mul nsw i32 %i.0, %incx
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds double, ptr %x, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8
  %mul1 = fmul double %alpha, %0
  %mul2 = mul nsw i32 %i.0, %incy
  %idxprom3 = sext i32 %mul2 to i64
  %arrayidx4 = getelementptr inbounds double, ptr %y, i64 %idxprom3
  %1 = load double, ptr %arrayidx4, align 8
  %add = fadd double %1, %mul1
  store double %add, ptr %arrayidx4, align 8
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @dlascl_(ptr noundef %typep, ptr noundef %klp, ptr noundef %kup, ptr noundef %cfromp, ptr noundef %ctop, ptr noundef %mp, ptr noundef %np, ptr noundef %A, ptr noundef %ldap, ptr noundef %infop, i32 noundef %typelen) noinline {
entry:
  %0 = load i8, ptr %typep, align 1
  %1 = load double, ptr %ctop, align 8
  %2 = load double, ptr %cfromp, align 8
  %div = fdiv double %1, %2
  %3 = load i32, ptr %mp, align 4
  %4 = load i32, ptr %np, align 4
  %5 = load i32, ptr %ldap, align 4
  %tobool = icmp ne ptr %infop, null
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %infop, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  br label %for.cond

for.cond:                                         ; preds = %for.inc26, %if.end
  %c.0 = phi i32 [ 0, %if.end ], [ %inc27, %for.inc26 ]
  %cmp = icmp slt i32 %c.0, %4
  br i1 %cmp, label %for.cond1, label %for.end28

for.cond1:                                        ; preds = %for.cond, %for.inc
  %r.0 = phi i32 [ %inc, %for.inc ], [ 0, %for.cond ]
  %cmp2 = icmp slt i32 %r.0, %3
  br i1 %cmp2, label %for.body3, label %for.inc26

for.body3:                                        ; preds = %for.cond1
  %conv = sext i8 %0 to i32
  %cmp4 = icmp eq i32 %conv, 85
  %conv6 = sext i8 %0 to i32
  %cmp7 = icmp eq i32 %conv6, 117
  %or.cond = or i1 %cmp4, %cmp7
  %cmp9 = icmp sgt i32 %r.0, %c.0
  %or.cond1 = and i1 %or.cond, %cmp9
  br i1 %or.cond1, label %for.inc, label %if.end12

if.end12:                                         ; preds = %for.body3
  %conv13 = sext i8 %0 to i32
  %cmp14 = icmp eq i32 %conv13, 76
  %conv17 = sext i8 %0 to i32
  %cmp18 = icmp eq i32 %conv17, 108
  %or.cond2 = or i1 %cmp14, %cmp18
  %cmp21 = icmp slt i32 %r.0, %c.0
  %or.cond3 = and i1 %or.cond2, %cmp21
  br i1 %or.cond3, label %for.inc, label %if.end24

if.end24:                                         ; preds = %if.end12
  %mul = mul nsw i32 %c.0, %5
  %add = add nsw i32 %r.0, %mul
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %6 = load double, ptr %arrayidx, align 8
  %mul25 = fmul double %6, %div
  store double %mul25, ptr %arrayidx, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.end12, %for.body3, %if.end24
  %inc = add nsw i32 %r.0, 1
  br label %for.cond1

for.inc26:                                        ; preds = %for.cond1
  %inc27 = add nsw i32 %c.0, 1
  br label %for.cond

for.end28:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @cblas_dlascl(i32 noundef %layout, i8 noundef signext %type, i32 noundef %kl, i32 noundef %ku, double noundef %cfrom, double noundef %cto, i32 noundef %m, i32 noundef %n, ptr noundef %A, i32 noundef %lda, ptr noundef %info) noinline {
entry:
  %div = fdiv double %cto, %cfrom
  %tobool = icmp ne ptr %info, null
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %info, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  br label %for.cond

for.cond:                                         ; preds = %for.inc38, %if.end
  %r.0 = phi i32 [ 0, %if.end ], [ %inc39, %for.inc38 ]
  %cmp = icmp slt i32 %r.0, %m
  br i1 %cmp, label %for.cond1, label %for.end40

for.cond1:                                        ; preds = %for.cond, %for.inc
  %c.0 = phi i32 [ %inc, %for.inc ], [ 0, %for.cond ]
  %cmp2 = icmp slt i32 %c.0, %n
  br i1 %cmp2, label %for.body3, label %for.inc38

for.body3:                                        ; preds = %for.cond1
  %conv = sext i8 %type to i32
  %cmp4 = icmp eq i32 %conv, 85
  %conv6 = sext i8 %type to i32
  %cmp7 = icmp eq i32 %conv6, 117
  %or.cond = or i1 %cmp4, %cmp7
  %conv10 = sext i8 %type to i32
  %cmp11 = icmp eq i32 %conv10, 121
  %or.cond1 = or i1 %or.cond, %cmp11
  %cmp13 = icmp sgt i32 %r.0, %c.0
  %or.cond2 = and i1 %or.cond1, %cmp13
  br i1 %or.cond2, label %for.inc, label %if.end16

if.end16:                                         ; preds = %for.body3
  %conv17 = sext i8 %type to i32
  %cmp18 = icmp eq i32 %conv17, 76
  %conv21 = sext i8 %type to i32
  %cmp22 = icmp eq i32 %conv21, 108
  %or.cond3 = or i1 %cmp18, %cmp22
  %conv25 = sext i8 %type to i32
  %cmp26 = icmp eq i32 %conv25, 122
  %or.cond4 = or i1 %or.cond3, %cmp26
  %cmp29 = icmp slt i32 %r.0, %c.0
  %or.cond5 = and i1 %or.cond4, %cmp29
  br i1 %or.cond5, label %for.inc, label %if.end32

if.end32:                                         ; preds = %if.end16
  %cmp33 = icmp eq i32 %layout, 101
  %mul = mul nsw i32 %r.0, %lda
  %add = add nsw i32 %mul, %c.0
  %mul35 = mul nsw i32 %c.0, %lda
  %add36 = add nsw i32 %r.0, %mul35
  %cond = select i1 %cmp33, i32 %add, i32 %add36
  %idxprom = sext i32 %cond to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8
  %mul37 = fmul double %0, %div
  store double %mul37, ptr %arrayidx, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.end16, %for.body3, %if.end32
  %inc = add nsw i32 %c.0, 1
  br label %for.cond1

for.inc38:                                        ; preds = %for.cond1
  %inc39 = add nsw i32 %r.0, 1
  br label %for.cond

for.end40:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @dgemm_(ptr noundef %transap, ptr noundef %transbp, ptr noundef %mp, ptr noundef %np, ptr noundef %kp, ptr noundef %alphap, ptr noundef %A, ptr noundef %ldap, ptr noundef %B, ptr noundef %ldbp, ptr noundef %betap, ptr noundef %C, ptr noundef %ldcp, i32 noundef %len1, i32 noundef %len2) noinline {
entry:
  %0 = load i8, ptr %transap, align 1
  %conv = sext i8 %0 to i32
  %cmp = icmp eq i32 %conv, 84
  br i1 %cmp, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  %1 = load i8, ptr %transap, align 1
  %conv2 = sext i8 %1 to i32
  %cmp3 = icmp eq i32 %conv2, 116
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %2 = phi i1 [ true, %entry ], [ %cmp3, %lor.rhs ]
  %lor.ext = zext i1 %2 to i32
  %3 = load i8, ptr %transbp, align 1
  %conv5 = sext i8 %3 to i32
  %cmp6 = icmp eq i32 %conv5, 84
  br i1 %cmp6, label %lor.end12, label %lor.rhs8

lor.rhs8:                                         ; preds = %lor.end
  %4 = load i8, ptr %transbp, align 1
  %conv9 = sext i8 %4 to i32
  %cmp10 = icmp eq i32 %conv9, 116
  br label %lor.end12

lor.end12:                                        ; preds = %lor.rhs8, %lor.end
  %5 = phi i1 [ true, %lor.end ], [ %cmp10, %lor.rhs8 ]
  %lor.ext13 = zext i1 %5 to i32
  %6 = load i32, ptr %mp, align 4
  %7 = load i32, ptr %np, align 4
  %8 = load i32, ptr %kp, align 4
  %9 = load i32, ptr %ldap, align 4
  %10 = load i32, ptr %ldbp, align 4
  %11 = load i32, ptr %ldcp, align 4
  %12 = load double, ptr %alphap, align 8
  %13 = load double, ptr %betap, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc57, %lor.end12
  %c.0 = phi i32 [ 0, %lor.end12 ], [ %inc58, %for.inc57 ]
  %cmp14 = icmp slt i32 %c.0, %7
  br i1 %cmp14, label %for.cond16, label %for.end59

for.cond16:                                       ; preds = %for.cond, %for.end
  %r.0 = phi i32 [ %inc55, %for.end ], [ 0, %for.cond ]
  %cmp17 = icmp slt i32 %r.0, %6
  br i1 %cmp17, label %for.cond20, label %for.inc57

for.cond20:                                       ; preds = %for.cond16, %cond.end39
  %sum.0 = phi double [ %add42, %cond.end39 ], [ 0.000000e+00, %for.cond16 ]
  %p.0 = phi i32 [ %inc, %cond.end39 ], [ 0, %for.cond16 ]
  %cmp21 = icmp slt i32 %p.0, %8
  br i1 %cmp21, label %for.body23, label %for.end

for.body23:                                       ; preds = %for.cond20
  %tobool = icmp ne i32 %lor.ext, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body23
  %mul = mul nsw i32 %r.0, %9
  %add = add nsw i32 %p.0, %mul
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %14 = load double, ptr %arrayidx, align 8
  br label %cond.end

cond.false:                                       ; preds = %for.body23
  %mul24 = mul nsw i32 %p.0, %9
  %add25 = add nsw i32 %r.0, %mul24
  %idxprom26 = sext i32 %add25 to i64
  %arrayidx27 = getelementptr inbounds double, ptr %A, i64 %idxprom26
  %15 = load double, ptr %arrayidx27, align 8
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ %14, %cond.true ], [ %15, %cond.false ]
  %tobool28 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool28, label %cond.true29, label %cond.false34

cond.true29:                                      ; preds = %cond.end
  %mul30 = mul nsw i32 %p.0, %10
  %add31 = add nsw i32 %c.0, %mul30
  %idxprom32 = sext i32 %add31 to i64
  %arrayidx33 = getelementptr inbounds double, ptr %B, i64 %idxprom32
  %16 = load double, ptr %arrayidx33, align 8
  br label %cond.end39

cond.false34:                                     ; preds = %cond.end
  %mul35 = mul nsw i32 %c.0, %10
  %add36 = add nsw i32 %p.0, %mul35
  %idxprom37 = sext i32 %add36 to i64
  %arrayidx38 = getelementptr inbounds double, ptr %B, i64 %idxprom37
  %17 = load double, ptr %arrayidx38, align 8
  br label %cond.end39

cond.end39:                                       ; preds = %cond.false34, %cond.true29
  %cond40 = phi double [ %16, %cond.true29 ], [ %17, %cond.false34 ]
  %mul41 = fmul double %cond, %cond40
  %add42 = fadd double %sum.0, %mul41
  %inc = add nsw i32 %p.0, 1
  br label %for.cond20

for.end:                                          ; preds = %for.cond20
  %mul43 = fmul double %12, %sum.0
  %mul44 = mul nsw i32 %c.0, %11
  %add45 = add nsw i32 %r.0, %mul44
  %idxprom46 = sext i32 %add45 to i64
  %arrayidx47 = getelementptr inbounds double, ptr %C, i64 %idxprom46
  %18 = load double, ptr %arrayidx47, align 8
  %mul48 = fmul double %13, %18
  %add49 = fadd double %mul43, %mul48
  %mul50 = mul nsw i32 %c.0, %11
  %add51 = add nsw i32 %r.0, %mul50
  %idxprom52 = sext i32 %add51 to i64
  %arrayidx53 = getelementptr inbounds double, ptr %C, i64 %idxprom52
  store double %add49, ptr %arrayidx53, align 8
  %inc55 = add nsw i32 %r.0, 1
  br label %for.cond16

for.inc57:                                        ; preds = %for.cond16
  %inc58 = add nsw i32 %c.0, 1
  br label %for.cond

for.end59:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @cblas_dgemm(i32 noundef %layout, i32 noundef %transa, i32 noundef %transb, i32 noundef %m, i32 noundef %n, i32 noundef %k, double noundef %alpha, ptr noundef %A, i32 noundef %lda, ptr noundef %B, i32 noundef %ldb, double noundef %beta, ptr noundef %C, i32 noundef %ldc) noinline {
entry:
  %cmp = icmp eq i32 %layout, 101
  %conv = zext i1 %cmp to i32
  %cmp1 = icmp eq i32 %transa, 112
  %conv2 = zext i1 %cmp1 to i32
  %cmp3 = icmp eq i32 %transb, 112
  %conv4 = zext i1 %cmp3 to i32
  br label %for.cond

for.cond:                                         ; preds = %for.inc69, %entry
  %r.0 = phi i32 [ 0, %entry ], [ %inc70, %for.inc69 ]
  %cmp5 = icmp slt i32 %r.0, %m
  br i1 %cmp5, label %for.cond7, label %for.end71

for.cond7:                                        ; preds = %for.cond, %for.end
  %c.0 = phi i32 [ %inc67, %for.end ], [ 0, %for.cond ]
  %cmp8 = icmp slt i32 %c.0, %n
  br i1 %cmp8, label %for.cond11, label %for.inc69

for.cond11:                                       ; preds = %for.cond7, %for.body14
  %sum.0 = phi double [ %add49, %for.body14 ], [ 0.000000e+00, %for.cond7 ]
  %p.0 = phi i32 [ %inc, %for.body14 ], [ 0, %for.cond7 ]
  %cmp12 = icmp slt i32 %p.0, %k
  br i1 %cmp12, label %for.body14, label %for.end

for.body14:                                       ; preds = %for.cond11
  %tobool = icmp ne i32 %conv2, 0
  %p.0.r.0 = select i1 %tobool, i32 %p.0, i32 %r.0
  %tobool15 = icmp ne i32 %conv2, 0
  %cond19 = select i1 %tobool15, i32 %r.0, i32 %p.0
  %tobool20 = icmp ne i32 %conv4, 0
  %cond24 = select i1 %tobool20, i32 %c.0, i32 %p.0
  %tobool25 = icmp ne i32 %conv4, 0
  %cond29 = select i1 %tobool25, i32 %p.0, i32 %c.0
  %tobool30 = icmp ne i32 %conv, 0
  %mul = mul nsw i32 %p.0.r.0, %lda
  %add = add nsw i32 %mul, %cond19
  %mul33 = mul nsw i32 %cond19, %lda
  %add34 = add nsw i32 %p.0.r.0, %mul33
  %cond36 = select i1 %tobool30, i32 %add, i32 %add34
  %idxprom = sext i32 %cond36 to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8
  %tobool37 = icmp ne i32 %conv, 0
  %mul39 = mul nsw i32 %cond24, %ldb
  %add40 = add nsw i32 %mul39, %cond29
  %mul42 = mul nsw i32 %cond29, %ldb
  %add43 = add nsw i32 %cond24, %mul42
  %cond45 = select i1 %tobool37, i32 %add40, i32 %add43
  %idxprom46 = sext i32 %cond45 to i64
  %arrayidx47 = getelementptr inbounds double, ptr %B, i64 %idxprom46
  %1 = load double, ptr %arrayidx47, align 8
  %mul48 = fmul double %0, %1
  %add49 = fadd double %sum.0, %mul48
  %inc = add nsw i32 %p.0, 1
  br label %for.cond11

for.end:                                          ; preds = %for.cond11
  %tobool50 = icmp ne i32 %conv, 0
  %mul52 = mul nsw i32 %r.0, %ldc
  %add53 = add nsw i32 %mul52, %c.0
  %mul55 = mul nsw i32 %c.0, %ldc
  %add56 = add nsw i32 %r.0, %mul55
  %cond58 = select i1 %tobool50, i32 %add53, i32 %add56
  %mul59 = fmul double %alpha, %sum.0
  %idxprom60 = sext i32 %cond58 to i64
  %arrayidx61 = getelementptr inbounds double, ptr %C, i64 %idxprom60
  %2 = load double, ptr %arrayidx61, align 8
  %mul62 = fmul double %beta, %2
  %add63 = fadd double %mul59, %mul62
  %idxprom64 = sext i32 %cond58 to i64
  %arrayidx65 = getelementptr inbounds double, ptr %C, i64 %idxprom64
  store double %add63, ptr %arrayidx65, align 8
  %inc67 = add nsw i32 %c.0, 1
  br label %for.cond7

for.inc69:                                        ; preds = %for.cond7
  %inc70 = add nsw i32 %r.0, 1
  br label %for.cond

for.end71:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @dsymm_(ptr noundef %sidep, ptr noundef %uplop, ptr noundef %mp, ptr noundef %np, ptr noundef %alphap, ptr noundef %A, ptr noundef %ldap, ptr noundef %B, ptr noundef %ldbp, ptr noundef %betap, ptr noundef %C, ptr noundef %ldcp, i32 noundef %len1, i32 noundef %len2) noinline {
entry:
  %0 = load i8, ptr %sidep, align 1
  %conv = sext i8 %0 to i32
  %cmp = icmp eq i32 %conv, 76
  br i1 %cmp, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  %1 = load i8, ptr %sidep, align 1
  %conv2 = sext i8 %1 to i32
  %cmp3 = icmp eq i32 %conv2, 108
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %2 = phi i1 [ true, %entry ], [ %cmp3, %lor.rhs ]
  %lor.ext = zext i1 %2 to i32
  %3 = load i8, ptr %uplop, align 1
  %conv5 = sext i8 %3 to i32
  %cmp6 = icmp eq i32 %conv5, 85
  br i1 %cmp6, label %lor.end12, label %lor.rhs8

lor.rhs8:                                         ; preds = %lor.end
  %4 = load i8, ptr %uplop, align 1
  %conv9 = sext i8 %4 to i32
  %cmp10 = icmp eq i32 %conv9, 117
  br label %lor.end12

lor.end12:                                        ; preds = %lor.rhs8, %lor.end
  %5 = phi i1 [ true, %lor.end ], [ %cmp10, %lor.rhs8 ]
  %lor.ext13 = zext i1 %5 to i32
  %6 = load i32, ptr %mp, align 4
  %7 = load i32, ptr %np, align 4
  %8 = load i32, ptr %ldap, align 4
  %9 = load i32, ptr %ldbp, align 4
  %10 = load i32, ptr %ldcp, align 4
  %11 = load double, ptr %alphap, align 8
  %12 = load double, ptr %betap, align 8
  %tobool = icmp ne i32 %lor.ext, 0
  %. = select i1 %tobool, i32 %6, i32 %7
  br label %for.cond

for.cond:                                         ; preds = %for.inc116, %lor.end12
  %c.0 = phi i32 [ 0, %lor.end12 ], [ %inc117, %for.inc116 ]
  %cmp14 = icmp slt i32 %c.0, %7
  br i1 %cmp14, label %for.cond16, label %for.end118

for.cond16:                                       ; preds = %for.cond, %for.end
  %r.0 = phi i32 [ %inc114, %for.end ], [ 0, %for.cond ]
  %cmp17 = icmp slt i32 %r.0, %6
  br i1 %cmp17, label %for.cond20, label %for.inc116

for.cond20:                                       ; preds = %for.cond16, %for.inc
  %sum.0 = phi double [ %sum.1, %for.inc ], [ 0.000000e+00, %for.cond16 ]
  %p.0 = phi i32 [ %inc, %for.inc ], [ 0, %for.cond16 ]
  %cmp21 = icmp slt i32 %p.0, %.
  br i1 %cmp21, label %for.body23, label %for.end

for.body23:                                       ; preds = %for.cond20
  %tobool24 = icmp ne i32 %lor.ext, 0
  br i1 %tobool24, label %if.then, label %if.else

if.then:                                          ; preds = %for.body23
  %tobool25 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool25, label %cond.true26, label %cond.false37

cond.true26:                                      ; preds = %if.then
  %cmp27 = icmp sle i32 %r.0, %p.0
  br i1 %cmp27, label %cond.true29, label %cond.false30

cond.true29:                                      ; preds = %cond.true26
  %mul = mul nsw i32 %p.0, %8
  %add = add nsw i32 %r.0, %mul
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %13 = load double, ptr %arrayidx, align 8
  br label %cond.end52

cond.false30:                                     ; preds = %cond.true26
  %mul31 = mul nsw i32 %r.0, %8
  %add32 = add nsw i32 %p.0, %mul31
  %idxprom33 = sext i32 %add32 to i64
  %arrayidx34 = getelementptr inbounds double, ptr %A, i64 %idxprom33
  %14 = load double, ptr %arrayidx34, align 8
  br label %cond.end52

cond.false37:                                     ; preds = %if.then
  %cmp38 = icmp sge i32 %r.0, %p.0
  br i1 %cmp38, label %cond.true40, label %cond.false45

cond.true40:                                      ; preds = %cond.false37
  %mul41 = mul nsw i32 %p.0, %8
  %add42 = add nsw i32 %r.0, %mul41
  %idxprom43 = sext i32 %add42 to i64
  %arrayidx44 = getelementptr inbounds double, ptr %A, i64 %idxprom43
  %15 = load double, ptr %arrayidx44, align 8
  br label %cond.end52

cond.false45:                                     ; preds = %cond.false37
  %mul46 = mul nsw i32 %r.0, %8
  %add47 = add nsw i32 %p.0, %mul46
  %idxprom48 = sext i32 %add47 to i64
  %arrayidx49 = getelementptr inbounds double, ptr %A, i64 %idxprom48
  %16 = load double, ptr %arrayidx49, align 8
  br label %cond.end52

cond.end52:                                       ; preds = %cond.true40, %cond.false45, %cond.true29, %cond.false30
  %cond53 = phi double [ %13, %cond.true29 ], [ %14, %cond.false30 ], [ %15, %cond.true40 ], [ %16, %cond.false45 ]
  %mul54 = mul nsw i32 %c.0, %9
  %add55 = add nsw i32 %p.0, %mul54
  %idxprom56 = sext i32 %add55 to i64
  %arrayidx57 = getelementptr inbounds double, ptr %B, i64 %idxprom56
  %17 = load double, ptr %arrayidx57, align 8
  %mul58 = fmul double %cond53, %17
  %add59 = fadd double %sum.0, %mul58
  br label %for.inc

if.else:                                          ; preds = %for.body23
  %tobool63 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool63, label %cond.true64, label %cond.false79

cond.true64:                                      ; preds = %if.else
  %cmp65 = icmp sle i32 %p.0, %c.0
  br i1 %cmp65, label %cond.true67, label %cond.false72

cond.true67:                                      ; preds = %cond.true64
  %mul68 = mul nsw i32 %c.0, %8
  %add69 = add nsw i32 %p.0, %mul68
  %idxprom70 = sext i32 %add69 to i64
  %arrayidx71 = getelementptr inbounds double, ptr %A, i64 %idxprom70
  %18 = load double, ptr %arrayidx71, align 8
  br label %cond.end94

cond.false72:                                     ; preds = %cond.true64
  %mul73 = mul nsw i32 %p.0, %8
  %add74 = add nsw i32 %c.0, %mul73
  %idxprom75 = sext i32 %add74 to i64
  %arrayidx76 = getelementptr inbounds double, ptr %A, i64 %idxprom75
  %19 = load double, ptr %arrayidx76, align 8
  br label %cond.end94

cond.false79:                                     ; preds = %if.else
  %cmp80 = icmp sge i32 %p.0, %c.0
  br i1 %cmp80, label %cond.true82, label %cond.false87

cond.true82:                                      ; preds = %cond.false79
  %mul83 = mul nsw i32 %c.0, %8
  %add84 = add nsw i32 %p.0, %mul83
  %idxprom85 = sext i32 %add84 to i64
  %arrayidx86 = getelementptr inbounds double, ptr %A, i64 %idxprom85
  %20 = load double, ptr %arrayidx86, align 8
  br label %cond.end94

cond.false87:                                     ; preds = %cond.false79
  %mul88 = mul nsw i32 %p.0, %8
  %add89 = add nsw i32 %c.0, %mul88
  %idxprom90 = sext i32 %add89 to i64
  %arrayidx91 = getelementptr inbounds double, ptr %A, i64 %idxprom90
  %21 = load double, ptr %arrayidx91, align 8
  br label %cond.end94

cond.end94:                                       ; preds = %cond.true82, %cond.false87, %cond.true67, %cond.false72
  %cond95 = phi double [ %18, %cond.true67 ], [ %19, %cond.false72 ], [ %20, %cond.true82 ], [ %21, %cond.false87 ]
  %mul96 = mul nsw i32 %p.0, %9
  %add97 = add nsw i32 %r.0, %mul96
  %idxprom98 = sext i32 %add97 to i64
  %arrayidx99 = getelementptr inbounds double, ptr %B, i64 %idxprom98
  %22 = load double, ptr %arrayidx99, align 8
  %mul100 = fmul double %22, %cond95
  %add101 = fadd double %sum.0, %mul100
  br label %for.inc

for.inc:                                          ; preds = %cond.end52, %cond.end94
  %sum.1 = phi double [ %add59, %cond.end52 ], [ %add101, %cond.end94 ]
  %inc = add nsw i32 %p.0, 1
  br label %for.cond20

for.end:                                          ; preds = %for.cond20
  %mul102 = fmul double %11, %sum.0
  %mul103 = mul nsw i32 %c.0, %10
  %add104 = add nsw i32 %r.0, %mul103
  %idxprom105 = sext i32 %add104 to i64
  %arrayidx106 = getelementptr inbounds double, ptr %C, i64 %idxprom105
  %23 = load double, ptr %arrayidx106, align 8
  %mul107 = fmul double %12, %23
  %add108 = fadd double %mul102, %mul107
  %mul109 = mul nsw i32 %c.0, %10
  %add110 = add nsw i32 %r.0, %mul109
  %idxprom111 = sext i32 %add110 to i64
  %arrayidx112 = getelementptr inbounds double, ptr %C, i64 %idxprom111
  store double %add108, ptr %arrayidx112, align 8
  %inc114 = add nsw i32 %r.0, 1
  br label %for.cond16

for.inc116:                                       ; preds = %for.cond16
  %inc117 = add nsw i32 %c.0, 1
  br label %for.cond

for.end118:                                       ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @dsyrk_(ptr noundef %uplop, ptr noundef %transp, ptr noundef %np, ptr noundef %kp, ptr noundef %alphap, ptr noundef %A, ptr noundef %ldap, ptr noundef %betap, ptr noundef %C, ptr noundef %ldcp, i32 noundef %len1, i32 noundef %len2) noinline {
entry:
  %0 = load i8, ptr %uplop, align 1
  %conv = sext i8 %0 to i32
  %cmp = icmp eq i32 %conv, 85
  br i1 %cmp, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  %1 = load i8, ptr %uplop, align 1
  %conv2 = sext i8 %1 to i32
  %cmp3 = icmp eq i32 %conv2, 117
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %2 = phi i1 [ true, %entry ], [ %cmp3, %lor.rhs ]
  %lor.ext = zext i1 %2 to i32
  %3 = load i8, ptr %transp, align 1
  %conv5 = sext i8 %3 to i32
  %cmp6 = icmp eq i32 %conv5, 84
  br i1 %cmp6, label %lor.end12, label %lor.rhs8

lor.rhs8:                                         ; preds = %lor.end
  %4 = load i8, ptr %transp, align 1
  %conv9 = sext i8 %4 to i32
  %cmp10 = icmp eq i32 %conv9, 116
  br label %lor.end12

lor.end12:                                        ; preds = %lor.rhs8, %lor.end
  %5 = phi i1 [ true, %lor.end ], [ %cmp10, %lor.rhs8 ]
  %lor.ext13 = zext i1 %5 to i32
  %6 = load i32, ptr %np, align 4
  %7 = load i32, ptr %kp, align 4
  %8 = load i32, ptr %ldap, align 4
  %9 = load i32, ptr %ldcp, align 4
  %10 = load double, ptr %alphap, align 8
  %11 = load double, ptr %betap, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc66, %lor.end12
  %c.0 = phi i32 [ 0, %lor.end12 ], [ %inc67, %for.inc66 ]
  %cmp14 = icmp slt i32 %c.0, %6
  br i1 %cmp14, label %for.cond16, label %for.end68

for.cond16:                                       ; preds = %for.cond, %for.inc63
  %r.0 = phi i32 [ %inc64, %for.inc63 ], [ 0, %for.cond ]
  %cmp17 = icmp slt i32 %r.0, %6
  br i1 %cmp17, label %for.body19, label %for.inc66

for.body19:                                       ; preds = %for.cond16
  %tobool = icmp ne i32 %lor.ext, 0
  %cmp20 = icmp sgt i32 %r.0, %c.0
  %or.cond = and i1 %tobool, %cmp20
  br i1 %or.cond, label %for.inc63, label %if.end

if.end:                                           ; preds = %for.body19
  %tobool22 = icmp eq i32 %lor.ext, 0
  %cmp24 = icmp slt i32 %r.0, %c.0
  %or.cond1 = and i1 %tobool22, %cmp24
  br i1 %or.cond1, label %for.inc63, label %for.cond28

for.cond28:                                       ; preds = %if.end, %cond.end48
  %sum.0 = phi double [ %add51, %cond.end48 ], [ 0.000000e+00, %if.end ]
  %p.0 = phi i32 [ %inc, %cond.end48 ], [ 0, %if.end ]
  %cmp29 = icmp slt i32 %p.0, %7
  br i1 %cmp29, label %for.body31, label %for.end

for.body31:                                       ; preds = %for.cond28
  %tobool32 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool32, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body31
  %mul = mul nsw i32 %r.0, %8
  %add = add nsw i32 %p.0, %mul
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %12 = load double, ptr %arrayidx, align 8
  br label %cond.end

cond.false:                                       ; preds = %for.body31
  %mul33 = mul nsw i32 %p.0, %8
  %add34 = add nsw i32 %r.0, %mul33
  %idxprom35 = sext i32 %add34 to i64
  %arrayidx36 = getelementptr inbounds double, ptr %A, i64 %idxprom35
  %13 = load double, ptr %arrayidx36, align 8
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ %12, %cond.true ], [ %13, %cond.false ]
  %tobool37 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool37, label %cond.true38, label %cond.false43

cond.true38:                                      ; preds = %cond.end
  %mul39 = mul nsw i32 %c.0, %8
  %add40 = add nsw i32 %p.0, %mul39
  %idxprom41 = sext i32 %add40 to i64
  %arrayidx42 = getelementptr inbounds double, ptr %A, i64 %idxprom41
  %14 = load double, ptr %arrayidx42, align 8
  br label %cond.end48

cond.false43:                                     ; preds = %cond.end
  %mul44 = mul nsw i32 %p.0, %8
  %add45 = add nsw i32 %c.0, %mul44
  %idxprom46 = sext i32 %add45 to i64
  %arrayidx47 = getelementptr inbounds double, ptr %A, i64 %idxprom46
  %15 = load double, ptr %arrayidx47, align 8
  br label %cond.end48

cond.end48:                                       ; preds = %cond.false43, %cond.true38
  %cond49 = phi double [ %14, %cond.true38 ], [ %15, %cond.false43 ]
  %mul50 = fmul double %cond, %cond49
  %add51 = fadd double %sum.0, %mul50
  %inc = add nsw i32 %p.0, 1
  br label %for.cond28

for.end:                                          ; preds = %for.cond28
  %mul52 = fmul double %10, %sum.0
  %mul53 = mul nsw i32 %c.0, %9
  %add54 = add nsw i32 %r.0, %mul53
  %idxprom55 = sext i32 %add54 to i64
  %arrayidx56 = getelementptr inbounds double, ptr %C, i64 %idxprom55
  %16 = load double, ptr %arrayidx56, align 8
  %mul57 = fmul double %11, %16
  %add58 = fadd double %mul52, %mul57
  %mul59 = mul nsw i32 %c.0, %9
  %add60 = add nsw i32 %r.0, %mul59
  %idxprom61 = sext i32 %add60 to i64
  %arrayidx62 = getelementptr inbounds double, ptr %C, i64 %idxprom61
  store double %add58, ptr %arrayidx62, align 8
  br label %for.inc63

for.inc63:                                        ; preds = %if.end, %for.body19, %for.end
  %inc64 = add nsw i32 %r.0, 1
  br label %for.cond16

for.inc66:                                        ; preds = %for.cond16
  %inc67 = add nsw i32 %c.0, 1
  br label %for.cond

for.end68:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @cblas_dsyrk(i32 noundef %layout, i32 noundef %uplo, i32 noundef %trans, i32 noundef %n, i32 noundef %k, double noundef %alpha, ptr noundef %A, i32 noundef %lda, double noundef %beta, ptr noundef %C, i32 noundef %ldc) noinline {
entry:
  %cmp = icmp eq i32 %layout, 101
  %conv = zext i1 %cmp to i32
  %cmp1 = icmp eq i32 %uplo, 121
  %conv2 = zext i1 %cmp1 to i32
  %cmp3 = icmp eq i32 %trans, 112
  %conv4 = zext i1 %cmp3 to i32
  br label %for.cond

for.cond:                                         ; preds = %for.inc78, %entry
  %r.0 = phi i32 [ 0, %entry ], [ %inc79, %for.inc78 ]
  %cmp5 = icmp slt i32 %r.0, %n
  br i1 %cmp5, label %for.cond7, label %for.end80

for.cond7:                                        ; preds = %for.cond, %for.inc75
  %c.0 = phi i32 [ %inc76, %for.inc75 ], [ 0, %for.cond ]
  %cmp8 = icmp slt i32 %c.0, %n
  br i1 %cmp8, label %for.body10, label %for.inc78

for.body10:                                       ; preds = %for.cond7
  %tobool = icmp ne i32 %conv2, 0
  %cmp11 = icmp sgt i32 %r.0, %c.0
  %or.cond = and i1 %tobool, %cmp11
  br i1 %or.cond, label %for.inc75, label %if.end

if.end:                                           ; preds = %for.body10
  %tobool13 = icmp eq i32 %conv2, 0
  %cmp15 = icmp slt i32 %r.0, %c.0
  %or.cond1 = and i1 %tobool13, %cmp15
  br i1 %or.cond1, label %for.inc75, label %for.cond19

for.cond19:                                       ; preds = %if.end, %for.body22
  %sum.0 = phi double [ %add58, %for.body22 ], [ 0.000000e+00, %if.end ]
  %p.0 = phi i32 [ %inc, %for.body22 ], [ 0, %if.end ]
  %cmp20 = icmp slt i32 %p.0, %k
  br i1 %cmp20, label %for.body22, label %for.end

for.body22:                                       ; preds = %for.cond19
  %tobool23 = icmp ne i32 %conv4, 0
  %p.0.r.0 = select i1 %tobool23, i32 %p.0, i32 %r.0
  %tobool24 = icmp ne i32 %conv4, 0
  %cond28 = select i1 %tobool24, i32 %r.0, i32 %p.0
  %tobool29 = icmp ne i32 %conv4, 0
  %cond33 = select i1 %tobool29, i32 %p.0, i32 %c.0
  %tobool34 = icmp ne i32 %conv4, 0
  %cond38 = select i1 %tobool34, i32 %c.0, i32 %p.0
  %tobool39 = icmp ne i32 %conv, 0
  %mul = mul nsw i32 %p.0.r.0, %lda
  %add = add nsw i32 %mul, %cond28
  %mul42 = mul nsw i32 %cond28, %lda
  %add43 = add nsw i32 %p.0.r.0, %mul42
  %cond45 = select i1 %tobool39, i32 %add, i32 %add43
  %idxprom = sext i32 %cond45 to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8
  %tobool46 = icmp ne i32 %conv, 0
  %mul48 = mul nsw i32 %cond33, %lda
  %add49 = add nsw i32 %mul48, %cond38
  %mul51 = mul nsw i32 %cond38, %lda
  %add52 = add nsw i32 %cond33, %mul51
  %cond54 = select i1 %tobool46, i32 %add49, i32 %add52
  %idxprom55 = sext i32 %cond54 to i64
  %arrayidx56 = getelementptr inbounds double, ptr %A, i64 %idxprom55
  %1 = load double, ptr %arrayidx56, align 8
  %mul57 = fmul double %0, %1
  %add58 = fadd double %sum.0, %mul57
  %inc = add nsw i32 %p.0, 1
  br label %for.cond19

for.end:                                          ; preds = %for.cond19
  %tobool59 = icmp ne i32 %conv, 0
  %mul61 = mul nsw i32 %r.0, %ldc
  %add62 = add nsw i32 %mul61, %c.0
  %mul64 = mul nsw i32 %c.0, %ldc
  %add65 = add nsw i32 %r.0, %mul64
  %cond67 = select i1 %tobool59, i32 %add62, i32 %add65
  %mul68 = fmul double %alpha, %sum.0
  %idxprom69 = sext i32 %cond67 to i64
  %arrayidx70 = getelementptr inbounds double, ptr %C, i64 %idxprom69
  %2 = load double, ptr %arrayidx70, align 8
  %mul71 = fmul double %beta, %2
  %add72 = fadd double %mul68, %mul71
  %idxprom73 = sext i32 %cond67 to i64
  %arrayidx74 = getelementptr inbounds double, ptr %C, i64 %idxprom73
  store double %add72, ptr %arrayidx74, align 8
  br label %for.inc75

for.inc75:                                        ; preds = %if.end, %for.body10, %for.end
  %inc76 = add nsw i32 %c.0, 1
  br label %for.cond7

for.inc78:                                        ; preds = %for.cond7
  %inc79 = add nsw i32 %r.0, 1
  br label %for.cond

for.end80:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @dsyr2k_(ptr noundef %uplop, ptr noundef %transp, ptr noundef %np, ptr noundef %kp, ptr noundef %alphap, ptr noundef %A, ptr noundef %ldap, ptr noundef %B, ptr noundef %ldbp, ptr noundef %betap, ptr noundef %C, ptr noundef %ldcp, i32 noundef %len1, i32 noundef %len2) noinline {
entry:
  %0 = load i8, ptr %uplop, align 1
  %conv = sext i8 %0 to i32
  %cmp = icmp eq i32 %conv, 85
  br i1 %cmp, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  %1 = load i8, ptr %uplop, align 1
  %conv2 = sext i8 %1 to i32
  %cmp3 = icmp eq i32 %conv2, 117
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %2 = phi i1 [ true, %entry ], [ %cmp3, %lor.rhs ]
  %lor.ext = zext i1 %2 to i32
  %3 = load i8, ptr %transp, align 1
  %conv5 = sext i8 %3 to i32
  %cmp6 = icmp eq i32 %conv5, 84
  br i1 %cmp6, label %lor.end12, label %lor.rhs8

lor.rhs8:                                         ; preds = %lor.end
  %4 = load i8, ptr %transp, align 1
  %conv9 = sext i8 %4 to i32
  %cmp10 = icmp eq i32 %conv9, 116
  br label %lor.end12

lor.end12:                                        ; preds = %lor.rhs8, %lor.end
  %5 = phi i1 [ true, %lor.end ], [ %cmp10, %lor.rhs8 ]
  %lor.ext13 = zext i1 %5 to i32
  %6 = load i32, ptr %np, align 4
  %7 = load i32, ptr %kp, align 4
  %8 = load i32, ptr %ldap, align 4
  %9 = load i32, ptr %ldbp, align 4
  %10 = load i32, ptr %ldcp, align 4
  %11 = load double, ptr %alphap, align 8
  %12 = load double, ptr %betap, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc94, %lor.end12
  %c.0 = phi i32 [ 0, %lor.end12 ], [ %inc95, %for.inc94 ]
  %cmp14 = icmp slt i32 %c.0, %6
  br i1 %cmp14, label %for.cond16, label %for.end96

for.cond16:                                       ; preds = %for.cond, %for.inc91
  %r.0 = phi i32 [ %inc92, %for.inc91 ], [ 0, %for.cond ]
  %cmp17 = icmp slt i32 %r.0, %6
  br i1 %cmp17, label %for.body19, label %for.inc94

for.body19:                                       ; preds = %for.cond16
  %tobool = icmp ne i32 %lor.ext, 0
  %cmp20 = icmp sgt i32 %r.0, %c.0
  %or.cond = and i1 %tobool, %cmp20
  br i1 %or.cond, label %for.inc91, label %if.end

if.end:                                           ; preds = %for.body19
  %tobool22 = icmp eq i32 %lor.ext, 0
  %cmp24 = icmp slt i32 %r.0, %c.0
  %or.cond1 = and i1 %tobool22, %cmp24
  br i1 %or.cond1, label %for.inc91, label %for.cond28

for.cond28:                                       ; preds = %if.end, %cond.end74
  %sum.0 = phi double [ %add79, %cond.end74 ], [ 0.000000e+00, %if.end ]
  %p.0 = phi i32 [ %inc, %cond.end74 ], [ 0, %if.end ]
  %cmp29 = icmp slt i32 %p.0, %7
  br i1 %cmp29, label %for.body31, label %for.end

for.body31:                                       ; preds = %for.cond28
  %tobool32 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool32, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body31
  %mul = mul nsw i32 %r.0, %8
  %add = add nsw i32 %p.0, %mul
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %13 = load double, ptr %arrayidx, align 8
  br label %cond.end

cond.false:                                       ; preds = %for.body31
  %mul33 = mul nsw i32 %p.0, %8
  %add34 = add nsw i32 %r.0, %mul33
  %idxprom35 = sext i32 %add34 to i64
  %arrayidx36 = getelementptr inbounds double, ptr %A, i64 %idxprom35
  %14 = load double, ptr %arrayidx36, align 8
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ %13, %cond.true ], [ %14, %cond.false ]
  %tobool37 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool37, label %cond.true38, label %cond.false43

cond.true38:                                      ; preds = %cond.end
  %mul39 = mul nsw i32 %c.0, %9
  %add40 = add nsw i32 %p.0, %mul39
  %idxprom41 = sext i32 %add40 to i64
  %arrayidx42 = getelementptr inbounds double, ptr %B, i64 %idxprom41
  %15 = load double, ptr %arrayidx42, align 8
  br label %cond.end48

cond.false43:                                     ; preds = %cond.end
  %mul44 = mul nsw i32 %p.0, %9
  %add45 = add nsw i32 %c.0, %mul44
  %idxprom46 = sext i32 %add45 to i64
  %arrayidx47 = getelementptr inbounds double, ptr %B, i64 %idxprom46
  %16 = load double, ptr %arrayidx47, align 8
  br label %cond.end48

cond.end48:                                       ; preds = %cond.false43, %cond.true38
  %cond49 = phi double [ %15, %cond.true38 ], [ %16, %cond.false43 ]
  %tobool50 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool50, label %cond.true51, label %cond.false56

cond.true51:                                      ; preds = %cond.end48
  %mul52 = mul nsw i32 %r.0, %9
  %add53 = add nsw i32 %p.0, %mul52
  %idxprom54 = sext i32 %add53 to i64
  %arrayidx55 = getelementptr inbounds double, ptr %B, i64 %idxprom54
  %17 = load double, ptr %arrayidx55, align 8
  br label %cond.end61

cond.false56:                                     ; preds = %cond.end48
  %mul57 = mul nsw i32 %p.0, %9
  %add58 = add nsw i32 %r.0, %mul57
  %idxprom59 = sext i32 %add58 to i64
  %arrayidx60 = getelementptr inbounds double, ptr %B, i64 %idxprom59
  %18 = load double, ptr %arrayidx60, align 8
  br label %cond.end61

cond.end61:                                       ; preds = %cond.false56, %cond.true51
  %cond62 = phi double [ %17, %cond.true51 ], [ %18, %cond.false56 ]
  %tobool63 = icmp ne i32 %lor.ext13, 0
  br i1 %tobool63, label %cond.true64, label %cond.false69

cond.true64:                                      ; preds = %cond.end61
  %mul65 = mul nsw i32 %c.0, %8
  %add66 = add nsw i32 %p.0, %mul65
  %idxprom67 = sext i32 %add66 to i64
  %arrayidx68 = getelementptr inbounds double, ptr %A, i64 %idxprom67
  %19 = load double, ptr %arrayidx68, align 8
  br label %cond.end74

cond.false69:                                     ; preds = %cond.end61
  %mul70 = mul nsw i32 %p.0, %8
  %add71 = add nsw i32 %c.0, %mul70
  %idxprom72 = sext i32 %add71 to i64
  %arrayidx73 = getelementptr inbounds double, ptr %A, i64 %idxprom72
  %20 = load double, ptr %arrayidx73, align 8
  br label %cond.end74

cond.end74:                                       ; preds = %cond.false69, %cond.true64
  %cond75 = phi double [ %19, %cond.true64 ], [ %20, %cond.false69 ]
  %mul76 = fmul double %cond, %cond49
  %mul77 = fmul double %cond62, %cond75
  %add78 = fadd double %mul76, %mul77
  %add79 = fadd double %sum.0, %add78
  %inc = add nsw i32 %p.0, 1
  br label %for.cond28

for.end:                                          ; preds = %for.cond28
  %mul80 = fmul double %11, %sum.0
  %mul81 = mul nsw i32 %c.0, %10
  %add82 = add nsw i32 %r.0, %mul81
  %idxprom83 = sext i32 %add82 to i64
  %arrayidx84 = getelementptr inbounds double, ptr %C, i64 %idxprom83
  %21 = load double, ptr %arrayidx84, align 8
  %mul85 = fmul double %12, %21
  %add86 = fadd double %mul80, %mul85
  %mul87 = mul nsw i32 %c.0, %10
  %add88 = add nsw i32 %r.0, %mul87
  %idxprom89 = sext i32 %add88 to i64
  %arrayidx90 = getelementptr inbounds double, ptr %C, i64 %idxprom89
  store double %add86, ptr %arrayidx90, align 8
  br label %for.inc91

for.inc91:                                        ; preds = %if.end, %for.body19, %for.end
  %inc92 = add nsw i32 %r.0, 1
  br label %for.cond16

for.inc94:                                        ; preds = %for.cond16
  %inc95 = add nsw i32 %c.0, 1
  br label %for.cond

for.end96:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @primal_dgemm(ptr noundef %alpha, ptr noundef %A, ptr noundef %B, ptr noundef %beta, ptr noundef %C) noinline {
entry:
  %ta = alloca i8, align 1
  %tb = alloca i8, align 1
  %m = alloca i32, align 4
  %n = alloca i32, align 4
  %k = alloca i32, align 4
  %lda = alloca i32, align 4
  %ldb = alloca i32, align 4
  %ldc = alloca i32, align 4
  store i8 78, ptr %ta, align 1
  store i8 78, ptr %tb, align 1
  store i32 2, ptr %m, align 4
  store i32 3, ptr %n, align 4
  store i32 2, ptr %k, align 4
  store i32 3, ptr %lda, align 4
  store i32 3, ptr %ldb, align 4
  store i32 4, ptr %ldc, align 4
  call void @dgemm_(ptr noundef %ta, ptr noundef %tb, ptr noundef %m, ptr noundef %n, ptr noundef %k, ptr noundef %alpha, ptr noundef %A, ptr noundef %lda, ptr noundef %B, ptr noundef %ldb, ptr noundef %beta, ptr noundef %C, ptr noundef %ldc, i32 noundef 1, i32 noundef 1)
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @primal_cblas_rowmajor(double noundef %alpha, ptr noundef %A, ptr noundef %B, double noundef %beta, ptr noundef %C) noinline {
entry:
  call void @cblas_dgemm(i32 noundef 101, i32 noundef 111, i32 noundef 111, i32 noundef 2, i32 noundef 3, i32 noundef 2, double noundef %alpha, ptr noundef %A, i32 noundef 2, ptr noundef %B, i32 noundef 3, double noundef %beta, ptr noundef %C, i32 noundef 5)
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @primal_dsymm(ptr noundef %alpha, ptr noundef %A, ptr noundef %B, ptr noundef %beta, ptr noundef %C) noinline {
entry:
  %side = alloca i8, align 1
  %uplo = alloca i8, align 1
  %m = alloca i32, align 4
  %n = alloca i32, align 4
  %lda = alloca i32, align 4
  %ldb = alloca i32, align 4
  %ldc = alloca i32, align 4
  store i8 76, ptr %side, align 1
  store i8 85, ptr %uplo, align 1
  store i32 2, ptr %m, align 4
  store i32 3, ptr %n, align 4
  store i32 3, ptr %lda, align 4
  store i32 3, ptr %ldb, align 4
  store i32 4, ptr %ldc, align 4
  call void @dsymm_(ptr noundef %side, ptr noundef %uplo, ptr noundef %m, ptr noundef %n, ptr noundef %alpha, ptr noundef %A, ptr noundef %lda, ptr noundef %B, ptr noundef %ldb, ptr noundef %beta, ptr noundef %C, ptr noundef %ldc, i32 noundef 1, i32 noundef 1)
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @primal_dsyrk(i32 noundef %uplo_val, ptr noundef %alpha, ptr noundef %A, ptr noundef %beta, ptr noundef %C) noinline {
entry:
  %uplo = alloca i8, align 1
  %trans = alloca i8, align 1
  %n = alloca i32, align 4
  %k = alloca i32, align 4
  %lda = alloca i32, align 4
  %ldc = alloca i32, align 4
  %conv = trunc i32 %uplo_val to i8
  store i8 %conv, ptr %uplo, align 1
  store i8 78, ptr %trans, align 1
  store i32 3, ptr %n, align 4
  store i32 2, ptr %k, align 4
  store i32 4, ptr %lda, align 4
  store i32 5, ptr %ldc, align 4
  call void @dsyrk_(ptr noundef %uplo, ptr noundef %trans, ptr noundef %n, ptr noundef %k, ptr noundef %alpha, ptr noundef %A, ptr noundef %lda, ptr noundef %beta, ptr noundef %C, ptr noundef %ldc, i32 noundef 1, i32 noundef 1)
  ret void
}

; Function Attrs: noinline nounwind uwtable
define void @primal_cblas_dsyrk_rowmajor(i32 noundef %uplo, double noundef %alpha, ptr noundef %A, double noundef %beta, ptr noundef %C) noinline {
entry:
  call void @cblas_dsyrk(i32 noundef 101, i32 noundef %uplo, i32 noundef 111, i32 noundef 3, i32 noundef 2, double noundef %alpha, ptr noundef %A, i32 noundef 2, double noundef %beta, ptr noundef %C, i32 noundef 5)
  ret void
}

; Function Attrs: nounwind uwtable
define i32 @main() {
entry:
  %alpha = alloca double, align 8
  %dalpha = alloca double, align 8
  %beta = alloca double, align 8
  %dbeta = alloca double, align 8
  %C = alloca [12 x double], align 16
  %dC = alloca [12 x double], align 16
  %Cp = alloca [12 x double], align 16
  %Cm = alloca [12 x double], align 16
  %ap = alloca double, align 8
  %am = alloca double, align 8
  %bp = alloca double, align 8
  %bm = alloca double, align 8
  %Ap = alloca [6 x double], align 16
  %Am = alloca [6 x double], align 16
  %Bp = alloca [9 x double], align 16
  %Bm = alloca [9 x double], align 16
  %C109 = alloca [10 x double], align 16
  %dC110 = alloca [10 x double], align 16
  %alpha160 = alloca double, align 8
  %dalpha161 = alloca double, align 8
  %beta162 = alloca double, align 8
  %dbeta163 = alloca double, align 8
  %C164 = alloca [12 x double], align 16
  %dC165 = alloca [12 x double], align 16
  %Cp166 = alloca [12 x double], align 16
  %Cm167 = alloca [12 x double], align 16
  %ap202 = alloca double, align 8
  %am205 = alloca double, align 8
  %bp208 = alloca double, align 8
  %bm211 = alloca double, align 8
  %Ap214 = alloca [6 x double], align 16
  %Am215 = alloca [6 x double], align 16
  %Bp216 = alloca [9 x double], align 16
  %Bm217 = alloca [9 x double], align 16
  %alpha306 = alloca double, align 8
  %dalpha307 = alloca double, align 8
  %beta308 = alloca double, align 8
  %dbeta309 = alloca double, align 8
  %C_init = alloca [15 x double], align 16
  %dC_init = alloca [15 x double], align 16
  %C310 = alloca [15 x double], align 16
  %dC311 = alloca [15 x double], align 16
  %Cp312 = alloca [15 x double], align 16
  %Cm313 = alloca [15 x double], align 16
  %ap415 = alloca double, align 8
  %am418 = alloca double, align 8
  %bp421 = alloca double, align 8
  %bm424 = alloca double, align 8
  %Ap427 = alloca [8 x double], align 16
  %Am428 = alloca [8 x double], align 16
  %C_init533 = alloca [15 x double], align 16
  %dC_init534 = alloca [15 x double], align 16
  %C535 = alloca [15 x double], align 16
  %dC536 = alloca [15 x double], align 16
  %Cp537 = alloca [15 x double], align 16
  %Cm538 = alloca [15 x double], align 16
  store double 1.700000e+00, ptr %alpha, align 8
  store double -3.000000e-01, ptr %dalpha, align 8
  store double 2.500000e+00, ptr %beta, align 8
  store double 0x3FE6666666666666, ptr %dbeta, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 12
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [12 x double], ptr @main.C_init, i64 0, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8
  %idxprom1 = sext i32 %i.0 to i64
  %arrayidx2 = getelementptr inbounds [12 x double], ptr %C, i64 0, i64 %idxprom1
  store double %0, ptr %arrayidx2, align 8
  %idxprom3 = sext i32 %i.0 to i64
  %arrayidx4 = getelementptr inbounds [12 x double], ptr @main.dC_init, i64 0, i64 %idxprom3
  %1 = load double, ptr %arrayidx4, align 8
  %idxprom5 = sext i32 %i.0 to i64
  %arrayidx6 = getelementptr inbounds [12 x double], ptr %dC, i64 0, i64 %idxprom5
  store double %1, ptr %arrayidx6, align 8
  %idxprom7 = sext i32 %i.0 to i64
  %arrayidx8 = getelementptr inbounds [12 x double], ptr @main.C_init, i64 0, i64 %idxprom7
  %2 = load double, ptr %arrayidx8, align 8
  %idxprom9 = sext i32 %i.0 to i64
  %arrayidx10 = getelementptr inbounds [12 x double], ptr @main.dC_init, i64 0, i64 %idxprom9
  %3 = load double, ptr %arrayidx10, align 8
  %mul = fmul double 0x3EB0C6F7A0B5ED8D, %3
  %add = fadd double %2, %mul
  %idxprom11 = sext i32 %i.0 to i64
  %arrayidx12 = getelementptr inbounds [12 x double], ptr %Cp, i64 0, i64 %idxprom11
  store double %add, ptr %arrayidx12, align 8
  %idxprom13 = sext i32 %i.0 to i64
  %arrayidx14 = getelementptr inbounds [12 x double], ptr @main.C_init, i64 0, i64 %idxprom13
  %4 = load double, ptr %arrayidx14, align 8
  %idxprom15 = sext i32 %i.0 to i64
  %arrayidx16 = getelementptr inbounds [12 x double], ptr @main.dC_init, i64 0, i64 %idxprom15
  %5 = load double, ptr %arrayidx16, align 8
  %mul17 = fmul double 0x3EB0C6F7A0B5ED8D, %5
  %sub = fsub double %4, %mul17
  %idxprom18 = sext i32 %i.0 to i64
  %arrayidx19 = getelementptr inbounds [12 x double], ptr %Cm, i64 0, i64 %idxprom18
  store double %sub, ptr %arrayidx19, align 8
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, ptr @enzyme_dup, align 4
  %7 = load i32, ptr @enzyme_dup, align 4
  %8 = load i32, ptr @enzyme_dup, align 4
  %9 = load i32, ptr @enzyme_dup, align 4
  %10 = load i32, ptr @enzyme_dup, align 4
  %arraydecay = getelementptr inbounds [12 x double], ptr %C, i64 0, i64 0
  %arraydecay20 = getelementptr inbounds [12 x double], ptr %dC, i64 0, i64 0
  call void (ptr, ...) @__enzyme_fwddiff(ptr noundef @primal_dgemm, i32 noundef %6, ptr noundef %alpha, ptr noundef %dalpha, i32 noundef %7, ptr noundef @main.A, ptr noundef @main.dA, i32 noundef %8, ptr noundef @main.B, ptr noundef @main.dB, i32 noundef %9, ptr noundef %beta, ptr noundef %dbeta, i32 noundef %10, ptr noundef %arraydecay, ptr noundef %arraydecay20)
  %11 = load double, ptr %alpha, align 8
  %12 = load double, ptr %dalpha, align 8
  %mul21 = fmul double 0x3EB0C6F7A0B5ED8D, %12
  %add22 = fadd double %11, %mul21
  store double %add22, ptr %ap, align 8
  %13 = load double, ptr %alpha, align 8
  %14 = load double, ptr %dalpha, align 8
  %mul23 = fmul double 0x3EB0C6F7A0B5ED8D, %14
  %sub24 = fsub double %13, %mul23
  store double %sub24, ptr %am, align 8
  %15 = load double, ptr %beta, align 8
  %16 = load double, ptr %dbeta, align 8
  %mul25 = fmul double 0x3EB0C6F7A0B5ED8D, %16
  %add26 = fadd double %15, %mul25
  store double %add26, ptr %bp, align 8
  %17 = load double, ptr %beta, align 8
  %18 = load double, ptr %dbeta, align 8
  %mul27 = fmul double 0x3EB0C6F7A0B5ED8D, %18
  %sub28 = fsub double %17, %mul27
  store double %sub28, ptr %bm, align 8
  br label %for.cond30

for.cond30:                                       ; preds = %for.body32, %for.end
  %i29.0 = phi i32 [ 0, %for.end ], [ %inc50, %for.body32 ]
  %cmp31 = icmp slt i32 %i29.0, 6
  br i1 %cmp31, label %for.body32, label %for.cond53

for.body32:                                       ; preds = %for.cond30
  %idxprom33 = sext i32 %i29.0 to i64
  %arrayidx34 = getelementptr inbounds [6 x double], ptr @main.A, i64 0, i64 %idxprom33
  %19 = load double, ptr %arrayidx34, align 8
  %idxprom35 = sext i32 %i29.0 to i64
  %arrayidx36 = getelementptr inbounds [6 x double], ptr @main.dA, i64 0, i64 %idxprom35
  %20 = load double, ptr %arrayidx36, align 8
  %mul37 = fmul double 0x3EB0C6F7A0B5ED8D, %20
  %add38 = fadd double %19, %mul37
  %idxprom39 = sext i32 %i29.0 to i64
  %arrayidx40 = getelementptr inbounds [6 x double], ptr %Ap, i64 0, i64 %idxprom39
  store double %add38, ptr %arrayidx40, align 8
  %idxprom41 = sext i32 %i29.0 to i64
  %arrayidx42 = getelementptr inbounds [6 x double], ptr @main.A, i64 0, i64 %idxprom41
  %21 = load double, ptr %arrayidx42, align 8
  %idxprom43 = sext i32 %i29.0 to i64
  %arrayidx44 = getelementptr inbounds [6 x double], ptr @main.dA, i64 0, i64 %idxprom43
  %22 = load double, ptr %arrayidx44, align 8
  %mul45 = fmul double 0x3EB0C6F7A0B5ED8D, %22
  %sub46 = fsub double %21, %mul45
  %idxprom47 = sext i32 %i29.0 to i64
  %arrayidx48 = getelementptr inbounds [6 x double], ptr %Am, i64 0, i64 %idxprom47
  store double %sub46, ptr %arrayidx48, align 8
  %inc50 = add nsw i32 %i29.0, 1
  br label %for.cond30

for.cond53:                                       ; preds = %for.cond30, %for.body55
  %i52.0 = phi i32 [ %inc73, %for.body55 ], [ 0, %for.cond30 ]
  %cmp54 = icmp slt i32 %i52.0, 9
  br i1 %cmp54, label %for.body55, label %for.end74

for.body55:                                       ; preds = %for.cond53
  %idxprom56 = sext i32 %i52.0 to i64
  %arrayidx57 = getelementptr inbounds [9 x double], ptr @main.B, i64 0, i64 %idxprom56
  %23 = load double, ptr %arrayidx57, align 8
  %idxprom58 = sext i32 %i52.0 to i64
  %arrayidx59 = getelementptr inbounds [9 x double], ptr @main.dB, i64 0, i64 %idxprom58
  %24 = load double, ptr %arrayidx59, align 8
  %mul60 = fmul double 0x3EB0C6F7A0B5ED8D, %24
  %add61 = fadd double %23, %mul60
  %idxprom62 = sext i32 %i52.0 to i64
  %arrayidx63 = getelementptr inbounds [9 x double], ptr %Bp, i64 0, i64 %idxprom62
  store double %add61, ptr %arrayidx63, align 8
  %idxprom64 = sext i32 %i52.0 to i64
  %arrayidx65 = getelementptr inbounds [9 x double], ptr @main.B, i64 0, i64 %idxprom64
  %25 = load double, ptr %arrayidx65, align 8
  %idxprom66 = sext i32 %i52.0 to i64
  %arrayidx67 = getelementptr inbounds [9 x double], ptr @main.dB, i64 0, i64 %idxprom66
  %26 = load double, ptr %arrayidx67, align 8
  %mul68 = fmul double 0x3EB0C6F7A0B5ED8D, %26
  %sub69 = fsub double %25, %mul68
  %idxprom70 = sext i32 %i52.0 to i64
  %arrayidx71 = getelementptr inbounds [9 x double], ptr %Bm, i64 0, i64 %idxprom70
  store double %sub69, ptr %arrayidx71, align 8
  %inc73 = add nsw i32 %i52.0, 1
  br label %for.cond53

for.end74:                                        ; preds = %for.cond53
  %arraydecay75 = getelementptr inbounds [6 x double], ptr %Ap, i64 0, i64 0
  %arraydecay76 = getelementptr inbounds [9 x double], ptr %Bp, i64 0, i64 0
  %arraydecay77 = getelementptr inbounds [12 x double], ptr %Cp, i64 0, i64 0
  call void @primal_dgemm(ptr noundef %ap, ptr noundef %arraydecay75, ptr noundef %arraydecay76, ptr noundef %bp, ptr noundef %arraydecay77)
  %arraydecay78 = getelementptr inbounds [6 x double], ptr %Am, i64 0, i64 0
  %arraydecay79 = getelementptr inbounds [9 x double], ptr %Bm, i64 0, i64 0
  %arraydecay80 = getelementptr inbounds [12 x double], ptr %Cm, i64 0, i64 0
  call void @primal_dgemm(ptr noundef %am, ptr noundef %arraydecay78, ptr noundef %arraydecay79, ptr noundef %bm, ptr noundef %arraydecay80)
  br label %for.cond81

for.cond81:                                       ; preds = %for.inc103, %for.end74
  %c.0 = phi i32 [ 0, %for.end74 ], [ %inc104, %for.inc103 ]
  %cmp82 = icmp slt i32 %c.0, 3
  br i1 %cmp82, label %for.cond84, label %for.cond112

for.cond84:                                       ; preds = %for.cond81, %for.inc100
  %r.0 = phi i32 [ %inc101, %for.inc100 ], [ 0, %for.cond81 ]
  %cmp85 = icmp slt i32 %r.0, 4
  br i1 %cmp85, label %for.body86, label %for.inc103

for.body86:                                       ; preds = %for.cond84
  %mul87 = mul nsw i32 %c.0, 4
  %add88 = add nsw i32 %r.0, %mul87
  %cmp89 = icmp slt i32 %r.0, 2
  br i1 %cmp89, label %if.then, label %if.else

if.then:                                          ; preds = %for.body86
  %idxprom90 = sext i32 %add88 to i64
  %arrayidx91 = getelementptr inbounds [12 x double], ptr %dC, i64 0, i64 %idxprom90
  %27 = load double, ptr %arrayidx91, align 8
  %idxprom92 = sext i32 %add88 to i64
  %arrayidx93 = getelementptr inbounds [12 x double], ptr %Cp, i64 0, i64 %idxprom92
  %28 = load double, ptr %arrayidx93, align 8
  %idxprom94 = sext i32 %add88 to i64
  %arrayidx95 = getelementptr inbounds [12 x double], ptr %Cm, i64 0, i64 %idxprom94
  %29 = load double, ptr %arrayidx95, align 8
  %sub96 = fsub double %28, %29
  %mul97 = fmul double 2.000000e+00, 0x3EB0C6F7A0B5ED8D
  %div = fdiv double %sub96, %mul97
  call void @check_close(ptr noundef @.str, i32 noundef %add88, double noundef %27, double noundef %div)
  br label %for.inc100

if.else:                                          ; preds = %for.body86
  %idxprom98 = sext i32 %add88 to i64
  %arrayidx99 = getelementptr inbounds [12 x double], ptr %dC, i64 0, i64 %idxprom98
  %30 = load double, ptr %arrayidx99, align 8
  call void @check_close(ptr noundef @.str.1, i32 noundef %add88, double noundef %30, double noundef 7.770000e+02)
  br label %for.inc100

for.inc100:                                       ; preds = %if.then, %if.else
  %inc101 = add nsw i32 %r.0, 1
  br label %for.cond84

for.inc103:                                       ; preds = %for.cond84
  %inc104 = add nsw i32 %c.0, 1
  br label %for.cond81

for.cond112:                                      ; preds = %for.cond81, %for.body114
  %i111.0 = phi i32 [ %inc124, %for.body114 ], [ 0, %for.cond81 ]
  %cmp113 = icmp slt i32 %i111.0, 10
  br i1 %cmp113, label %for.body114, label %for.end125

for.body114:                                      ; preds = %for.cond112
  %idxprom115 = sext i32 %i111.0 to i64
  %arrayidx116 = getelementptr inbounds [10 x double], ptr @main.C_init.4, i64 0, i64 %idxprom115
  %31 = load double, ptr %arrayidx116, align 8
  %idxprom117 = sext i32 %i111.0 to i64
  %arrayidx118 = getelementptr inbounds [10 x double], ptr %C109, i64 0, i64 %idxprom117
  store double %31, ptr %arrayidx118, align 8
  %idxprom119 = sext i32 %i111.0 to i64
  %arrayidx120 = getelementptr inbounds [10 x double], ptr @main.dC_init.5, i64 0, i64 %idxprom119
  %32 = load double, ptr %arrayidx120, align 8
  %idxprom121 = sext i32 %i111.0 to i64
  %arrayidx122 = getelementptr inbounds [10 x double], ptr %dC110, i64 0, i64 %idxprom121
  store double %32, ptr %arrayidx122, align 8
  %inc124 = add nsw i32 %i111.0, 1
  br label %for.cond112

for.end125:                                       ; preds = %for.cond112
  %33 = load i32, ptr @enzyme_const, align 4
  %34 = load i32, ptr @enzyme_const, align 4
  %35 = load i32, ptr @enzyme_const, align 4
  %36 = load i32, ptr @enzyme_dup, align 4
  %37 = load i32, ptr @enzyme_dup, align 4
  %arraydecay126 = getelementptr inbounds [10 x double], ptr %C109, i64 0, i64 0
  %arraydecay127 = getelementptr inbounds [10 x double], ptr %dC110, i64 0, i64 0
  call void (ptr, ...) @__enzyme_fwddiff(ptr noundef @primal_cblas_rowmajor, i32 noundef %33, double noundef 1.300000e+00, i32 noundef %34, ptr noundef @main.A.2, i32 noundef %35, ptr noundef @main.B.3, i32 noundef %36, double noundef -1.500000e+00, double noundef 2.000000e+00, i32 noundef %37, ptr noundef %arraydecay126, ptr noundef %arraydecay127)
  br label %for.cond129

for.cond129:                                      ; preds = %for.inc157, %for.end125
  %r128.0 = phi i32 [ 0, %for.end125 ], [ %inc158, %for.inc157 ]
  %cmp130 = icmp slt i32 %r128.0, 2
  br i1 %cmp130, label %for.cond133, label %for.end159

for.cond133:                                      ; preds = %for.cond129, %for.inc154
  %c132.0 = phi i32 [ %inc155, %for.inc154 ], [ 0, %for.cond129 ]
  %cmp134 = icmp slt i32 %c132.0, 5
  br i1 %cmp134, label %for.body135, label %for.inc157

for.body135:                                      ; preds = %for.cond133
  %mul137 = mul nsw i32 %r128.0, 5
  %add138 = add nsw i32 %mul137, %c132.0
  %cmp139 = icmp slt i32 %c132.0, 3
  br i1 %cmp139, label %if.then140, label %if.else150

if.then140:                                       ; preds = %for.body135
  %idxprom141 = sext i32 %add138 to i64
  %arrayidx142 = getelementptr inbounds [10 x double], ptr %dC110, i64 0, i64 %idxprom141
  %38 = load double, ptr %arrayidx142, align 8
  %idxprom143 = sext i32 %add138 to i64
  %arrayidx144 = getelementptr inbounds [10 x double], ptr @main.dC_init.5, i64 0, i64 %idxprom143
  %39 = load double, ptr %arrayidx144, align 8
  %mul145 = fmul double -1.500000e+00, %39
  %idxprom146 = sext i32 %add138 to i64
  %arrayidx147 = getelementptr inbounds [10 x double], ptr @main.C_init.4, i64 0, i64 %idxprom146
  %40 = load double, ptr %arrayidx147, align 8
  %mul148 = fmul double 2.000000e+00, %40
  %add149 = fadd double %mul145, %mul148
  call void @check_close(ptr noundef @.str.6, i32 noundef %add138, double noundef %38, double noundef %add149)
  br label %for.inc154

if.else150:                                       ; preds = %for.body135
  %idxprom151 = sext i32 %add138 to i64
  %arrayidx152 = getelementptr inbounds [10 x double], ptr %dC110, i64 0, i64 %idxprom151
  %41 = load double, ptr %arrayidx152, align 8
  call void @check_close(ptr noundef @.str.7, i32 noundef %add138, double noundef %41, double noundef 7.770000e+02)
  br label %for.inc154

for.inc154:                                       ; preds = %if.then140, %if.else150
  %inc155 = add nsw i32 %c132.0, 1
  br label %for.cond133

for.inc157:                                       ; preds = %for.cond133
  %inc158 = add nsw i32 %r128.0, 1
  br label %for.cond129

for.end159:                                       ; preds = %for.cond129
  store double -9.000000e-01, ptr %alpha160, align 8
  store double 4.000000e-01, ptr %dalpha161, align 8
  store double 5.000000e-01, ptr %beta162, align 8
  store double -1.250000e+00, ptr %dbeta163, align 8
  br label %for.cond170

for.cond170:                                      ; preds = %for.body172, %for.end159
  %i169.0 = phi i32 [ 0, %for.end159 ], [ %inc198, %for.body172 ]
  %cmp171 = icmp slt i32 %i169.0, 12
  br i1 %cmp171, label %for.body172, label %for.end199

for.body172:                                      ; preds = %for.cond170
  %idxprom173 = sext i32 %i169.0 to i64
  %arrayidx174 = getelementptr inbounds [12 x double], ptr @main.C_init.12, i64 0, i64 %idxprom173
  %42 = load double, ptr %arrayidx174, align 8
  %idxprom175 = sext i32 %i169.0 to i64
  %arrayidx176 = getelementptr inbounds [12 x double], ptr %C164, i64 0, i64 %idxprom175
  store double %42, ptr %arrayidx176, align 8
  %idxprom177 = sext i32 %i169.0 to i64
  %arrayidx178 = getelementptr inbounds [12 x double], ptr @main.dC_init.13, i64 0, i64 %idxprom177
  %43 = load double, ptr %arrayidx178, align 8
  %idxprom179 = sext i32 %i169.0 to i64
  %arrayidx180 = getelementptr inbounds [12 x double], ptr %dC165, i64 0, i64 %idxprom179
  store double %43, ptr %arrayidx180, align 8
  %idxprom181 = sext i32 %i169.0 to i64
  %arrayidx182 = getelementptr inbounds [12 x double], ptr @main.C_init.12, i64 0, i64 %idxprom181
  %44 = load double, ptr %arrayidx182, align 8
  %idxprom183 = sext i32 %i169.0 to i64
  %arrayidx184 = getelementptr inbounds [12 x double], ptr @main.dC_init.13, i64 0, i64 %idxprom183
  %45 = load double, ptr %arrayidx184, align 8
  %mul185 = fmul double 0x3EB0C6F7A0B5ED8D, %45
  %add186 = fadd double %44, %mul185
  %idxprom187 = sext i32 %i169.0 to i64
  %arrayidx188 = getelementptr inbounds [12 x double], ptr %Cp166, i64 0, i64 %idxprom187
  store double %add186, ptr %arrayidx188, align 8
  %idxprom189 = sext i32 %i169.0 to i64
  %arrayidx190 = getelementptr inbounds [12 x double], ptr @main.C_init.12, i64 0, i64 %idxprom189
  %46 = load double, ptr %arrayidx190, align 8
  %idxprom191 = sext i32 %i169.0 to i64
  %arrayidx192 = getelementptr inbounds [12 x double], ptr @main.dC_init.13, i64 0, i64 %idxprom191
  %47 = load double, ptr %arrayidx192, align 8
  %mul193 = fmul double 0x3EB0C6F7A0B5ED8D, %47
  %sub194 = fsub double %46, %mul193
  %idxprom195 = sext i32 %i169.0 to i64
  %arrayidx196 = getelementptr inbounds [12 x double], ptr %Cm167, i64 0, i64 %idxprom195
  store double %sub194, ptr %arrayidx196, align 8
  %inc198 = add nsw i32 %i169.0, 1
  br label %for.cond170

for.end199:                                       ; preds = %for.cond170
  %48 = load i32, ptr @enzyme_dup, align 4
  %49 = load i32, ptr @enzyme_dup, align 4
  %50 = load i32, ptr @enzyme_dup, align 4
  %51 = load i32, ptr @enzyme_dup, align 4
  %52 = load i32, ptr @enzyme_dup, align 4
  %arraydecay200 = getelementptr inbounds [12 x double], ptr %C164, i64 0, i64 0
  %arraydecay201 = getelementptr inbounds [12 x double], ptr %dC165, i64 0, i64 0
  call void (ptr, ...) @__enzyme_fwddiff(ptr noundef @primal_dsymm, i32 noundef %48, ptr noundef %alpha160, ptr noundef %dalpha161, i32 noundef %49, ptr noundef @main.A.8, ptr noundef @main.dA.9, i32 noundef %50, ptr noundef @main.B.10, ptr noundef @main.dB.11, i32 noundef %51, ptr noundef %beta162, ptr noundef %dbeta163, i32 noundef %52, ptr noundef %arraydecay200, ptr noundef %arraydecay201)
  %53 = load double, ptr %alpha160, align 8
  %54 = load double, ptr %dalpha161, align 8
  %mul203 = fmul double 0x3EB0C6F7A0B5ED8D, %54
  %add204 = fadd double %53, %mul203
  store double %add204, ptr %ap202, align 8
  %55 = load double, ptr %alpha160, align 8
  %56 = load double, ptr %dalpha161, align 8
  %mul206 = fmul double 0x3EB0C6F7A0B5ED8D, %56
  %sub207 = fsub double %55, %mul206
  store double %sub207, ptr %am205, align 8
  %57 = load double, ptr %beta162, align 8
  %58 = load double, ptr %dbeta163, align 8
  %mul209 = fmul double 0x3EB0C6F7A0B5ED8D, %58
  %add210 = fadd double %57, %mul209
  store double %add210, ptr %bp208, align 8
  %59 = load double, ptr %beta162, align 8
  %60 = load double, ptr %dbeta163, align 8
  %mul212 = fmul double 0x3EB0C6F7A0B5ED8D, %60
  %sub213 = fsub double %59, %mul212
  store double %sub213, ptr %bm211, align 8
  br label %for.cond219

for.cond219:                                      ; preds = %for.body221, %for.end199
  %i218.0 = phi i32 [ 0, %for.end199 ], [ %inc239, %for.body221 ]
  %cmp220 = icmp slt i32 %i218.0, 6
  br i1 %cmp220, label %for.body221, label %for.cond242

for.body221:                                      ; preds = %for.cond219
  %idxprom222 = sext i32 %i218.0 to i64
  %arrayidx223 = getelementptr inbounds [6 x double], ptr @main.A.8, i64 0, i64 %idxprom222
  %61 = load double, ptr %arrayidx223, align 8
  %idxprom224 = sext i32 %i218.0 to i64
  %arrayidx225 = getelementptr inbounds [6 x double], ptr @main.dA.9, i64 0, i64 %idxprom224
  %62 = load double, ptr %arrayidx225, align 8
  %mul226 = fmul double 0x3EB0C6F7A0B5ED8D, %62
  %add227 = fadd double %61, %mul226
  %idxprom228 = sext i32 %i218.0 to i64
  %arrayidx229 = getelementptr inbounds [6 x double], ptr %Ap214, i64 0, i64 %idxprom228
  store double %add227, ptr %arrayidx229, align 8
  %idxprom230 = sext i32 %i218.0 to i64
  %arrayidx231 = getelementptr inbounds [6 x double], ptr @main.A.8, i64 0, i64 %idxprom230
  %63 = load double, ptr %arrayidx231, align 8
  %idxprom232 = sext i32 %i218.0 to i64
  %arrayidx233 = getelementptr inbounds [6 x double], ptr @main.dA.9, i64 0, i64 %idxprom232
  %64 = load double, ptr %arrayidx233, align 8
  %mul234 = fmul double 0x3EB0C6F7A0B5ED8D, %64
  %sub235 = fsub double %63, %mul234
  %idxprom236 = sext i32 %i218.0 to i64
  %arrayidx237 = getelementptr inbounds [6 x double], ptr %Am215, i64 0, i64 %idxprom236
  store double %sub235, ptr %arrayidx237, align 8
  %inc239 = add nsw i32 %i218.0, 1
  br label %for.cond219

for.cond242:                                      ; preds = %for.cond219, %for.body244
  %i241.0 = phi i32 [ %inc262, %for.body244 ], [ 0, %for.cond219 ]
  %cmp243 = icmp slt i32 %i241.0, 9
  br i1 %cmp243, label %for.body244, label %for.end263

for.body244:                                      ; preds = %for.cond242
  %idxprom245 = sext i32 %i241.0 to i64
  %arrayidx246 = getelementptr inbounds [9 x double], ptr @main.B.10, i64 0, i64 %idxprom245
  %65 = load double, ptr %arrayidx246, align 8
  %idxprom247 = sext i32 %i241.0 to i64
  %arrayidx248 = getelementptr inbounds [9 x double], ptr @main.dB.11, i64 0, i64 %idxprom247
  %66 = load double, ptr %arrayidx248, align 8
  %mul249 = fmul double 0x3EB0C6F7A0B5ED8D, %66
  %add250 = fadd double %65, %mul249
  %idxprom251 = sext i32 %i241.0 to i64
  %arrayidx252 = getelementptr inbounds [9 x double], ptr %Bp216, i64 0, i64 %idxprom251
  store double %add250, ptr %arrayidx252, align 8
  %idxprom253 = sext i32 %i241.0 to i64
  %arrayidx254 = getelementptr inbounds [9 x double], ptr @main.B.10, i64 0, i64 %idxprom253
  %67 = load double, ptr %arrayidx254, align 8
  %idxprom255 = sext i32 %i241.0 to i64
  %arrayidx256 = getelementptr inbounds [9 x double], ptr @main.dB.11, i64 0, i64 %idxprom255
  %68 = load double, ptr %arrayidx256, align 8
  %mul257 = fmul double 0x3EB0C6F7A0B5ED8D, %68
  %sub258 = fsub double %67, %mul257
  %idxprom259 = sext i32 %i241.0 to i64
  %arrayidx260 = getelementptr inbounds [9 x double], ptr %Bm217, i64 0, i64 %idxprom259
  store double %sub258, ptr %arrayidx260, align 8
  %inc262 = add nsw i32 %i241.0, 1
  br label %for.cond242

for.end263:                                       ; preds = %for.cond242
  %arraydecay264 = getelementptr inbounds [6 x double], ptr %Ap214, i64 0, i64 0
  %arraydecay265 = getelementptr inbounds [9 x double], ptr %Bp216, i64 0, i64 0
  %arraydecay266 = getelementptr inbounds [12 x double], ptr %Cp166, i64 0, i64 0
  call void @primal_dsymm(ptr noundef %ap202, ptr noundef %arraydecay264, ptr noundef %arraydecay265, ptr noundef %bp208, ptr noundef %arraydecay266)
  %arraydecay267 = getelementptr inbounds [6 x double], ptr %Am215, i64 0, i64 0
  %arraydecay268 = getelementptr inbounds [9 x double], ptr %Bm217, i64 0, i64 0
  %arraydecay269 = getelementptr inbounds [12 x double], ptr %Cm167, i64 0, i64 0
  call void @primal_dsymm(ptr noundef %am205, ptr noundef %arraydecay267, ptr noundef %arraydecay268, ptr noundef %bm211, ptr noundef %arraydecay269)
  br label %for.cond271

for.cond271:                                      ; preds = %for.inc299, %for.end263
  %c270.0 = phi i32 [ 0, %for.end263 ], [ %inc300, %for.inc299 ]
  %cmp272 = icmp slt i32 %c270.0, 3
  br i1 %cmp272, label %for.cond275, label %for.cond302

for.cond275:                                      ; preds = %for.cond271, %for.inc296
  %r274.0 = phi i32 [ %inc297, %for.inc296 ], [ 0, %for.cond271 ]
  %cmp276 = icmp slt i32 %r274.0, 4
  br i1 %cmp276, label %for.body277, label %for.inc299

for.body277:                                      ; preds = %for.cond275
  %mul279 = mul nsw i32 %c270.0, 4
  %add280 = add nsw i32 %r274.0, %mul279
  %cmp281 = icmp slt i32 %r274.0, 2
  br i1 %cmp281, label %if.then282, label %if.else292

if.then282:                                       ; preds = %for.body277
  %idxprom283 = sext i32 %add280 to i64
  %arrayidx284 = getelementptr inbounds [12 x double], ptr %dC165, i64 0, i64 %idxprom283
  %69 = load double, ptr %arrayidx284, align 8
  %idxprom285 = sext i32 %add280 to i64
  %arrayidx286 = getelementptr inbounds [12 x double], ptr %Cp166, i64 0, i64 %idxprom285
  %70 = load double, ptr %arrayidx286, align 8
  %idxprom287 = sext i32 %add280 to i64
  %arrayidx288 = getelementptr inbounds [12 x double], ptr %Cm167, i64 0, i64 %idxprom287
  %71 = load double, ptr %arrayidx288, align 8
  %sub289 = fsub double %70, %71
  %mul290 = fmul double 2.000000e+00, 0x3EB0C6F7A0B5ED8D
  %div291 = fdiv double %sub289, %mul290
  call void @check_close(ptr noundef @.str.14, i32 noundef %add280, double noundef %69, double noundef %div291)
  br label %for.inc296

if.else292:                                       ; preds = %for.body277
  %idxprom293 = sext i32 %add280 to i64
  %arrayidx294 = getelementptr inbounds [12 x double], ptr %dC165, i64 0, i64 %idxprom293
  %72 = load double, ptr %arrayidx294, align 8
  call void @check_close(ptr noundef @.str.15, i32 noundef %add280, double noundef %72, double noundef 7.770000e+02)
  br label %for.inc296

for.inc296:                                       ; preds = %if.then282, %if.else292
  %inc297 = add nsw i32 %r274.0, 1
  br label %for.cond275

for.inc299:                                       ; preds = %for.cond275
  %inc300 = add nsw i32 %c270.0, 1
  br label %for.cond271

for.cond302:                                      ; preds = %for.cond271, %for.inc517
  %mode.0 = phi i32 [ %inc518, %for.inc517 ], [ 0, %for.cond271 ]
  %cmp303 = icmp slt i32 %mode.0, 2
  br i1 %cmp303, label %for.body304, label %for.cond521

for.body304:                                      ; preds = %for.cond302
  %cmp305 = icmp eq i32 %mode.0, 0
  %cond = select i1 %cmp305, i32 85, i32 76
  %conv = trunc i32 %cond to i8
  store double 1.400000e+00, ptr %alpha306, align 8
  store double -6.000000e-01, ptr %dalpha307, align 8
  store double 3.000000e+00, ptr %beta308, align 8
  store double -5.000000e-01, ptr %dbeta309, align 8
  br label %for.cond316

for.cond316:                                      ; preds = %for.inc409, %for.body304
  %c315.0 = phi i32 [ 0, %for.body304 ], [ %inc410, %for.inc409 ]
  %cmp317 = icmp slt i32 %c315.0, 3
  br i1 %cmp317, label %for.cond321, label %for.end411

for.cond321:                                      ; preds = %for.cond316, %cond.end401
  %r320.0 = phi i32 [ %inc407, %cond.end401 ], [ 0, %for.cond316 ]
  %cmp322 = icmp slt i32 %r320.0, 5
  br i1 %cmp322, label %for.body324, label %for.inc409

for.body324:                                      ; preds = %for.cond321
  %mul326 = mul nsw i32 %c315.0, 5
  %add327 = add nsw i32 %r320.0, %mul326
  %cmp328 = icmp slt i32 %r320.0, 3
  br i1 %cmp328, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %for.body324
  %cmp330 = icmp eq i32 %mode.0, 0
  %cmp332 = icmp sle i32 %r320.0, %c315.0
  %conv333 = zext i1 %cmp332 to i32
  %cmp334 = icmp sge i32 %r320.0, %c315.0
  %conv335 = zext i1 %cmp334 to i32
  %cond336 = select i1 %cmp330, i32 %conv333, i32 %conv335
  %tobool = icmp ne i32 %cond336, 0
  br label %land.end

land.end:                                         ; preds = %land.rhs, %for.body324
  %73 = phi i1 [ false, %for.body324 ], [ %tobool, %land.rhs ]
  %land.ext = zext i1 %73 to i32
  %cmp337 = icmp sge i32 %r320.0, 3
  br i1 %cmp337, label %if.then339, label %if.else344

if.then339:                                       ; preds = %land.end
  %idxprom340 = sext i32 %add327 to i64
  %arrayidx341 = getelementptr inbounds [15 x double], ptr %C_init, i64 0, i64 %idxprom340
  store double 1.000000e+30, ptr %arrayidx341, align 8
  %idxprom342 = sext i32 %add327 to i64
  %arrayidx343 = getelementptr inbounds [15 x double], ptr %dC_init, i64 0, i64 %idxprom342
  store double 7.770000e+02, ptr %arrayidx343, align 8
  br label %if.end371

if.else344:                                       ; preds = %land.end
  %tobool345 = icmp ne i32 %land.ext, 0
  br i1 %tobool345, label %if.else351, label %if.then346

if.then346:                                       ; preds = %if.else344
  %idxprom347 = sext i32 %add327 to i64
  %arrayidx348 = getelementptr inbounds [15 x double], ptr %C_init, i64 0, i64 %idxprom347
  store double 1.000000e+30, ptr %arrayidx348, align 8
  %idxprom349 = sext i32 %add327 to i64
  %arrayidx350 = getelementptr inbounds [15 x double], ptr %dC_init, i64 0, i64 %idxprom349
  store double -5.550000e+02, ptr %arrayidx350, align 8
  br label %if.end371

if.else351:                                       ; preds = %if.else344
  %add352 = add nsw i32 %r320.0, 1
  %conv353 = sitofp i32 %add352 to double
  %mul354 = fmul double 1.500000e+00, %conv353
  %add355 = add nsw i32 %c315.0, 1
  %conv356 = sitofp i32 %add355 to double
  %mul357 = fmul double 0x3FE6666666666666, %conv356
  %sub358 = fsub double %mul354, %mul357
  %idxprom359 = sext i32 %add327 to i64
  %arrayidx360 = getelementptr inbounds [15 x double], ptr %C_init, i64 0, i64 %idxprom359
  store double %sub358, ptr %arrayidx360, align 8
  %add361 = add nsw i32 %r320.0, 1
  %conv362 = sitofp i32 %add361 to double
  %mul363 = fmul double -4.000000e-01, %conv362
  %add364 = add nsw i32 %c315.0, 1
  %conv365 = sitofp i32 %add364 to double
  %mul366 = fmul double 9.000000e-01, %conv365
  %add367 = fadd double %mul363, %mul366
  %idxprom368 = sext i32 %add327 to i64
  %arrayidx369 = getelementptr inbounds [15 x double], ptr %dC_init, i64 0, i64 %idxprom368
  store double %add367, ptr %arrayidx369, align 8
  br label %if.end371

if.end371:                                        ; preds = %if.then346, %if.else351, %if.then339
  %idxprom372 = sext i32 %add327 to i64
  %arrayidx373 = getelementptr inbounds [15 x double], ptr %C_init, i64 0, i64 %idxprom372
  %74 = load double, ptr %arrayidx373, align 8
  %idxprom374 = sext i32 %add327 to i64
  %arrayidx375 = getelementptr inbounds [15 x double], ptr %C310, i64 0, i64 %idxprom374
  store double %74, ptr %arrayidx375, align 8
  %idxprom376 = sext i32 %add327 to i64
  %arrayidx377 = getelementptr inbounds [15 x double], ptr %dC_init, i64 0, i64 %idxprom376
  %75 = load double, ptr %arrayidx377, align 8
  %idxprom378 = sext i32 %add327 to i64
  %arrayidx379 = getelementptr inbounds [15 x double], ptr %dC311, i64 0, i64 %idxprom378
  store double %75, ptr %arrayidx379, align 8
  %idxprom380 = sext i32 %add327 to i64
  %arrayidx381 = getelementptr inbounds [15 x double], ptr %C_init, i64 0, i64 %idxprom380
  %76 = load double, ptr %arrayidx381, align 8
  %tobool382 = icmp ne i32 %land.ext, 0
  br i1 %tobool382, label %cond.true383, label %cond.end388

cond.true383:                                     ; preds = %if.end371
  %idxprom384 = sext i32 %add327 to i64
  %arrayidx385 = getelementptr inbounds [15 x double], ptr %dC_init, i64 0, i64 %idxprom384
  %77 = load double, ptr %arrayidx385, align 8
  %mul386 = fmul double 0x3EB0C6F7A0B5ED8D, %77
  br label %cond.end388

cond.end388:                                      ; preds = %if.end371, %cond.true383
  %cond389 = phi double [ %mul386, %cond.true383 ], [ 0.000000e+00, %if.end371 ]
  %add390 = fadd double %76, %cond389
  %idxprom391 = sext i32 %add327 to i64
  %arrayidx392 = getelementptr inbounds [15 x double], ptr %Cp312, i64 0, i64 %idxprom391
  store double %add390, ptr %arrayidx392, align 8
  %idxprom393 = sext i32 %add327 to i64
  %arrayidx394 = getelementptr inbounds [15 x double], ptr %C_init, i64 0, i64 %idxprom393
  %78 = load double, ptr %arrayidx394, align 8
  %tobool395 = icmp ne i32 %land.ext, 0
  br i1 %tobool395, label %cond.true396, label %cond.end401

cond.true396:                                     ; preds = %cond.end388
  %idxprom397 = sext i32 %add327 to i64
  %arrayidx398 = getelementptr inbounds [15 x double], ptr %dC_init, i64 0, i64 %idxprom397
  %79 = load double, ptr %arrayidx398, align 8
  %mul399 = fmul double 0x3EB0C6F7A0B5ED8D, %79
  br label %cond.end401

cond.end401:                                      ; preds = %cond.end388, %cond.true396
  %cond402 = phi double [ %mul399, %cond.true396 ], [ 0.000000e+00, %cond.end388 ]
  %sub403 = fsub double %78, %cond402
  %idxprom404 = sext i32 %add327 to i64
  %arrayidx405 = getelementptr inbounds [15 x double], ptr %Cm313, i64 0, i64 %idxprom404
  store double %sub403, ptr %arrayidx405, align 8
  %inc407 = add nsw i32 %r320.0, 1
  br label %for.cond321

for.inc409:                                       ; preds = %for.cond321
  %inc410 = add nsw i32 %c315.0, 1
  br label %for.cond316

for.end411:                                       ; preds = %for.cond316
  %80 = load i32, ptr @enzyme_const, align 4
  %conv412 = sext i8 %conv to i32
  %81 = load i32, ptr @enzyme_dup, align 4
  %82 = load i32, ptr @enzyme_dup, align 4
  %83 = load i32, ptr @enzyme_dup, align 4
  %84 = load i32, ptr @enzyme_dup, align 4
  %arraydecay413 = getelementptr inbounds [15 x double], ptr %C310, i64 0, i64 0
  %arraydecay414 = getelementptr inbounds [15 x double], ptr %dC311, i64 0, i64 0
  call void (ptr, ...) @__enzyme_fwddiff(ptr noundef @primal_dsyrk, i32 noundef %80, i32 noundef %conv412, i32 noundef %81, ptr noundef %alpha306, ptr noundef %dalpha307, i32 noundef %82, ptr noundef @main.A.16, ptr noundef @main.dA.17, i32 noundef %83, ptr noundef %beta308, ptr noundef %dbeta309, i32 noundef %84, ptr noundef %arraydecay413, ptr noundef %arraydecay414)
  %85 = load double, ptr %alpha306, align 8
  %86 = load double, ptr %dalpha307, align 8
  %mul416 = fmul double 0x3EB0C6F7A0B5ED8D, %86
  %add417 = fadd double %85, %mul416
  store double %add417, ptr %ap415, align 8
  %87 = load double, ptr %alpha306, align 8
  %88 = load double, ptr %dalpha307, align 8
  %mul419 = fmul double 0x3EB0C6F7A0B5ED8D, %88
  %sub420 = fsub double %87, %mul419
  store double %sub420, ptr %am418, align 8
  %89 = load double, ptr %beta308, align 8
  %90 = load double, ptr %dbeta309, align 8
  %mul422 = fmul double 0x3EB0C6F7A0B5ED8D, %90
  %add423 = fadd double %89, %mul422
  store double %add423, ptr %bp421, align 8
  %91 = load double, ptr %beta308, align 8
  %92 = load double, ptr %dbeta309, align 8
  %mul425 = fmul double 0x3EB0C6F7A0B5ED8D, %92
  %sub426 = fsub double %91, %mul425
  store double %sub426, ptr %bm424, align 8
  br label %for.cond430

for.cond430:                                      ; preds = %for.body433, %for.end411
  %i429.0 = phi i32 [ 0, %for.end411 ], [ %inc451, %for.body433 ]
  %cmp431 = icmp slt i32 %i429.0, 8
  br i1 %cmp431, label %for.body433, label %for.end452

for.body433:                                      ; preds = %for.cond430
  %idxprom434 = sext i32 %i429.0 to i64
  %arrayidx435 = getelementptr inbounds [8 x double], ptr @main.A.16, i64 0, i64 %idxprom434
  %93 = load double, ptr %arrayidx435, align 8
  %idxprom436 = sext i32 %i429.0 to i64
  %arrayidx437 = getelementptr inbounds [8 x double], ptr @main.dA.17, i64 0, i64 %idxprom436
  %94 = load double, ptr %arrayidx437, align 8
  %mul438 = fmul double 0x3EB0C6F7A0B5ED8D, %94
  %add439 = fadd double %93, %mul438
  %idxprom440 = sext i32 %i429.0 to i64
  %arrayidx441 = getelementptr inbounds [8 x double], ptr %Ap427, i64 0, i64 %idxprom440
  store double %add439, ptr %arrayidx441, align 8
  %idxprom442 = sext i32 %i429.0 to i64
  %arrayidx443 = getelementptr inbounds [8 x double], ptr @main.A.16, i64 0, i64 %idxprom442
  %95 = load double, ptr %arrayidx443, align 8
  %idxprom444 = sext i32 %i429.0 to i64
  %arrayidx445 = getelementptr inbounds [8 x double], ptr @main.dA.17, i64 0, i64 %idxprom444
  %96 = load double, ptr %arrayidx445, align 8
  %mul446 = fmul double 0x3EB0C6F7A0B5ED8D, %96
  %sub447 = fsub double %95, %mul446
  %idxprom448 = sext i32 %i429.0 to i64
  %arrayidx449 = getelementptr inbounds [8 x double], ptr %Am428, i64 0, i64 %idxprom448
  store double %sub447, ptr %arrayidx449, align 8
  %inc451 = add nsw i32 %i429.0, 1
  br label %for.cond430

for.end452:                                       ; preds = %for.cond430
  %conv453 = sext i8 %conv to i32
  %arraydecay454 = getelementptr inbounds [8 x double], ptr %Ap427, i64 0, i64 0
  %arraydecay455 = getelementptr inbounds [15 x double], ptr %Cp312, i64 0, i64 0
  call void @primal_dsyrk(i32 noundef %conv453, ptr noundef %ap415, ptr noundef %arraydecay454, ptr noundef %bp421, ptr noundef %arraydecay455)
  %conv456 = sext i8 %conv to i32
  %arraydecay457 = getelementptr inbounds [8 x double], ptr %Am428, i64 0, i64 0
  %arraydecay458 = getelementptr inbounds [15 x double], ptr %Cm313, i64 0, i64 0
  call void @primal_dsyrk(i32 noundef %conv456, ptr noundef %am418, ptr noundef %arraydecay457, ptr noundef %bm424, ptr noundef %arraydecay458)
  br label %for.cond460

for.cond460:                                      ; preds = %for.inc514, %for.end452
  %c459.0 = phi i32 [ 0, %for.end452 ], [ %inc515, %for.inc514 ]
  %cmp461 = icmp slt i32 %c459.0, 3
  br i1 %cmp461, label %for.cond465, label %for.inc517

for.cond465:                                      ; preds = %for.cond460, %for.inc511
  %r464.0 = phi i32 [ %inc512, %for.inc511 ], [ 0, %for.cond460 ]
  %cmp466 = icmp slt i32 %r464.0, 5
  br i1 %cmp466, label %for.body468, label %for.inc514

for.body468:                                      ; preds = %for.cond465
  %mul470 = mul nsw i32 %c459.0, 5
  %add471 = add nsw i32 %r464.0, %mul470
  %cmp473 = icmp slt i32 %r464.0, 3
  br i1 %cmp473, label %land.rhs475, label %land.end487

land.rhs475:                                      ; preds = %for.body468
  %cmp476 = icmp eq i32 %mode.0, 0
  %cmp479 = icmp sle i32 %r464.0, %c459.0
  %conv480 = zext i1 %cmp479 to i32
  %cmp482 = icmp sge i32 %r464.0, %c459.0
  %conv483 = zext i1 %cmp482 to i32
  %cond485 = select i1 %cmp476, i32 %conv480, i32 %conv483
  %tobool486 = icmp ne i32 %cond485, 0
  br label %land.end487

land.end487:                                      ; preds = %land.rhs475, %for.body468
  %97 = phi i1 [ false, %for.body468 ], [ %tobool486, %land.rhs475 ]
  %land.ext488 = zext i1 %97 to i32
  %cmp489 = icmp sge i32 %r464.0, 3
  br i1 %cmp489, label %if.then491, label %if.else494

if.then491:                                       ; preds = %land.end487
  %idxprom492 = sext i32 %add471 to i64
  %arrayidx493 = getelementptr inbounds [15 x double], ptr %dC311, i64 0, i64 %idxprom492
  %98 = load double, ptr %arrayidx493, align 8
  call void @check_close(ptr noundef @.str.18, i32 noundef %add471, double noundef %98, double noundef 7.770000e+02)
  br label %for.inc511

if.else494:                                       ; preds = %land.end487
  %tobool495 = icmp ne i32 %land.ext488, 0
  br i1 %tobool495, label %if.else499, label %if.then496

if.then496:                                       ; preds = %if.else494
  %idxprom497 = sext i32 %add471 to i64
  %arrayidx498 = getelementptr inbounds [15 x double], ptr %dC311, i64 0, i64 %idxprom497
  %99 = load double, ptr %arrayidx498, align 8
  call void @check_close(ptr noundef @.str.19, i32 noundef %add471, double noundef %99, double noundef -5.550000e+02)
  br label %for.inc511

if.else499:                                       ; preds = %if.else494
  %idxprom500 = sext i32 %add471 to i64
  %arrayidx501 = getelementptr inbounds [15 x double], ptr %dC311, i64 0, i64 %idxprom500
  %100 = load double, ptr %arrayidx501, align 8
  %idxprom502 = sext i32 %add471 to i64
  %arrayidx503 = getelementptr inbounds [15 x double], ptr %Cp312, i64 0, i64 %idxprom502
  %101 = load double, ptr %arrayidx503, align 8
  %idxprom504 = sext i32 %add471 to i64
  %arrayidx505 = getelementptr inbounds [15 x double], ptr %Cm313, i64 0, i64 %idxprom504
  %102 = load double, ptr %arrayidx505, align 8
  %sub506 = fsub double %101, %102
  %mul507 = fmul double 2.000000e+00, 0x3EB0C6F7A0B5ED8D
  %div508 = fdiv double %sub506, %mul507
  call void @check_close(ptr noundef @.str.20, i32 noundef %add471, double noundef %100, double noundef %div508)
  br label %for.inc511

for.inc511:                                       ; preds = %if.then491, %if.else499, %if.then496
  %inc512 = add nsw i32 %r464.0, 1
  br label %for.cond465

for.inc514:                                       ; preds = %for.cond465
  %inc515 = add nsw i32 %c459.0, 1
  br label %for.cond460

for.inc517:                                       ; preds = %for.cond460
  %inc518 = add nsw i32 %mode.0, 1
  br label %for.cond302

for.cond521:                                      ; preds = %for.cond302, %for.inc715
  %mode520.0 = phi i32 [ %inc716, %for.inc715 ], [ 0, %for.cond302 ]
  %cmp522 = icmp slt i32 %mode520.0, 2
  br i1 %cmp522, label %for.body524, label %for.end717

for.body524:                                      ; preds = %for.cond521
  %cmp526 = icmp eq i32 %mode520.0, 0
  %cond528 = select i1 %cmp526, i32 121, i32 122
  br label %for.cond541

for.cond541:                                      ; preds = %for.inc642, %for.body524
  %r540.0 = phi i32 [ 0, %for.body524 ], [ %inc643, %for.inc642 ]
  %cmp542 = icmp slt i32 %r540.0, 3
  br i1 %cmp542, label %for.cond546, label %for.end644

for.cond546:                                      ; preds = %for.cond541, %cond.end634
  %c545.0 = phi i32 [ %inc640, %cond.end634 ], [ 0, %for.cond541 ]
  %cmp547 = icmp slt i32 %c545.0, 5
  br i1 %cmp547, label %for.body549, label %for.inc642

for.body549:                                      ; preds = %for.cond546
  %mul551 = mul nsw i32 %r540.0, 5
  %add552 = add nsw i32 %mul551, %c545.0
  %cmp554 = icmp slt i32 %c545.0, 3
  br i1 %cmp554, label %land.rhs556, label %land.end568

land.rhs556:                                      ; preds = %for.body549
  %cmp557 = icmp eq i32 %mode520.0, 0
  %cmp560 = icmp sle i32 %r540.0, %c545.0
  %conv561 = zext i1 %cmp560 to i32
  %cmp563 = icmp sge i32 %r540.0, %c545.0
  %conv564 = zext i1 %cmp563 to i32
  %cond566 = select i1 %cmp557, i32 %conv561, i32 %conv564
  %tobool567 = icmp ne i32 %cond566, 0
  br label %land.end568

land.end568:                                      ; preds = %land.rhs556, %for.body549
  %103 = phi i1 [ false, %for.body549 ], [ %tobool567, %land.rhs556 ]
  %land.ext569 = zext i1 %103 to i32
  %cmp570 = icmp sge i32 %c545.0, 3
  br i1 %cmp570, label %if.then572, label %if.else577

if.then572:                                       ; preds = %land.end568
  %idxprom573 = sext i32 %add552 to i64
  %arrayidx574 = getelementptr inbounds [15 x double], ptr %C_init533, i64 0, i64 %idxprom573
  store double 1.000000e+30, ptr %arrayidx574, align 8
  %idxprom575 = sext i32 %add552 to i64
  %arrayidx576 = getelementptr inbounds [15 x double], ptr %dC_init534, i64 0, i64 %idxprom575
  store double 7.770000e+02, ptr %arrayidx576, align 8
  br label %if.end604

if.else577:                                       ; preds = %land.end568
  %tobool578 = icmp ne i32 %land.ext569, 0
  br i1 %tobool578, label %if.else584, label %if.then579

if.then579:                                       ; preds = %if.else577
  %idxprom580 = sext i32 %add552 to i64
  %arrayidx581 = getelementptr inbounds [15 x double], ptr %C_init533, i64 0, i64 %idxprom580
  store double 1.000000e+30, ptr %arrayidx581, align 8
  %idxprom582 = sext i32 %add552 to i64
  %arrayidx583 = getelementptr inbounds [15 x double], ptr %dC_init534, i64 0, i64 %idxprom582
  store double -5.550000e+02, ptr %arrayidx583, align 8
  br label %if.end604

if.else584:                                       ; preds = %if.else577
  %add585 = add nsw i32 %r540.0, 1
  %conv586 = sitofp i32 %add585 to double
  %mul587 = fmul double 2.000000e+00, %conv586
  %add588 = add nsw i32 %c545.0, 1
  %conv589 = sitofp i32 %add588 to double
  %mul590 = fmul double 5.000000e-01, %conv589
  %sub591 = fsub double %mul587, %mul590
  %idxprom592 = sext i32 %add552 to i64
  %arrayidx593 = getelementptr inbounds [15 x double], ptr %C_init533, i64 0, i64 %idxprom592
  store double %sub591, ptr %arrayidx593, align 8
  %add594 = add nsw i32 %r540.0, 1
  %conv595 = sitofp i32 %add594 to double
  %mul596 = fmul double 6.000000e-01, %conv595
  %add597 = add nsw i32 %c545.0, 1
  %conv598 = sitofp i32 %add597 to double
  %mul599 = fmul double 1.100000e+00, %conv598
  %add600 = fadd double %mul596, %mul599
  %idxprom601 = sext i32 %add552 to i64
  %arrayidx602 = getelementptr inbounds [15 x double], ptr %dC_init534, i64 0, i64 %idxprom601
  store double %add600, ptr %arrayidx602, align 8
  br label %if.end604

if.end604:                                        ; preds = %if.then579, %if.else584, %if.then572
  %idxprom605 = sext i32 %add552 to i64
  %arrayidx606 = getelementptr inbounds [15 x double], ptr %C_init533, i64 0, i64 %idxprom605
  %104 = load double, ptr %arrayidx606, align 8
  %idxprom607 = sext i32 %add552 to i64
  %arrayidx608 = getelementptr inbounds [15 x double], ptr %C535, i64 0, i64 %idxprom607
  store double %104, ptr %arrayidx608, align 8
  %idxprom609 = sext i32 %add552 to i64
  %arrayidx610 = getelementptr inbounds [15 x double], ptr %dC_init534, i64 0, i64 %idxprom609
  %105 = load double, ptr %arrayidx610, align 8
  %idxprom611 = sext i32 %add552 to i64
  %arrayidx612 = getelementptr inbounds [15 x double], ptr %dC536, i64 0, i64 %idxprom611
  store double %105, ptr %arrayidx612, align 8
  %idxprom613 = sext i32 %add552 to i64
  %arrayidx614 = getelementptr inbounds [15 x double], ptr %C_init533, i64 0, i64 %idxprom613
  %106 = load double, ptr %arrayidx614, align 8
  %tobool615 = icmp ne i32 %land.ext569, 0
  br i1 %tobool615, label %cond.true616, label %cond.end621

cond.true616:                                     ; preds = %if.end604
  %idxprom617 = sext i32 %add552 to i64
  %arrayidx618 = getelementptr inbounds [15 x double], ptr %dC_init534, i64 0, i64 %idxprom617
  %107 = load double, ptr %arrayidx618, align 8
  %mul619 = fmul double 0x3EB0C6F7A0B5ED8D, %107
  br label %cond.end621

cond.end621:                                      ; preds = %if.end604, %cond.true616
  %cond622 = phi double [ %mul619, %cond.true616 ], [ 0.000000e+00, %if.end604 ]
  %add623 = fadd double %106, %cond622
  %idxprom624 = sext i32 %add552 to i64
  %arrayidx625 = getelementptr inbounds [15 x double], ptr %Cp537, i64 0, i64 %idxprom624
  store double %add623, ptr %arrayidx625, align 8
  %idxprom626 = sext i32 %add552 to i64
  %arrayidx627 = getelementptr inbounds [15 x double], ptr %C_init533, i64 0, i64 %idxprom626
  %108 = load double, ptr %arrayidx627, align 8
  %tobool628 = icmp ne i32 %land.ext569, 0
  br i1 %tobool628, label %cond.true629, label %cond.end634

cond.true629:                                     ; preds = %cond.end621
  %idxprom630 = sext i32 %add552 to i64
  %arrayidx631 = getelementptr inbounds [15 x double], ptr %dC_init534, i64 0, i64 %idxprom630
  %109 = load double, ptr %arrayidx631, align 8
  %mul632 = fmul double 0x3EB0C6F7A0B5ED8D, %109
  br label %cond.end634

cond.end634:                                      ; preds = %cond.end621, %cond.true629
  %cond635 = phi double [ %mul632, %cond.true629 ], [ 0.000000e+00, %cond.end621 ]
  %sub636 = fsub double %108, %cond635
  %idxprom637 = sext i32 %add552 to i64
  %arrayidx638 = getelementptr inbounds [15 x double], ptr %Cm538, i64 0, i64 %idxprom637
  store double %sub636, ptr %arrayidx638, align 8
  %inc640 = add nsw i32 %c545.0, 1
  br label %for.cond546

for.inc642:                                       ; preds = %for.cond546
  %inc643 = add nsw i32 %r540.0, 1
  br label %for.cond541

for.end644:                                       ; preds = %for.cond541
  %110 = load i32, ptr @enzyme_const, align 4
  %111 = load i32, ptr @enzyme_dup, align 4
  %112 = load i32, ptr @enzyme_const, align 4
  %113 = load i32, ptr @enzyme_dup, align 4
  %114 = load i32, ptr @enzyme_dup, align 4
  %arraydecay645 = getelementptr inbounds [15 x double], ptr %C535, i64 0, i64 0
  %arraydecay646 = getelementptr inbounds [15 x double], ptr %dC536, i64 0, i64 0
  call void (ptr, ...) @__enzyme_fwddiff(ptr noundef @primal_cblas_dsyrk_rowmajor, i32 noundef %110, i32 noundef %cond528, i32 noundef %111, double noundef 8.000000e-01, double noundef -3.000000e-01, i32 noundef %112, ptr noundef @main.A.21, i32 noundef %113, double noundef -2.000000e+00, double noundef 1.500000e+00, i32 noundef %114, ptr noundef %arraydecay645, ptr noundef %arraydecay646)
  %mul647 = fmul double 0x3EB0C6F7A0B5ED8D, -3.000000e-01
  %add648 = fadd double 8.000000e-01, %mul647
  %mul649 = fmul double 0x3EB0C6F7A0B5ED8D, 1.500000e+00
  %add650 = fadd double -2.000000e+00, %mul649
  %arraydecay651 = getelementptr inbounds [15 x double], ptr %Cp537, i64 0, i64 0
  call void @primal_cblas_dsyrk_rowmajor(i32 noundef %cond528, double noundef %add648, ptr noundef @main.A.21, double noundef %add650, ptr noundef %arraydecay651)
  %mul652 = fmul double 0x3EB0C6F7A0B5ED8D, -3.000000e-01
  %sub653 = fsub double 8.000000e-01, %mul652
  %mul654 = fmul double 0x3EB0C6F7A0B5ED8D, 1.500000e+00
  %sub655 = fsub double -2.000000e+00, %mul654
  %arraydecay656 = getelementptr inbounds [15 x double], ptr %Cm538, i64 0, i64 0
  call void @primal_cblas_dsyrk_rowmajor(i32 noundef %cond528, double noundef %sub653, ptr noundef @main.A.21, double noundef %sub655, ptr noundef %arraydecay656)
  br label %for.cond658

for.cond658:                                      ; preds = %for.inc712, %for.end644
  %r657.0 = phi i32 [ 0, %for.end644 ], [ %inc713, %for.inc712 ]
  %cmp659 = icmp slt i32 %r657.0, 3
  br i1 %cmp659, label %for.cond663, label %for.inc715

for.cond663:                                      ; preds = %for.cond658, %for.inc709
  %c662.0 = phi i32 [ %inc710, %for.inc709 ], [ 0, %for.cond658 ]
  %cmp664 = icmp slt i32 %c662.0, 5
  br i1 %cmp664, label %for.body666, label %for.inc712

for.body666:                                      ; preds = %for.cond663
  %mul668 = mul nsw i32 %r657.0, 5
  %add669 = add nsw i32 %mul668, %c662.0
  %cmp671 = icmp slt i32 %c662.0, 3
  br i1 %cmp671, label %land.rhs673, label %land.end685

land.rhs673:                                      ; preds = %for.body666
  %cmp674 = icmp eq i32 %mode520.0, 0
  %cmp677 = icmp sle i32 %r657.0, %c662.0
  %conv678 = zext i1 %cmp677 to i32
  %cmp680 = icmp sge i32 %r657.0, %c662.0
  %conv681 = zext i1 %cmp680 to i32
  %cond683 = select i1 %cmp674, i32 %conv678, i32 %conv681
  %tobool684 = icmp ne i32 %cond683, 0
  br label %land.end685

land.end685:                                      ; preds = %land.rhs673, %for.body666
  %115 = phi i1 [ false, %for.body666 ], [ %tobool684, %land.rhs673 ]
  %land.ext686 = zext i1 %115 to i32
  %cmp687 = icmp sge i32 %c662.0, 3
  br i1 %cmp687, label %if.then689, label %if.else692

if.then689:                                       ; preds = %land.end685
  %idxprom690 = sext i32 %add669 to i64
  %arrayidx691 = getelementptr inbounds [15 x double], ptr %dC536, i64 0, i64 %idxprom690
  %116 = load double, ptr %arrayidx691, align 8
  call void @check_close(ptr noundef @.str.22, i32 noundef %add669, double noundef %116, double noundef 7.770000e+02)
  br label %for.inc709

if.else692:                                       ; preds = %land.end685
  %tobool693 = icmp ne i32 %land.ext686, 0
  br i1 %tobool693, label %if.else697, label %if.then694

if.then694:                                       ; preds = %if.else692
  %idxprom695 = sext i32 %add669 to i64
  %arrayidx696 = getelementptr inbounds [15 x double], ptr %dC536, i64 0, i64 %idxprom695
  %117 = load double, ptr %arrayidx696, align 8
  call void @check_close(ptr noundef @.str.23, i32 noundef %add669, double noundef %117, double noundef -5.550000e+02)
  br label %for.inc709

if.else697:                                       ; preds = %if.else692
  %idxprom698 = sext i32 %add669 to i64
  %arrayidx699 = getelementptr inbounds [15 x double], ptr %dC536, i64 0, i64 %idxprom698
  %118 = load double, ptr %arrayidx699, align 8
  %idxprom700 = sext i32 %add669 to i64
  %arrayidx701 = getelementptr inbounds [15 x double], ptr %Cp537, i64 0, i64 %idxprom700
  %119 = load double, ptr %arrayidx701, align 8
  %idxprom702 = sext i32 %add669 to i64
  %arrayidx703 = getelementptr inbounds [15 x double], ptr %Cm538, i64 0, i64 %idxprom702
  %120 = load double, ptr %arrayidx703, align 8
  %sub704 = fsub double %119, %120
  %mul705 = fmul double 2.000000e+00, 0x3EB0C6F7A0B5ED8D
  %div706 = fdiv double %sub704, %mul705
  call void @check_close(ptr noundef @.str.24, i32 noundef %add669, double noundef %118, double noundef %div706)
  br label %for.inc709

for.inc709:                                       ; preds = %if.then689, %if.else697, %if.then694
  %inc710 = add nsw i32 %c662.0, 1
  br label %for.cond663

for.inc712:                                       ; preds = %for.cond663
  %inc713 = add nsw i32 %r657.0, 1
  br label %for.cond658

for.inc715:                                       ; preds = %for.cond658
  %inc716 = add nsw i32 %mode520.0, 1
  br label %for.cond521

for.end717:                                       ; preds = %for.cond521
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str.25)
  ret i32 0
}

declare void @__enzyme_fwddiff(ptr noundef, ...)

; Function Attrs: nounwind uwtable
define internal void @check_close(ptr noundef %tag, i32 noundef %idx, double noundef %got, double noundef %expected) {
entry:
  %sub = fsub double %got, %expected
  %call = call double @abs_val(double noundef %sub)
  %cmp = fcmp ogt double %call, 0x3E7AD7F29ABCAF48
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = call i32 (ptr, ...) @printf(ptr noundef @.str.26, ptr noundef %tag, i32 noundef %idx, double noundef %got, double noundef %expected)
  call void @abort()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare i32 @printf(ptr noundef, ...)

; Function Attrs: nounwind uwtable
define internal double @abs_val(double noundef %x) {
entry:
  %cmp = fcmp olt double %x, 0.000000e+00
  %fneg = fneg double %x
  %cond = select i1 %cmp, double %fneg, double %x
  ret double %cond
}

declare void @abort()


; CHECK-LABEL: define internal void @fwddiffeprimal_dgemm(
; CHECK: call void @dgemm_(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %"B'", ptr %ldb, ptr %beta, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @dgemm_(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %"A'", ptr %lda, ptr %B, ptr %ldb, ptr %{{.+}}, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @dgemm_(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %"alpha'", ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %{{.+}}, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @daxpy_(ptr %byref.axpy.n.i, ptr %"beta'", ptr %{{.+}}, ptr %byref.axpy.one.i, ptr %{{.+}}, ptr %byref.axpy.one.i)
; CHECK: call void @dgemm_(ptr noundef %ta, ptr noundef %tb, ptr noundef %m, ptr noundef %n, ptr noundef %k, ptr noundef %alpha, ptr noundef %A, ptr noundef %lda, ptr noundef %B, ptr noundef %ldb, ptr noundef %beta, ptr noundef %C, ptr noundef %ldc, i32 noundef 1, i32 noundef 1)

; CHECK-LABEL: define internal void @fwddiffeprimal_cblas_rowmajor(
; CHECK: call void @cblas_dlascl(i32 101, i8 71, i32 0, i32 0, double 1.000000e+00, double %beta, i32 2, i32 3, ptr %"C'", i32 5, ptr %{{.+}})
; CHECK: %[[ROW_INDEX:.+]] = phi i32 [ 0, %entry ], [ %[[ROW_NEXT:.+]], %for.body.i ]
; CHECK: %[[SRC_OFFSET:.+]] = mul i32 %[[ROW_INDEX]], 5
; CHECK: %[[SRC_PTR:.+]] = getelementptr double, ptr %C, i32 %[[SRC_OFFSET]]
; CHECK: %[[DST_OFFSET:.+]] = mul i32 %[[ROW_INDEX]], 5
; CHECK: %[[DST_PTR:.+]] = getelementptr double, ptr %"C'", i32 %[[DST_OFFSET]]
; CHECK: call void @cblas_daxpy(i32 3, double %"beta'", ptr %[[SRC_PTR]], i32 1, ptr %[[DST_PTR]], i32 1)
; CHECK: %[[ROW_DONE:.+]] = icmp eq i32 %[[ROW_NEXT]], 2
; CHECK: call void @cblas_dgemm(i32 noundef 101, i32 noundef 111, i32 noundef 111, i32 noundef 2, i32 noundef 3, i32 noundef 2, double noundef %alpha, ptr noundef %A, i32 noundef 2, ptr noundef %B, i32 noundef 3, double noundef %beta, ptr noundef %C, i32 noundef 5)

; CHECK-LABEL: define internal void @fwddiffeprimal_dsymm(
; CHECK: call void @dsymm_(ptr %side, ptr %uplo, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %"B'", ptr %ldb, ptr %beta, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @dsymm_(ptr %side, ptr %uplo, ptr %m, ptr %n, ptr %alpha, ptr %"A'", ptr %lda, ptr %B, ptr %ldb, ptr %{{.+}}, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @dsymm_(ptr %side, ptr %uplo, ptr %m, ptr %n, ptr %"alpha'", ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %{{.+}}, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @daxpy_(ptr %byref.axpy.n.i, ptr %"beta'", ptr %{{.+}}, ptr %byref.axpy.one.i, ptr %{{.+}}, ptr %byref.axpy.one.i)
; CHECK: call void @dsymm_(ptr noundef %side, ptr noundef %uplo, ptr noundef %m, ptr noundef %n, ptr noundef %alpha, ptr noundef %A, ptr noundef %lda, ptr noundef %B, ptr noundef %ldb, ptr noundef %beta, ptr noundef %C, ptr noundef %ldc, i32 noundef 1, i32 noundef 1)

; CHECK-LABEL: define internal void @fwddiffeprimal_dsyrk(
; CHECK: call void @dsyr2k_(ptr %uplo, ptr %trans, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %"A'", ptr %lda, ptr %beta, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @dsyrk_(ptr %uplo, ptr %trans, ptr %n, ptr %k, ptr %"alpha'", ptr %A, ptr %lda, ptr %{{.+}}, ptr %"C'", ptr %ldc, i32 1, i32 1)
; CHECK: call void @daxpy_(ptr %byref.axpy.n.i, ptr %"beta'", ptr %{{.+}}, ptr %byref.axpy.one.i, ptr %{{.+}}, ptr %byref.axpy.one.i)
; CHECK: call void @dsyrk_(ptr noundef %uplo, ptr noundef %trans, ptr noundef %n, ptr noundef %k, ptr noundef %alpha, ptr noundef %A, ptr noundef %lda, ptr noundef %beta, ptr noundef %C, ptr noundef %ldc, i32 noundef 1, i32 noundef 1)

; CHECK-LABEL: define internal void @fwddiffeprimal_cblas_dsyrk_rowmajor(
; CHECK: call void @cblas_dsyrk(i32 101, i32 %uplo, i32 111, i32 3, i32 2, double %"alpha'", ptr %A, i32 2, double %beta, ptr %"C'", i32 5)
; CHECK: call void @cblas_daxpy(i32 %{{.+}}, double %"beta'", ptr %{{.+}}, i32 1, ptr %{{.+}}, i32 1)
; CHECK: call void @cblas_dsyrk(i32 noundef 101, i32 noundef %uplo, i32 noundef 111, i32 noundef 3, i32 noundef 2, double noundef %alpha, ptr noundef %A, i32 noundef 2, double noundef %beta, ptr noundef %C, i32 noundef 5)
