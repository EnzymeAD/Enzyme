; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instcombine -correlated-propagation -adce -instcombine -simplifycfg -early-cse -simplifycfg -instcombine -simplifycfg -gvn -jump-threading -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instcombine,correlated-propagation,adce,instcombine,%simplifycfg,early-cse,%simplifycfg,instcombine,%simplifycfg,gvn,jump-threading,instcombine,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local double @f(double* nocapture readonly %x, i64 %n) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %if.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %if.end ]
  %data.016 = phi double [ 0.000000e+00, %entry ], [ %add5, %if.end ]
  %cmp2 = fcmp fast ogt double %data.016, 1.000000e+01
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds double, double* %x, i64 %n
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %data.016
  br label %cleanup

if.end:                                           ; preds = %for.body
  %arrayidx4 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %1 = load double, double* %arrayidx4, align 8
  %add5 = fadd fast double %1, %data.016
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %cmp = icmp ult i64 %indvars.iv, %n
  br i1 %cmp, label %for.body, label %cleanup

cleanup:                                          ; preds = %if.end, %if.then
  %data.1 = phi double [ %add, %if.then ], [ %add5, %if.end ]
  ret double %data.1
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, i64 %n) #0 {
entry:
  %call = call double (...) @__enzyme_fwdsplit(i8* bitcast (double (double*, i64)* @f to i8*), metadata !"enzyme_nofree", double* %x, double* %xp, i64 %n, i8* null)
  ret double %call
}

declare dso_local double @__enzyme_fwdsplit(...)


attributes #0 = { noinline nounwind uwtable }


; CHECK: define internal double @fwddiffef(double* nocapture readonly %x, double* nocapture %"x'", i64 %n, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to i1**
; CHECK-NEXT:   %truetape = load i1*, i1** %0, align 8, !enzyme_mustcache !{{[0-9]+}}
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %if.end, %entry
; CHECK-DAG:   %iv = phi i64 [ %iv.next, %if.end ], [ 0, %entry ]
; CHECK-DAG:   %[[data016:.+]] = phi {{(fast )?}}double [ %[[i5:.+]], %if.end ], [ 0.000000e+00, %entry ]
; CHECK-NEXT:   %[[i1:.+]] = getelementptr inbounds i1, i1* %truetape, i64 %iv
; CHECK-NEXT:   %cmp2 = load i1, i1* %[[i1]], align 1, !invariant.group !{{[0-9]+}}
; CHECK-NEXT:   br i1 %cmp2, label %if.then, label %if.end

; CHECK: if.then:                                          ; preds = %for.body
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds double, double* %"x'", i64 %n
; CHECK-NEXT:   %[[i2:.+]] = load double, double* %"arrayidx'ipg", align 8
; CHECK-NEXT:   %[[i3:.+]] = fadd fast double %[[i2]], %[[data016]]
; CHECK-NEXT:   br label %cleanup

; CHECK: if.end:                                           ; preds = %for.body
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %"arrayidx4'ipg" = getelementptr inbounds double, double* %"x'", i64 %iv
; CHECK-NEXT:   %[[i4:.+]] = load double, double* %"arrayidx4'ipg", align 8
; CHECK-NEXT:   %[[i5]] = fadd fast double %[[i4]], %[[data016]]
; CHECK-NEXT:   %cmp = icmp ult i64 %iv, %n
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %cleanup

; CHECK: cleanup:                                          ; preds = %if.end, %if.then
; CHECK-NEXT:   %[[data1:.+]] = phi {{(fast )?}}double [ %[[i3]], %if.then ], [ %[[i5]], %if.end ]
; CHECK-NEXT:   ret double %[[data1]]
; CHECK-NEXT: }
