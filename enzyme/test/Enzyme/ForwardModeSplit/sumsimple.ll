; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -inline -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S -early-cse -simplifycfg | FileCheck %s

; Function Attrs: noinline nounwind uwtable
define dso_local void @f(double* %x, double** %y, i64 %n) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ule i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds double, double* %x, i64 0
  %0 = load double, double* %arrayidx
  %1 = load double*, double** %y
  %2 = load double, double* %1
  %add = fadd fast double %2, %0
  store double %add, double* %1
  %inc = add i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, double** %y, double** %yp, i64 %n) #0 {
entry:
  %call = call double (...) @__enzyme_fwdsplit(i8* bitcast (void (double*, double**, i64)* @f to i8*), metadata !"enzyme_nofree", double* %x, double* %xp, double** %y, double** %yp, i64 %n, i8* null)
  ret double %call
}

declare dso_local double @__enzyme_fwdsplit(...)


attributes #0 = { noinline nounwind uwtable }


; CHECK: define internal void @fwddiffef(double* %x, double* %"x'", double** %y, double** %"y'", i64 %n, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to double***
; CHECK-NEXT:   %truetape = load double**, double*** %0
; CHECK-NEXT:   %1 = add {{(nuw )?}}i64 %n, 1
; CHECK-NEXT:   br label %for.cond

; CHECK: for.cond:                                         ; preds = %for.body, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv, %1
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end

; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %[[i2:.+]] = load double, double* %"x'"
; CHECK-NEXT:   %[[i3:.+]] = getelementptr inbounds double*, double** %truetape, i64 %iv
; CHECK-NEXT:   %[[il_phi:.+]] = load double*, double** %[[i3]], align 8, !invariant.group !{{[0-9]+}}
; CHECK-NEXT:   %[[i4:.+]] = load double, double* %[[il_phi]]
; CHECK-NEXT:   %[[i5:.+]] = fadd fast double %[[i4]], %[[i2]]
; CHECK-NEXT:   store double %[[i5]], double* %[[il_phi]]
; CHECK-NEXT:   br label %for.cond

; CHECK: for.end:                                          ; preds = %for.cond
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
