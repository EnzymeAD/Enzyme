; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S -early-cse -simplifycfg | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg,early-cse,%simplifycfg)" -S | FileCheck %s

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
  %call = call fast double @__enzyme_autodiff(i8* bitcast (void (double*, double**, i64)* @f to i8*), double* %x, double* %xp, double** %y, double** %yp, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double**, double**, i64)


attributes #0 = { noinline nounwind uwtable }

; CHECK: define internal {{(dso_local )?}}void @diffef(double* %x, double* %"x'", double** %y, double** %"y'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add {{(nuw )?}}i64 %n, 1
; CHECK-NEXT:   br label %for.cond

; CHECK: for.cond:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %cmp = icmp ne i64 %iv, %0
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %invertfor.cond

; CHECK: for.body:
; CHECK-NEXT:   %[[x2:.+]] = load double, double* %x
; CHECK-NEXT:   %[[yload:.+]] = load double*, double** %y
; CHECK-NEXT:   %[[finaly:.+]] = load double, double* %[[yload]]
; CHECK-NEXT:   %add = fadd fast double %[[finaly]], %[[x2]]
; CHECK-NEXT:   store double %add, double* %[[yload]]
; CHECK-NEXT:   br label %for.cond

; CHECK: invertentry:
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond:
; CHECK-NEXT:   %[[ivp:.+]] = phi i64 [ %[[sub:.+]], %incinvertfor.cond ], [ %0, %for.cond ]
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq i64 %[[ivp]], 0
; CHECK-NEXT:   br i1 %[[cmp]], label %invertentry, label %incinvertfor.cond

; CHECK: incinvertfor.cond:
; CHECK-NEXT:   %[[sub]] = add nsw i64 %[[ivp]], -1
; CHECK-NEXT:   %[[ipload:.+]] = load double*, double** %"y'"
; CHECK-NEXT:   %[[x10:.+]] = load double, double* %[[ipload]]
; CHECK-NEXT:   store double %[[x10]], double* %[[ipload]]
; CHECK-NEXT:   %[[x11:.+]] = load double, double* %"x'"
; CHECK-NEXT:   %[[x12:.+]] = fadd fast double %[[x11]], %[[x10]]
; CHECK-NEXT:   store double %[[x12]], double* %"x'"
; CHECK-NEXT:   br label %invertfor.cond
; CHECK-NEXT: }

