; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -early-cse -simplifycfg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg,early-cse,%simplifycfg,instsimplify)" -S | FileCheck %s

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local double @get(double* nocapture %x, i64 %i, i64 %j) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds double, double* %x, i64 %i
  %0 = load double, double* %arrayidx, align 8
  ret double %0
}

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local double @f(double* nocapture %x, i64 %n) #0 {
entry:
  br label %for.cond3.preheader

for.cond3.preheader:                              ; preds = %entry, %for.cond.cleanup6
  %i = phi i64 [ %i_inc, %for.cond.cleanup6 ], [ 0, %entry ]
  %outersum = phi double [ %sum.1.lcssa, %for.cond.cleanup6 ], [ 0.000000e+00, %entry ]
  %i_inc = add nuw i64 %i, 1
; note this is now technically not exactly triangular
;  %cmp423 = icmp eq i64 %i, 0
;  br i1 %cmp423, label %for.cond.cleanup6, label %for.body7
  br label %for.body7

for.body7:                                        ; preds = %for.cond3.preheader, %for.body7
  %j = phi i64 [ %j_inc, %for.body7 ], [ 0, %for.cond3.preheader ]
  %innersum = phi double [ %add, %for.body7 ], [ %outersum, %for.cond3.preheader ]
  %call = tail call fast double @get(double* %x, i64 undef, i64 %j)
  %mul = fmul fast double %call, %call
  %add = fadd fast double %mul, %innersum
  %j_inc = add nuw i64 %j, 1
  %exitcond = icmp eq i64 %j, %i
  br i1 %exitcond, label %for.cond.cleanup6, label %for.body7

for.cond.cleanup6:                                ; preds = %for.body7
  %sum.1.lcssa = phi double [ %add, %for.body7 ]
  %cmp1 = icmp eq i64 %i, %n
  br i1 %cmp1, label %return, label %for.cond3.preheader

return:                                           ; preds = %for.cond.cleanup6
  %retval.0 = phi double [ %sum.1.lcssa, %for.cond.cleanup6 ]
  ret double %retval.0
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @dsumsquare(double* %x, double* %xp, i64 %n) local_unnamed_addr #1 {
entry:
  %call = tail call fast double @__enzyme_autodiff(i8* bitcast (double (double*, i64)* @f to i8*), double* %x, double* %xp, i64 %n)
  ret double %call
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, i64) local_unnamed_addr

attributes #0 = { noinline norecurse nounwind uwtable }
attributes #1 = { noinline nounwind uwtable }

; The inner trip count depends on the outer induction variable, so the cache
; is sized by the maximum inner trip count and allocated once.

; CHECK: define internal void @diffef(double* nocapture %x, double* nocapture %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[np1:.+]] = add nsw i64 %n, 1
; CHECK-NEXT:   %[[smax:.+]] = call i64 @llvm.smax.i64(i64 %n, i64 0)
; CHECK-NEXT:   %[[inner:.+]] = add nuw nsw i64 %[[smax]], 1
; CHECK-NEXT:   %[[total:.+]] = mul nuw nsw i64 %[[inner]], %[[np1]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[total]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %call_malloccache = bitcast i8* %malloccall to double*
; CHECK-NEXT:   br label %for.cond3.preheader

; CHECK: for.cond3.preheader:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup6 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %for.body7

; CHECK: for.body7:
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body7 ], [ 0, %for.cond3.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %[[augmented:.+]] = call fast double @augmented_get(double* %x, double* %"x'", i64 undef, i64 %iv1)
; CHECK-NEXT:   %[[off:.+]] = mul nuw nsw i64 %iv, %[[inner]]
; CHECK-NEXT:   %[[idx:.+]] = add nuw nsw i64 %iv1, %[[off]]
; CHECK-NEXT:   %[[gep:.+]] = getelementptr inbounds double, double* %call_malloccache, i64 %[[idx]]
; CHECK-NEXT:   store double %[[augmented]], double* %[[gep]], align 8, !invariant.group ![[ig:[0-9]+]]
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv1, %iv
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup6, label %for.body7

; CHECK: invertentry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond3.preheader:
; CHECK-NEXT:   %[[done1:.+]] = icmp eq i64 %[[iv:.+]], 0
; CHECK-NEXT:   br i1 %[[done1]], label %invertentry, label %incinvertfor.cond3.preheader

; CHECK: invertfor.body7:
; CHECK-NEXT:   %[[iv1:.+]] = phi i64 [ %[[iv]], %invertfor.cond.cleanup6 ], [ %[[subinner:.+]], %incinvertfor.body7 ]
; CHECK-NEXT:   %[[roff:.+]] = mul nuw nsw i64 %[[iv]], %[[inner]]
; CHECK-NEXT:   %[[ridx:.+]] = add nuw nsw i64 %[[iv1]], %[[roff]]
; CHECK-NEXT:   %[[rgep:.+]] = getelementptr inbounds double, double* %call_malloccache, i64 %[[ridx]]
; CHECK-NEXT:   %[[cached:.+]] = load double, double* %[[rgep]], align 8, !invariant.group ![[ig]]
; CHECK-NEXT:   %[[m0diffecall:.+]] = fmul fast double %differeturn, %[[cached]]
; CHECK-NEXT:   %[[innerdiffe:.+]] = fadd fast double %[[m0diffecall]], %[[m0diffecall]]
; CHECK-NEXT:   call void @diffeget(double* %x, double* %"x'", i64 undef, i64 %[[iv1]], double %[[innerdiffe]])
; CHECK-NEXT:   %[[done2:.+]] = icmp eq i64 %[[iv1]], 0
; CHECK-NEXT:   br i1 %[[done2]], label %invertfor.cond3.preheader, label %incinvertfor.body7

; CHECK: invertfor.cond.cleanup6:
; CHECK-NEXT:   %[[iv]] = phi i64 [ %{{.+}}, %incinvertfor.cond3.preheader ], [ %n, %for.cond.cleanup6 ]
; CHECK-NEXT:   br label %invertfor.body7
; CHECK-NEXT: }
