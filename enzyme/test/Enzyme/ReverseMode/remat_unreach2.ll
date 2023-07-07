; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg)" -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_const = dso_local local_unnamed_addr global i32 0, align 4

declare nonnull i8* @malloc(i64) 

declare void @free(i8*) 

define linkonce_odr dso_local void @noop(double* writeonly %tmp1) {
entry:
  store double 0.0, double* %tmp1
  ret void
}

define internal double @matvec(double* %tmp4, i1 %cmp) {
entry:
  %tcall = call noalias i8* @malloc(i64 128)
  %tmp = bitcast i8* %tcall to double*
  %omem = call noalias i8* @malloc(i64 128)
  %tmp19 = bitcast i8* %omem to double*
  br i1 %cmp, label %meta, label %if.end

if.end: ; preds = %if.then.i11
  br label %meta

meta: ; preds = %if.end, %if.then.i11
  %srcEvaluator.i.i.sroa.23.1 = phi double* [ %tmp19, %if.end ], [ null, %entry ]
  %tmp21 = load double, double* %tmp4, align 8
  store double 0.0, double* %tmp4
  br label %for.body.i.i.i

for.body.i.i.i:
  %i.07.i.i.i = phi i64 [ %inc.i.i.i, %for.body.i.i.i ], [ 0, %meta ]
  %arrayidx.i = getelementptr inbounds double, double* %srcEvaluator.i.i.sroa.23.1, i64 %i.07.i.i.i
  store double %tmp21, double* %arrayidx.i, align 8
  %inc.i.i.i = add nuw nsw i64 %i.07.i.i.i, 1
  %exitcond.i.i.i = icmp eq i64 %inc.i.i.i, 16
  br i1 %exitcond.i.i.i, label %lph, label %for.body.i.i.i

lph:
  br label %for.body.i

for.body.i:       ; preds = %for.body.i, %for.body5.i.i.i
  %res.i.0 = phi double [ %add.i23.i, %for.body.i ], [ 0.0, %lph ]
  %i.047 = phi i64 [ %inc.i, %for.body.i ], [ 0, %lph ]
  %tmp28 = load double, double* %tmp19, align 8
  %mul.i.i.i37.i = fmul double %tmp28, %tmp28
  %add.i23.i = fadd double %res.i.0, %mul.i.i.i37.i
  %inc.i = add nuw nsw i64 %i.047, 1
  %cmp.i.i.i45 = icmp ult i64 %inc.i, 4
  br i1 %cmp.i.i.i45, label %for.body.i, label %el

el:
  store double %add.i23.i, double* %tmp, align 8
  br label %ph

ie:                 ; preds = %for.body
  call void @noop(double* %tmp)
  br label %ph

ph:
  %ldd = load double, double* %tmp
  call void @free(i8* %omem)
  ret double %ldd
}


define dso_local void @_Z3dondd(double *%arg, double *%arg1) {
bb:
  call void (...) @_Z17__enzyme_autodiffPFddiEz(double (double*, i1)* nonnull @matvec, double *%arg, double *%arg1, i1 false)
  ret void
}

declare dso_local void @_Z17__enzyme_autodiffPFddiEz(...)

; CHECK: define internal void @diffematvec(double* %tmp4, double* %"tmp4'", i1 %cmp, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"tcall'mi" = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(128) dereferenceable_or_null(128) %"tcall'mi", i8 0, i64 128, i1 false)
; CHECK-NEXT:   %tcall = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %"tmp'ipc" = bitcast i8* %"tcall'mi" to double*
; CHECK-NEXT:   %tmp = bitcast i8* %tcall to double*
; CHECK-NEXT:   %"omem'mi" = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(128) dereferenceable_or_null(128) %"omem'mi", i8 0, i64 128, i1 false)
; CHECK-NEXT:   %omem = call noalias nonnull dereferenceable(128) dereferenceable_or_null(128) i8* @malloc(i64 128)
; CHECK-NEXT:   %"tmp19'ipc" = bitcast i8* %"omem'mi" to double*
; CHECK-NEXT:   %tmp19 = bitcast i8* %omem to double*
; CHECK-NEXT:   %spec.select = select i1 %cmp, double* null, double* %"tmp19'ipc"
; CHECK-NEXT:   %spec.select1 = select i1 %cmp, double* null, double* %tmp19
; CHECK-NEXT:   %tmp21 = load double, double* %tmp4, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %tmp4
; CHECK-NEXT:   br label %for.body.i.i.i

; CHECK: for.body.i.i.i:                                   ; preds = %for.body.i.i.i, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.body.i.i.i ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %arrayidx.i = getelementptr inbounds double, double* %spec.select1, i64 %iv
; CHECK-NEXT:   store double %tmp21, double* %arrayidx.i, align 8
; CHECK-NEXT:   %exitcond.i.i.i = icmp eq i64 %iv.next, 16
; CHECK-NEXT:   br i1 %exitcond.i.i.i, label %for.body.i, label %for.body.i.i.i

; CHECK: for.body.i:                                       ; preds = %for.body.i.i.i, %for.body.i
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body.i ], [ 0, %for.body.i.i.i ]
; CHECK-NEXT:   %res.i.0 = phi double [ %add.i23.i, %for.body.i ], [ 0.000000e+00, %for.body.i.i.i ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %tmp28 = load double, double* %tmp19, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   %mul.i.i.i37.i = fmul double %tmp28, %tmp28
; CHECK-NEXT:   %add.i23.i = fadd double %res.i.0, %mul.i.i.i37.i
; CHECK-NEXT:   %cmp.i.i.i45 = icmp ne i64 %iv.next2, 4
; CHECK-NEXT:   br i1 %cmp.i.i.i45, label %for.body.i, label %el

; CHECK: el:                                               ; preds = %for.body.i
; CHECK-NEXT:   store double %add.i23.i, double* %tmp
; CHECK-NEXT:   %0 = load double, double* %"tmp'ipc"
; CHECK-NEXT:   %1 = fadd fast double %0, %differeturn
; CHECK-NEXT:   store double %1, double* %"tmp'ipc"
; CHECK-NEXT:   %2 = load double, double* %"tmp'ipc", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"tmp'ipc", align 8
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: invertmeta:
; CHECK-NEXT:   store double 0.000000e+00, double* %"tmp4'"
; CHECK-NEXT:   %3 = load double, double* %"tmp4'", align 8
; CHECK-NEXT:   %4 = fadd fast double %3, %6
; CHECK-NEXT:   store double %4, double* %"tmp4'", align 8
; CHECK-NEXT:   call void @free(i8* nonnull %"omem'mi")
; CHECK-NEXT:   call void @free(i8* nonnull %omem)
; CHECK-NEXT:   call void @free(i8* nonnull %"tcall'mi")
; CHECK-NEXT:   call void @free(i8* nonnull %tcall)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body.i.i.i:                             ; preds = %invertfor.body.i, %incinvertfor.body.i.i.i
; CHECK-NEXT:   %"tmp21'de.0" = phi double [ %6, %incinvertfor.body.i.i.i ], [ 0.000000e+00, %invertfor.body.i ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %8, %incinvertfor.body.i.i.i ], [ 15, %invertfor.body.i ]
; CHECK-NEXT:   %"arrayidx.i'ipg_unwrap" = getelementptr inbounds double, double* %spec.select, i64 %"iv'ac.0"
; CHECK-NEXT:   %5 = load double, double* %"arrayidx.i'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx.i'ipg_unwrap", align 8
; CHECK-NEXT:   %6 = fadd fast double %"tmp21'de.0", %5
; CHECK-NEXT:   %7 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertmeta, label %incinvertfor.body.i.i.i

; CHECK: incinvertfor.body.i.i.i:                          ; preds = %invertfor.body.i.i.i
; CHECK-NEXT:   %8 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i.i.i

; CHECK: invertfor.body.i: 
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 3, %el ], [ %[[i14:.+]], %incinvertfor.body.i ]
; CHECK-NEXT:   %tmp28_unwrap = load double, double* %tmp19, align 8
; CHECK-NEXT:   %[[m0diffetmp28:.+]] = fmul fast double %2, %tmp28_unwrap
; CHECK-NEXT:   %[[m1diffetmp28:.+]] = fmul fast double %2, %tmp28_unwrap
; CHECK-NEXT:   %[[i9:.+]] = fadd fast double %[[m0diffetmp28]], %[[m1diffetmp28]]
; CHECK-NEXT:   %[[i10:.+]] = load double, double* %"tmp19'ipc", align 8
; CHECK-NEXT:   %[[i11:.+]] = fadd fast double %[[i10]], %[[i9]]
; CHECK-NEXT:   store double %[[i11]], double* %"tmp19'ipc", align 8
; CHECK-NEXT:   %[[i12:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %{{.*}} = select {{(fast )?}}i1 %[[i12]], double 0.000000e+00, double %2
; CHECK-NEXT:   br i1 %[[i12]], label %invertfor.body.i.i.i, label %incinvertfor.body.i

; CHECK: incinvertfor.body.i:                              ; preds = %invertfor.body.i
; CHECK-NEXT:   %[[i14]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body.i
; CHECK-NEXT: }
