; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg)" -S | FileCheck %s

define double @calc(i64 %n, double* nocapture readonly %x) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  ret double %add

for.body:                                         ; preds = %for.cond.cleanup4, %entry
  %output.021 = phi double [ 0.000000e+00, %entry ], [ %add, %for.cond.cleanup4 ]
  %i.020 = phi i64 [ 0, %entry ], [ %inc9, %for.cond.cleanup4 ]
  %0 = tail call i8* @llvm.stacksave()
  %vla = alloca double, i64 %n, align 16
  br label %for.body5

for.cond.cleanup4:                                ; preds = %for.body5
  %1 = load double, double* %vla, align 16
  %add = fadd double %output.021, %1
  tail call void @llvm.stackrestore(i8* %0)
  %inc9 = add nuw nsw i64 %i.020, 1
  %exitcond22.not = icmp eq i64 %inc9, 12
  br i1 %exitcond22.not, label %for.cond.cleanup, label %for.body

for.body5:                                        ; preds = %for.body5, %for.body
  %i1.019 = phi i64 [ 0, %for.body ], [ %inc, %for.body5 ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %i1.019
  %2 = load double, double* %arrayidx, align 8
  %mul = fmul double %2, 2.000000e+00
  %arrayidx6 = getelementptr inbounds double, double* %vla, i64 %i1.019
  store double %mul, double* %arrayidx6, align 8
  %inc = add nuw nsw i64 %i1.019, 1
  %exitcond.not = icmp eq i64 %inc, 3
  br i1 %exitcond.not, label %for.cond.cleanup4, label %for.body5
}

declare i8* @llvm.stacksave()

declare void @llvm.stackrestore(i8*)

define void @obj(double* %x, double* %grad) {
entry:
  call void (...) @__enzyme_autodiff(i8* bitcast (double (i64, double*)* @calc to i8*), i64 10, metadata !"enzyme_dup", double* %x, double* %grad)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffecalc(i64 %n, double* nocapture readonly %x, double* nocapture %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %for.body5

; CHECK: for.cond.cleanup4:                                ; preds = %for.body5
; CHECK-NEXT:   %exitcond22.not = icmp eq i64 %iv.next, 12
; CHECK-NEXT:   br i1 %exitcond22.not, label %remat_enter, label %for.body

; CHECK: for.body5:                                        ; preds = %for.body5, %for.body
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.body5 ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %exitcond.not = icmp eq i64 %iv.next2, 3
; CHECK-NEXT:   br i1 %exitcond.not, label %for.cond.cleanup4, label %for.body5

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertfor.body5
; CHECK-NEXT:   %0 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %1 = select {{(fast )?}}i1 %0, double 0.000000e+00, double %"add'de.0"
; CHECK-NEXT:   br i1 %0, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %2 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertfor.body5:                                  ; preds = %incinvertfor.body5, %remat_for.body_for.cond.cleanup4
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 2, %remat_for.body_for.cond.cleanup4 ], [ %8, %incinvertfor.body5 ]
; CHECK-NEXT:   %"arrayidx6'ipg_unwrap" = getelementptr inbounds double, double* %"vla'ipc_unwrap", i64 %"iv1'ac.0"
; CHECK-NEXT:   %3 = load double, double* %"arrayidx6'ipg_unwrap", align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   store double 0.000000e+00, double* %"arrayidx6'ipg_unwrap", align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %4 = fmul fast double %3, 2.000000e+00
; CHECK-NEXT:   %"arrayidx'ipg_unwrap" = getelementptr inbounds double, double* %"x'", i64 %"iv1'ac.0"
; CHECK-NEXT:   %5 = load double, double* %"arrayidx'ipg_unwrap", align 8, !alias.scope !5, !noalias !8
; CHECK-NEXT:   %6 = fadd fast double %5, %4
; CHECK-NEXT:   store double %6, double* %"arrayidx'ipg_unwrap", align 8, !alias.scope !5, !noalias !8
; CHECK-NEXT:   %7 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertfor.body, label %incinvertfor.body5

; CHECK: incinvertfor.body5:                               ; preds = %invertfor.body5
; CHECK-NEXT:   %8 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body5

; CHECK: remat_enter:                                      ; preds = %for.cond.cleanup4, %incinvertfor.body
; CHECK-NEXT:   %"add'de.0" = phi double [ %1, %incinvertfor.body ], [ %differeturn, %for.cond.cleanup4 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %2, %incinvertfor.body ], [ 11, %for.cond.cleanup4 ]
; CHECK-NEXT:   %mallocsize_unwrap = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %"malloccall'mi" = alloca i8, i64 %mallocsize_unwrap, align 16
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"malloccall'mi", i8 0, i64 %mallocsize_unwrap, i1 false)
; CHECK-NEXT:   br label %remat_for.body_for.body5

; CHECK: remat_for.body_for.body5:                         ; preds = %remat_for.body_for.body5, %remat_enter
; CHECK-NEXT:   %fiv = phi i64 [ %9, %remat_for.body_for.body5 ], [ 0, %remat_enter ]
; CHECK-NEXT:   %9 = add i64 %fiv, 1
; CHECK-NEXT:   %exitcond.not_unwrap = icmp eq i64 %9, 3
; CHECK-NEXT:   br i1 %exitcond.not_unwrap, label %remat_for.body_for.cond.cleanup4, label %remat_for.body_for.body5

; CHECK: remat_for.body_for.cond.cleanup4:                 ; preds = %remat_for.body_for.body5
; CHECK-NEXT:   %"vla'ipc_unwrap" = bitcast i8* %"malloccall'mi" to double*
; CHECK-NEXT:   %10 = load double, double* %"vla'ipc_unwrap", align 16
; CHECK-NEXT:   %11 = fadd fast double %10, %"add'de.0"
; CHECK-NEXT:   store double %11, double* %"vla'ipc_unwrap", align 16
; CHECK-NEXT:   br label %invertfor.body5
; CHECK-NEXT: }


