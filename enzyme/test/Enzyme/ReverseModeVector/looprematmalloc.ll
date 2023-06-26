; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -gvn -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,%simplifycfg,instsimplify,gvn,adce)" -enzyme-preopt=false -S | FileCheck %s

source_filename = "/app/example.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i8* @malloc(i64)
declare void @free(i8*)

; Function Attrs: nounwind readnone uwtable
define dso_local double @square(double %arg) #0 {
entry:
  br label %loop


loop:                                              ; preds = %bb6, %bb
  %i4 = phi i64 [ 0, %entry ], [ %i11, %loopExit ]
  %i5 = phi double [ 0.000000e+00, %entry ], [ %i10, %loopExit ]
  %i1 = call i8* @malloc(i64 240)
  %i = bitcast i8* %i1 to [30 x double]*
  call void @llvm.lifetime.start.p0i8(i64 240, i8* nonnull %i1) #4
  br label %setLoop

setLoop:                                             ; preds = %bb13, %bb3
  %i14 = phi i64 [ 0, %loop ], [ %i21, %setExit ]
  %i15 = and i64 %i14, 1
  %i16 = icmp eq i64 %i15, 0
  br i1 %i16, label %if.true, label %if.false

if.true:
  %i17 = trunc i64 %i14 to i32
  %i18 = call fast double @llvm.powi.f64(double %arg, i32 %i17)
  br label %setExit

if.false:
  br label %setExit

setExit:
  %i19 = phi double [ %i18, %if.true ], [ 0.000000e+00, %if.false ]
  %i20 = getelementptr inbounds [30 x double], [30 x double]* %i, i64 0, i64 %i14
  store double %i19, double* %i20, align 8, !tbaa !2
  %i21 = add nuw nsw i64 %i14, 1
  %i22 = icmp eq i64 %i21, 30
  br i1 %i22, label %loopExit, label %setLoop

loopExit:                                              ; preds = %bb13
  %i7 = getelementptr inbounds [30 x double], [30 x double]* %i, i64 0, i64 %i4
  %i8 = load double, double* %i7, align 8, !tbaa !2
  call void @free(i8* %i1)
  %i9 = fmul fast double %i8, %i8
  %i10 = fadd fast double %i9, %i5
  call void @llvm.lifetime.end.p0i8(i64 240, i8* nonnull %i1) #4
  %i11 = add nuw nsw i64 %i4, 1
  %i12 = icmp eq i64 %i11, 20
  br i1 %i12, label %exit, label %loop

exit:                                              ; preds = %bb6
  ret double %i10
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local [3 x double] @dsquare(double %arg) local_unnamed_addr #2 {
bb:
  %i = call [3 x double] (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @square to i8*), metadata !"enzyme_width", i64 3, double %arg) #4
  ret [3 x double] %i
}

declare dso_local [3 x double] @__enzyme_autodiff(i8*, ...)

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.powi.f64(double, i32) #3

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git d28af7c654d8db0b68c175db5ce212d74fb5e9bc)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}


; CHECK: define internal { [3 x double] } @diffe3square(double %arg, [3 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loopExit, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loopExit ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %i1 = call i8* @malloc(i64 240)
; CHECK-NEXT:   %i = bitcast i8* %i1 to [30 x double]*
; CHECK-NEXT:   br label %setLoop

; CHECK: setLoop:                                          ; preds = %setLoop, %loop
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %setLoop ], [ 0, %loop ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %i15 = and i64 %iv1, 1
; CHECK-NEXT:   %i16 = icmp eq i64 %i15, 0
; CHECK-NEXT:   %i17 = trunc i64 %iv1 to i32
; CHECK-NEXT:   %i18 = call fast double @llvm.powi.f64{{(.i32)?}}(double %arg, i32 %i17)
; CHECK-NEXT:   %i19 = select i1 %i16, double %i18, double 0.000000e+00
; CHECK-NEXT:   %i20 = getelementptr inbounds [30 x double], [30 x double]* %i, i64 0, i64 %iv1
; CHECK-NEXT:   store double %i19, double* %i20
; CHECK-NEXT:   %i22 = icmp eq i64 %iv.next2, 30
; CHECK-NEXT:   br i1 %i22, label %loopExit, label %setLoop

; CHECK: loopExit:                                         ; preds = %setLoop
; CHECK-NEXT:   call void @free(i8* %i1)
; CHECK-NEXT:   %i12 = icmp eq i64 %iv.next, 20
; CHECK-NEXT:   br i1 %i12, label %invertexit, label %loop

; CHECK: invertentry:                                      ; preds = %invertloop
; CHECK-NEXT:   %[[insert134:.+]] = insertvalue [3 x double] {{(undef|poison)}}, double %"arg'de.sroa.0.0", 0
; CHECK-NEXT:   %[[insert136:.+]] = insertvalue [3 x double] %[[insert134]], double %"arg'de.sroa.5.0", 1
; CHECK-NEXT:   %[[insert138:.+]] = insertvalue [3 x double] %[[insert136]], double %"arg'de.sroa.10.0", 2
; CHECK-NEXT:   %0 = insertvalue { [3 x double] } undef, [3 x double] %[[insert138]], 0
; CHECK-NEXT:   ret { [3 x double] } %0

; CHECK: invertloop:                                       ; preds = %invertsetLoop
; CHECK-NEXT:   call void @free(i8* nonnull %"i1'mi")
; CHECK-NEXT:   call void @free(i8* nonnull %"i1'mi1")
; CHECK-NEXT:   call void @free(i8* nonnull %"i1'mi2")
; CHECK-NEXT:   call void @free(i8* %remat_i1)
; CHECK-NEXT:   %1 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %[[i4:.+]] = select {{(fast )?}}i1 %1, double 0.000000e+00, double %"i10'de.sroa.0.0"
; CHECK-NEXT:   %[[i5:.+]] = select {{(fast )?}}i1 %1, double 0.000000e+00, double %"i10'de.sroa.7.0"
; CHECK-NEXT:   %[[i6:.+]] = select {{(fast )?}}i1 %1, double 0.000000e+00, double %"i10'de.sroa.14.0"
; CHECK-NEXT:   br i1 %1, label %invertentry, label %incinvertloop

; CHECK: incinvertloop:                                    ; preds = %invertloop
; CHECK-NEXT:   %[[i7:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertsetLoop:                                    ; preds = %invertsetExit, %invertif.true
; CHECK-NEXT:   %"arg'de.sroa.10.0" = phi double [ %[[i29:.+]], %invertif.true ], [ %"arg'de.sroa.10.1", %invertsetExit ]
; CHECK-NEXT:   %"arg'de.sroa.5.0" = phi double [ %[[i27:.+]], %invertif.true ], [ %"arg'de.sroa.5.1", %invertsetExit ]
; CHECK-NEXT:   %"arg'de.sroa.0.0" = phi double [ %[[i25:.+]], %invertif.true ], [ %"arg'de.sroa.0.1", %invertsetExit ]
; CHECK-NEXT:   %"i18'de.sroa.12.0" = phi double [ 0.000000e+00, %invertif.true ], [ %[[i39:.+]], %invertsetExit ]
; CHECK-NEXT:   %"i18'de.sroa.6.0" = phi double [ 0.000000e+00, %invertif.true ], [ %[[i37:.+]], %invertsetExit ]
; CHECK-NEXT:   %"i18'de.sroa.0.0" = phi double [ 0.000000e+00, %invertif.true ], [ %[[i35:.+]], %invertsetExit ]
; CHECK-NEXT:   %[[i8:.+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[i8]], label %invertloop, label %incinvertsetLoop

; CHECK: incinvertsetLoop:                                 ; preds = %invertsetLoop
; CHECK-NEXT:   %[[i9:.+]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertsetExit

; CHECK: invertif.true:                                    ; preds = %invertsetExit
; CHECK-NEXT:   %i17_unwrap6 = trunc i64 %"iv1'ac.0" to i32
; CHECK-NEXT:   %[[i22:.+]] = icmp eq i32 %i17_unwrap6, 0
; CHECK-NEXT:    %[[i13:.+]] = sitofp i32 %i17_unwrap6 to double
; CHECK-NEXT:   %[[i10:.+]] = sub i32 %i17_unwrap6, 1
; CHECK-NEXT:   %[[i11:.+]] = call fast double @llvm.powi.f64{{(.i32)?}}(double %arg, i32 %[[i10]])
; CHECK-NEXT:    %[[i12:.+]] = fmul fast double %[[i13]], %[[i11]] 
; CHECK-NEXT:   %[[i14:.+]] = fmul fast double %[[i35]], %[[i12]]
; CHECK-NEXT:   %[[i17:.+]] = fmul fast double %[[i37]], %[[i12]]
; CHECK-NEXT:   %[[i20:.+]] = fmul fast double %[[i39]], %[[i12]]
; CHECK-NEXT:   %[[m25:.+]] = fadd fast double %"arg'de.sroa.0.1", %[[i14]]
; CHECK-NEXT:   %[[i25]] = select {{(fast )?}}i1 %[[i22]], double %"arg'de.sroa.0.1", double %[[m25]]
; CHECK-NEXT:   %[[m27:.+]] = fadd fast double %"arg'de.sroa.5.1", %[[i17]]
; CHECK-NEXT:   %[[i27]] = select {{(fast )?}}i1 %[[i22]], double %"arg'de.sroa.5.1", double %[[m27]]
; CHECK-NEXT:   %[[m29:.+]] = fadd fast double %"arg'de.sroa.10.1", %[[i20]]
; CHECK-NEXT:   %[[i29]] = select {{(fast )?}}i1 %[[i22]], double %"arg'de.sroa.10.1", double %[[m29]]
; CHECK-NEXT:   br label %invertsetLoop

; CHECK: invertsetExit:                                    ; preds = %remat_loop_loopExit, %incinvertsetLoop
; CHECK-NEXT:   %"arg'de.sroa.10.1" = phi double [ %"arg'de.sroa.10.2", %remat_loop_loopExit ], [ %"arg'de.sroa.10.0", %incinvertsetLoop ]
; CHECK-NEXT:   %"arg'de.sroa.5.1" = phi double [ %"arg'de.sroa.5.2", %remat_loop_loopExit ], [ %"arg'de.sroa.5.0", %incinvertsetLoop ]
; CHECK-NEXT:   %"arg'de.sroa.0.1" = phi double [ %"arg'de.sroa.0.2", %remat_loop_loopExit ], [ %"arg'de.sroa.0.0", %incinvertsetLoop ]
; CHECK-NEXT:   %"i18'de.sroa.12.1" = phi double [ %"i18'de.sroa.12.2", %remat_loop_loopExit ], [ %"i18'de.sroa.12.0", %incinvertsetLoop ]
; CHECK-NEXT:   %"i18'de.sroa.6.1" = phi double [ %"i18'de.sroa.6.2", %remat_loop_loopExit ], [ %"i18'de.sroa.6.0", %incinvertsetLoop ]
; CHECK-NEXT:   %"i18'de.sroa.0.1" = phi double [ %"i18'de.sroa.0.2", %remat_loop_loopExit ], [ %"i18'de.sroa.0.0", %incinvertsetLoop ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 29, %remat_loop_loopExit ], [ %[[i9]], %incinvertsetLoop ]
; CHECK-NEXT:   %[[ipg10_unwrap:.+]] = getelementptr inbounds [30 x double], [30 x double]* %[[ipc8_unwrap26:.+]], i64 0, i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[ipg9_unwrap:.+]] = getelementptr inbounds [30 x double], [30 x double]* %[[ipc7_unwrap28:.+]], i64 0, i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[ipg_unwrap:.+]] = getelementptr inbounds [30 x double], [30 x double]* %[[ipc_unwrap30:.+]], i64 0, i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[i30:.+]] = load double, double* %[[ipg_unwrap]]
; CHECK-NEXT:   %[[i31:.+]] = load double, double* %[[ipg9_unwrap]]
; CHECK-NEXT:   %[[i32:.+]] = load double, double* %[[ipg10_unwrap]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[ipg_unwrap]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[ipg9_unwrap]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[ipg10_unwrap]]

; CHECK-NEXT:   %i15_unwrap16 = and i64 %"iv1'ac.0", 1
; CHECK-NEXT:   %i16_unwrap17 = icmp eq i64 %i15_unwrap16, 0
; CHECK-NEXT:   %[[q25:.+]] = fadd fast double %"i18'de.sroa.0.1", %[[i30]]
; CHECK-NEXT:   %[[i35]] = select {{(fast )?}}i1 %i16_unwrap17, double %[[q25]], double %"i18'de.sroa.0.1"
; CHECK-NEXT:   %[[q27:.+]] = fadd fast double %"i18'de.sroa.6.1", %[[i31]]
; CHECK-NEXT:   %[[i37]] = select {{(fast )?}}i1 %i16_unwrap17, double %[[q27]], double %"i18'de.sroa.6.1"
; CHECK-NEXT:   %[[q29:.+]] = fadd fast double %"i18'de.sroa.12.1", %[[i32]]
; CHECK-NEXT:   %[[i39]] = select {{(fast )?}}i1 %i16_unwrap17, double %[[q29]], double %"i18'de.sroa.12.1"
; CHECK-NEXT:   br i1 %i16_unwrap17, label %invertif.true, label %invertsetLoop

; CHECK: invertexit:                                       ; preds = %loopExit
; CHECK-NEXT:   %differeturn.fca.0.extract = extractvalue [3 x double] %differeturn, 0
; CHECK-NEXT:   %differeturn.fca.1.extract = extractvalue [3 x double] %differeturn, 1
; CHECK-NEXT:   %differeturn.fca.2.extract = extractvalue [3 x double] %differeturn, 2
; CHECK-NEXT:   br label %remat_enter

; CHECK: remat_enter:                                      ; preds = %invertexit, %incinvertloop
; CHECK-NEXT:   %"i10'de.sroa.14.0" = phi double [ %differeturn.fca.2.extract, %invertexit ], [ %[[i6]], %incinvertloop ]
; CHECK-NEXT:   %"i10'de.sroa.7.0" = phi double [ %differeturn.fca.1.extract, %invertexit ], [ %[[i5]], %incinvertloop ]
; CHECK-NEXT:   %"i10'de.sroa.0.0" = phi double [ %differeturn.fca.0.extract, %invertexit ], [ %[[i4]], %incinvertloop ]
; CHECK-NEXT:   %"arg'de.sroa.10.2" = phi double [ 0.000000e+00, %invertexit ], [ %"arg'de.sroa.10.0", %incinvertloop ]
; CHECK-NEXT:   %"arg'de.sroa.5.2" = phi double [ 0.000000e+00, %invertexit ], [ %"arg'de.sroa.5.0", %incinvertloop ]
; CHECK-NEXT:   %"arg'de.sroa.0.2" = phi double [ 0.000000e+00, %invertexit ], [ %"arg'de.sroa.0.0", %incinvertloop ]
; CHECK-NEXT:   %"i18'de.sroa.12.2" = phi double [ 0.000000e+00, %invertexit ], [ %"i18'de.sroa.12.0", %incinvertloop ]
; CHECK-NEXT:   %"i18'de.sroa.6.2" = phi double [ 0.000000e+00, %invertexit ], [ %"i18'de.sroa.6.0", %incinvertloop ]
; CHECK-NEXT:   %"i18'de.sroa.0.2" = phi double [ 0.000000e+00, %invertexit ], [ %"i18'de.sroa.0.0", %incinvertloop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 19, %invertexit ], [ %[[i7]], %incinvertloop ]
; CHECK-NEXT:   %remat_i1 = call i8* @malloc(i64 240)
; CHECK-NEXT:   %"i1'mi" = call noalias nonnull i8* @malloc(i64 240)
; CHECK-NEXT:   %"i1'mi1" = call noalias nonnull i8* @malloc(i64 240)
; CHECK-NEXT:   %"i1'mi2" = call noalias nonnull i8* @malloc(i64 240)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(240) dereferenceable_or_null(240) %"i1'mi", i8 0, i64 240, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(240) dereferenceable_or_null(240) %"i1'mi1", i8 0, i64 240, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(240) dereferenceable_or_null(240) %"i1'mi2", i8 0, i64 240, i1 false)
; CHECK-NEXT:   br label %remat_loop_setLoop

; CHECK: remat_loop_setLoop:                               ; preds = %remat_loop_setLoop, %remat_enter
; CHECK-NEXT:   %fiv = phi i64 [ %[[i40:.+]], %remat_loop_setLoop ], [ 0, %remat_enter ]
; CHECK-NEXT:   %[[i40]] = add i64 %fiv, 1
; CHECK-DAG:   %i15_unwrap3 = and i64 %fiv, 1
; CHECK-DAG:   %i16_unwrap4 = icmp eq i64 %i15_unwrap3, 0
; CHECK-DAG:   %i17_unwrap = trunc i64 %fiv to i32
; CHECK-DAG:   %[[i41:.+]] = call fast double @llvm.powi.f64{{(.i32)?}}(double %arg, i32 %i17_unwrap)
; CHECK-DAG:   %[[i42:.+]] = select i1 %i16_unwrap4, double %[[i41]], double 0.000000e+00
; CHECK-DAG:   %i_unwrap = bitcast i8* %remat_i1 to [30 x double]*
; CHECK-DAG:   %i20_unwrap = getelementptr inbounds [30 x double], [30 x double]* %i_unwrap, i64 0, i64 %fiv
; CHECK-NEXT:   store double %[[i42:.+]], double* %i20_unwrap
; CHECK-NEXT:   %i22_unwrap = icmp eq i64 %[[i40]], 30
; CHECK-NEXT:   br i1 %i22_unwrap, label %remat_loop_loopExit, label %remat_loop_setLoop

; CHECK: remat_loop_loopExit:                              ; preds = %remat_loop_setLoop
; CHECK-NEXT:   %i7_unwrap = getelementptr inbounds [30 x double], [30 x double]* %i_unwrap, i64 0, i64 %"iv'ac.0"
; CHECK-NEXT:   %i8_unwrap = load double, double* %i7_unwrap
; CHECK-NEXT:   %[[m0diffei8:.+]] = fmul fast double %"i10'de.sroa.0.0", %i8_unwrap
; CHECK-NEXT:   %[[m0diffei819:.+]] = fmul fast double %"i10'de.sroa.7.0", %i8_unwrap
; CHECK-NEXT:   %[[m0diffei820:.+]] = fmul fast double %"i10'de.sroa.14.0", %i8_unwrap
; CHECK-NEXT:   %[[i43:.+]] = fadd fast double %[[m0diffei8]], %[[m0diffei8]]
; CHECK-NEXT:   %[[i44:.+]] = fadd fast double %[[m0diffei819]], %[[m0diffei819]]
; CHECK-NEXT:   %[[i45:.+]] = fadd fast double %[[m0diffei820]], %[[m0diffei820]]
; CHECK-NEXT:   %[[ipc8_unwrap26]] = bitcast i8* %"i1'mi2" to [30 x double]*
; CHECK-NEXT:   %[[ipg24_unwrap:.+]] = getelementptr inbounds [30 x double], [30 x double]* %[[ipc8_unwrap26]], i64 0, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[ipc7_unwrap28]] = bitcast i8* %"i1'mi1" to [30 x double]*
; CHECK-NEXT:   %[[ipg23_unwrap:.+]] = getelementptr inbounds [30 x double], [30 x double]* %[[ipc7_unwrap28]], i64 0, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[ipc_unwrap30]] = bitcast i8* %"i1'mi" to [30 x double]*
; CHECK-NEXT:   %[[i7ipg_unwrap:.+]] = getelementptr inbounds [30 x double], [30 x double]* %[[ipc_unwrap30]], i64 0, i64 %"iv'ac.0"
; CHECK-NEXT:   %[[i46:.+]] = load double, double* %[[i7ipg_unwrap]]
; CHECK-NEXT:   %[[i49:.+]] = fadd fast double %[[i46]], %[[i43]]
; CHECK-NEXT:   store double %[[i49]], double* %[[i7ipg_unwrap]]
; CHECK-NEXT:   %[[i47:.+]] = load double, double* %[[ipg23_unwrap]]
; CHECK-NEXT:   %[[i50:.+]] = fadd fast double %[[i47]], %[[i44]]
; CHECK-NEXT:   store double %[[i50]], double* %[[ipg23_unwrap]]
; CHECK-NEXT:   %[[i48:.+]] = load double, double* %[[ipg24_unwrap]]
; CHECK-NEXT:   %[[i51:.+]] = fadd fast double %[[i48]], %[[i45]]
; CHECK-NEXT:   store double %[[i51]], double* %[[ipg24_unwrap]]
; CHECK-NEXT:   br label %invertsetExit
; CHECK-NEXT: }
