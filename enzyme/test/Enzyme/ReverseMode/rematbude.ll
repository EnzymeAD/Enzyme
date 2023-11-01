; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, i8* nocapture)

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, i8* nocapture)

; Function Attrs: nofree noinline nosync nounwind uwtable
define dso_local float @fasten_main(float %i6) {
bb:
  %i = alloca [256 x float], align 16
  %p = bitcast [256 x float]* %i to i8*
  br label %bb9

bb9:                                              ; preds = %bb22, %bb
  %i10 = phi i64 [ %i27, %bb9 ], [ 0, %bb ]
  %i11 = phi float [ %i26, %bb9 ], [ 0.000000e+00, %bb ]
  call void @llvm.lifetime.start.p0(i64 1024, i8* nonnull %p) 
  %i19 = getelementptr inbounds [256 x float], [256 x float]* %i, i64 0, i64 0
  store float %i6, float* %i19, align 4
  %p2 = bitcast [256 x float]* %i to float*
  %i23 = load float, float* %p2, align 16
  ; here we make two explicit loads to force the cache algorithm to make a specific choice  
  %i8 = load float, float* %p2, align 16
  %i24 = fadd fast float %i23, %i8
  %i25 = fmul fast float %i24, %i24
  %i26 = fadd fast float %i25, %i11
  %i27 = add nuw nsw i64 %i10, 1
  %i28 = icmp eq i64 %i27, 10
  br i1 %i28, label %bb29, label %bb9

bb29:                                             ; preds = %bb22
  ret float %i26
}

; Function Attrs: nounwind uwtable
define dso_local void @runOpenMP() {
bb:
  call void (...) @__enzyme_reverse(float (float)* @fasten_main, float 2.0, float 1.0, i8* null) 
  ret void
}

declare void @__enzyme_reverse(...)

; CHECK: define internal { float } @diffefasten_main(float %i6, float %differeturn, i8* %tapeArg) 
; CHECK-NEXT: bb:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %i = alloca [256 x float], i64 1, align 16
; CHECK-NEXT:   %"i'ai" = alloca [256 x float], i64 1, align 16
; CHECK-NEXT:   %0 = bitcast [256 x float]* %"i'ai" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(1024) dereferenceable_or_null(1024) %0, i8 0, i64 1024, i1 false)
; CHECK-NEXT:   br label %bb9

; CHECK: bb9:                                              ; preds = %bb9, %bb
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb9 ], [ 0, %bb ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %i28 = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %i28, label %remat_enter, label %bb9

; CHECK: invertbb:                                         ; preds = %remat_enter
; CHECK-NEXT:   %1 = insertvalue { float } {{(undef|poison)}}, float %11, 0
; CHECK-NEXT:   ret { float } %1

; CHECK: incinvertbb9:                                     ; preds = %remat_enter
; CHECK-NEXT:   %2 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: remat_enter:                                      ; preds = %bb9, %incinvertbb9
; CHECK-NEXT:   %"i6'de.0" = phi float [ %11, %incinvertbb9 ], [ 0.000000e+00, %bb9 ]
; CHECK-NEXT:   %"i26'de.0" = phi float [ %13, %incinvertbb9 ], [ %differeturn, %bb9 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %2, %incinvertbb9 ], [ 9, %bb9 ]
; CHECK-NEXT:   %i19_unwrap = getelementptr inbounds [256 x float], [256 x float]* %i, i64 0, i64 0
; CHECK-NEXT:   store float %i6, float* %i19_unwrap, align 4
; CHECK-NEXT:   %p2_unwrap = bitcast [256 x float]* %i to float*
; CHECK-NEXT:   %i23_unwrap = load float, float* %p2_unwrap, align 16
; CHECK-NEXT:   %i8_unwrap = load float, float* %p2_unwrap, align 16
; CHECK-NEXT:   %i24_unwrap = fadd fast float %i23_unwrap, %i8_unwrap
; CHECK-NEXT:   %3 = fmul fast float %"i26'de.0", %i24_unwrap
; CHECK-NEXT:   %4 = fmul fast float %"i26'de.0", %i24_unwrap
; CHECK-NEXT:   %5 = fadd fast float %3, %4
; CHECK-NEXT:   %"p2'ipc_unwrap" = bitcast [256 x float]* %"i'ai" to float*
; CHECK-NEXT:   %6 = load float, float* %"p2'ipc_unwrap", align 16
; CHECK-NEXT:   %7 = fadd fast float %6, %5
; CHECK-NEXT:   store float %7, float* %"p2'ipc_unwrap", align 16
; CHECK-NEXT:   %8 = load float, float* %"p2'ipc_unwrap", align 16
; CHECK-NEXT:   %9 = fadd fast float %8, %5
; CHECK-NEXT:   store float %9, float* %"p2'ipc_unwrap", align 16
; CHECK-NEXT:   %"i19'ipg_unwrap" = getelementptr inbounds [256 x float], [256 x float]* %"i'ai", i64 0, i64 0
; CHECK-NEXT:   %10 = load float, float* %"i19'ipg_unwrap", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"i19'ipg_unwrap", align 4
; CHECK-NEXT:   %11 = fadd fast float %"i6'de.0", %10
; CHECK-NEXT:   %12 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   %13 = select fast i1 %12, float 0.000000e+00, float %"i26'de.0"
; CHECK-NEXT:   br i1 %12, label %invertbb, label %incinvertbb9
; CHECK-NEXT: }
