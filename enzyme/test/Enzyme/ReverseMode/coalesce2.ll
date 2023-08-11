; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -enzyme-coalese=1 -enzyme-postopt=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-coalese=1 -enzyme-postopt=1 -S | FileCheck %s; fi

define double @square(double* noalias nocapture %arg, double* noalias nocapture %arg1, i1 %cond) {
bb:
  br i1 %cond, label %bb2, label %bb3

bb2:                                              ; preds = %bb3
  %res = phi double [ 0.000000e+00, %bb ], [ %i10, %bb3 ]
  store double 0.000000e+00, double* %arg, align 8
  store double 0.000000e+00, double* %arg1, align 8
  ret double %res

bb3:                                              ; preds = %bb3, %bb
  %i = phi i64 [ 0, %bb ], [ %i11, %bb3 ]
  %i4 = phi double [ 0.000000e+00, %bb ], [ %i10, %bb3 ]
  %i5 = getelementptr inbounds double, double* %arg, i64 %i
  %i6 = load double, double* %i5, align 8
  %i7 = getelementptr inbounds double, double* %arg1, i64 %i
  %i8 = load double, double* %i7, align 8
  %i9 = fmul double %i6, %i8
  %i10 = fadd double %i4, %i9
  %i11 = add nuw nsw i64 %i, 1
  %i12 = icmp eq i64 %i11, 100
  br i1 %i12, label %bb2, label %bb3
}

define dso_local double @dsquare(double* %arg, double* %arg1, double* %arg2, double* %arg3, i1 %cond) {
bb:
  %i = tail call double @__enzyme_autodiff(i8* bitcast (double (double*, double*, i1)* @square to i8*), double* %arg, double* %arg1, double* %arg2, double* %arg3, i1 %cond)
  ret double %i
}

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*, i1)

; CHECK: define dso_local double @dsquare(double* nocapture %arg, double* nocapture %arg1, double* nocapture %arg2, double* nocapture %arg3, i1 %cond)
; CHECK: bb:
; CHECK:   br i1 %cond, label %[[tblk:.+]], label %[[blk:.+]]

; CHECK: [[tblk]]:                                     ; preds = %bb
; CHECK-NEXT:   store double 0.000000e+00, double* %arg, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %arg2, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %arg3, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %arg1, align 8
; CHECK-NEXT:   br label %diffesquare.exit

; CHECK: [[blk]]:                            ; preds = %bb
; CHECK-NEXT:   %0 = tail call {{(dereferenceable_or_null\(1600\) )?}}i8* @malloc(i64 1600){{( #[0-9]+)?}}, !noalias !{{[0-9]+}}, !enzyme_cache_alloc ![[ascope:.+]]
; CHECK-NEXT:   %1 = bitcast double* %arg2 to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{(noundef )?}}nonnull align 8 {{(dereferenceable\(800\) )?}}%0, i8* {{(noundef )?}}nonnull align 8 {{(dereferenceable\(800\) )?}}%1, i64 800, i1 false)
; CHECK-NEXT:   %2 = getelementptr inbounds i8, i8* %0, i64 800
; CHECK-NEXT:   %3 = bitcast double* %arg to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{(noundef )?}}nonnull align 8 {{(dereferenceable\(800\) )?}}%2, i8* {{(noundef )?}}nonnull align 8 {{(dereferenceable\(800\) )?}}%3, i64 800, i1 false)
; CHECK-NEXT:   %i8_malloccache.i = bitcast i8* %0 to double*
; CHECK-NEXT:   %i6_malloccache.i = bitcast i8* %2 to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %arg, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %arg2, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %arg3, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %arg1, align 8
; CHECK-NEXT:   br label %invertbb3.i

; CHECK: invertbb3.preheader.i:                            ; preds = %invertbb3.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %0){{( #[0-9]+)?}}, !noalias !{{[0-9]+}}, !enzyme_cache_free ![[ascope]]
; CHECK-NEXT:   br label %diffesquare.exit

; CHECK: invertbb3.i: 
; CHECK-NEXT:   %"iv'ac.0.i" = phi i64 [ %13, %invertbb3.i ], [ 99, %[[blk]] ]
; CHECK-NEXT:   %4 = getelementptr inbounds double, double* %i8_malloccache.i, i64 %"iv'ac.0.i"
; CHECK-NEXT:   %5 = load double, double* %4, align 8
; CHECK-NEXT:   %6 = getelementptr inbounds double, double* %i6_malloccache.i, i64 %"iv'ac.0.i"
; CHECK-NEXT:   %7 = load double, double* %6, align 8
; CHECK-NEXT:   %"i7'ipg_unwrap.i" = getelementptr inbounds double, double* %arg3, i64 %"iv'ac.0.i"
; CHECK-NEXT:   %8 = load double, double* %"i7'ipg_unwrap.i", align 8
; CHECK-NEXT:   %9 = fadd fast double %8, %7
; CHECK-NEXT:   store double %9, double* %"i7'ipg_unwrap.i", align 8
; CHECK-NEXT:   %"i5'ipg_unwrap.i" = getelementptr inbounds double, double* %arg1, i64 %"iv'ac.0.i"
; CHECK-NEXT:   %10 = load double, double* %"i5'ipg_unwrap.i", align 8
; CHECK-NEXT:   %11 = fadd fast double %10, %5
; CHECK-NEXT:   store double %11, double* %"i5'ipg_unwrap.i", align 8
; CHECK-NEXT:   %12 = icmp eq i64 %"iv'ac.0.i", 0
; CHECK-NEXT:   %13 = add nsw i64 %"iv'ac.0.i", -1
; CHECK-NEXT:   br i1 %12, label %invertbb3.preheader.i, label %invertbb3.i

; CHECK: diffesquare.exit:     
; CHECK-NEXT:   ret double undef
; CHECK-NEXT: }
