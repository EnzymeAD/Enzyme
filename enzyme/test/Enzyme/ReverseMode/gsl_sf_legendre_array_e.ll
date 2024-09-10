; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

declare dso_local i32 @gsl_sf_legendre_array_e(i32, i32, double, double, double*) local_unnamed_addr #1


; Function Attrs: noinline nounwind readnone uwtable
define dso_local void @tester(i32 %a0, i32 %a1, double %x, double %a3, double* %a4) {
entry:
  %c = call i32 @gsl_sf_legendre_array_e(i32 %a0, i32 %a1, double %x, double %a3, double* %a4)
  ret void
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(void (i32, i32, double, double, double*)* @tester, i32 0, i32 10, double %x, metadata !"enzyme_const", double %y, double* null, double* null)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.pow.f64(double, double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffetester(i32 %a0, i32 %a1, double %x, double %a3, double* %a4, double* %"a4'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %c = call i32 @gsl_sf_legendre_array_e(i32 %a0, i32 %a1, double %x, double %a3, double* %a4)
; CHECK-NEXT:   %[[as:.+]] = call i32 @gsl_sf_legendre_array_n(i32 %a1)
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %[[as]], 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %[[i0:.+]] = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i32 %[[as]], 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %[[i1:.+]] = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 -1, i8* %malloccall)
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 -1, i8* %malloccall2)
; CHECK-NEXT:   %[[i2:.+]] = call i32 @gsl_sf_legendre_deriv_array_e(i32 %a0, i32 %a1, double %x, double %a3, double* %[[i0]], double* %[[i1]])
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 -1, i8* %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   %[[i3:.+]] = icmp eq i32 %a1, 0
; CHECK-NEXT:   br i1 %[[i3]], label %invertentry_end, label %invertentry_loop

; CHECK: invertentry_loop: 
; CHECK-NEXT:   %[[i4:.+]] = phi i32 [ 0, %entry ], [ %[[i5:.+]], %invertentry_loop ]
; CHECK-NEXT:   %[[p5:.+]] = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %[[p12:.+]], %invertentry_loop ]
; CHECK-NEXT:   %[[i5]] = add nuw nsw i32 %[[i4]], 1
; CHECK-NEXT:   %[[i6:.+]] = getelementptr inbounds double, double* %[[i1]], i32 %[[i4]]
; CHECK-NEXT:   %[[i7:.+]] = getelementptr inbounds double, double* %"a4'", i32 %[[i4]]
; CHECK-NEXT:   %[[i8:.+]] = load double, double* %[[i7]], align 8
; CHECK-NEXT:   %[[i9:.+]] = load double, double* %[[i6]], align 8
; CHECK-NEXT:   %[[i10:.+]] = fmul fast double %[[i9]], %[[i8]]
; CHECK-NEXT:   %[[p12]] = fadd fast double %[[p5]], %[[i10]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[i7]], align 8
; CHECK-NEXT:   %[[c17:.+]] = icmp eq i32 %[[i5]], %a1
; CHECK-NEXT:   br i1 %[[c17]], label %invertentry_end, label %invertentry_loop

; CHECK: invertentry_end:
; CHECK-NEXT:   %[[res:.+]] = phi {{(fast )?}}double [ 0.000000e+00, %entry ], [ %[[p12]], %invertentry_loop ]
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 -1, i8* %malloccall2
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall2)
; CHECK-NEXT:   %[[i11:.+]] = insertvalue { double } {{(undef|poison)}}, double %[[res]], 0
; CHECK-NEXT:   ret { double } %[[i11]]
; CHECK-NEXT: }

