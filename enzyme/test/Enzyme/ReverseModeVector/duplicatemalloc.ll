; RUN: if [ %llvmver -lt 14 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instsimplify -early-cse -S | FileCheck %s -check-prefixes SHARED; fi
; RUN: if [ %llvmver -lt 14 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,%simplifycfg,instsimplify,early-cse)" -enzyme-preopt=false -S | FileCheck %s -check-prefixes SHARED; fi

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @f(double* nocapture readonly %x) local_unnamed_addr #0 {
entry:
  %0 = load double, double* %x, align 8, !tbaa !2
  ret double %0
}

; Function Attrs: nounwind uwtable
define dso_local double @malloced(double %x, i64 %n) #1 {
entry:
  %mul = shl i64 %n, 3
  %call = tail call i8* @malloc(i64 %mul)
  %0 = bitcast i8* %call to double*
  store double %x, double* %0, align 8, !tbaa !2
  %call1 = tail call fast double @f(double* %0)
  %call2 = tail call i32 (double*, ...) bitcast (i32 (...)* @free to i32 (double*, ...)*)(double* %0) #4
  %mul3 = fmul fast double %call1, %call1
  ret double %mul3
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

declare dso_local i32 @free(...) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define dso_local void @derivative(double %x, i64 %n) local_unnamed_addr #1 {
entry:
  %0 = tail call [3 x double] (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @malloced, metadata !"enzyme_width", i64 3, double %x, i64 %n)
  ret void
}

; Function Attrs: nounwind
declare [3 x double] @__enzyme_autodiff(double (double, i64)*, ...) #4

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"double", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}


; SHARED: define internal { [3 x double] } @diffe3malloced(double %x, i64 %n, [3 x double] %differeturn)
; SHARED-NEXT: entry:
; SHARED-NEXT:   %mul = shl i64 %n, 3
; SHARED-NEXT:   %[[alloc1:.+]] = tail call noalias nonnull i8* @malloc(i64 %mul)
; SHARED-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %[[alloc1]], i8 0, i64 %mul, i1 false)
; SHARED-NEXT:   %[[alloc2:.+]] = tail call noalias nonnull i8* @malloc(i64 %mul)
; SHARED-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %[[alloc2]], i8 0, i64 %mul, i1 false)
; SHARED-NEXT:   %[[alloc3:.+]] = tail call noalias nonnull i8* @malloc(i64 %mul)
; SHARED-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %[[alloc3]], i8 0, i64 %mul, i1 false)
; SHARED-NEXT:   %call = tail call i8* @malloc(i64 %mul)
; SHARED-NEXT:   %[[ipc:.+]] = bitcast i8* %[[alloc1]] to double*
; SHARED-NEXT:   %0 = insertvalue [3 x double*] {{(undef|poison)}}, double* %[[ipc]], 0
; SHARED-NEXT:   %[[ipc5:.+]] = bitcast i8* %[[alloc2]] to double*
; SHARED-NEXT:   %1 = insertvalue [3 x double*] %0, double* %[[ipc5]], 1
; SHARED-NEXT:   %[[ipc6:.+]] = bitcast i8* %[[alloc3]] to double*
; SHARED-NEXT:   %2 = insertvalue [3 x double*] %1, double* %[[ipc6]], 2
; SHARED-NEXT:   %3 = bitcast i8* %call to double*
; SHARED-NEXT:   store double %x, double* %3
; SHARED-NEXT:   %call1 = call fast double @augmented_f(double* %3, [3 x double*] %2)
; SHARED-NEXT:   %[[d4:.+]] = extractvalue [3 x double] %differeturn, 0
; SHARED-NEXT:   %[[m0diffecall1:.+]] = fmul fast double %[[d4]], %call1
; SHARED-NEXT:   %[[d5:.+]] = extractvalue [3 x double] %differeturn, 1
; SHARED-NEXT:   %[[m0diffecall11:.+]] = fmul fast double %[[d5]], %call1
; SHARED-NEXT:   %[[d6:.+]] = extractvalue [3 x double] %differeturn, 2
; SHARED-NEXT:   %[[m0diffecall12:.+]] = fmul fast double %[[d6]], %call1
; SHARED-NEXT:   %[[factor7:.+]] = fadd fast double %[[m0diffecall1]], %[[m0diffecall1]]
; SHARED-NEXT:   %[[factor8:.+]] = fadd fast double %[[m0diffecall11]], %[[m0diffecall11]]
; SHARED-NEXT:   %[[factor9:.+]] = fadd fast double %[[m0diffecall12]], %[[m0diffecall12]]
; SHARED-NEXT:   %[[i4:.+]] = insertvalue [3 x double] {{(undef|poison)}}, double %[[factor7]], 0
; SHARED-NEXT:   %[[i5:.+]] = insertvalue [3 x double] %[[i4]], double %[[factor8]], 1
; SHARED-NEXT:   %[[i6:.+]] = insertvalue [3 x double] %[[i5]], double %[[factor9]], 2
; SHARED-NEXT:   call void @diffe3f(double* %3, [3 x double*] %2, [3 x double] %[[i6]])
; SHARED-NEXT:   %[[res10:.+]] = load double, double* %[[ipc]]
; SHARED-NEXT:   %[[res11:.+]] = load double, double* %[[ipc5]]
; SHARED-NEXT:   %[[res12:.+]] = load double, double* %[[ipc6]]
; SHARED-NEXT:   store double 0.000000e+00, double* %[[ipc]]
; SHARED-NEXT:   store double 0.000000e+00, double* %[[ipc5]]
; SHARED-NEXT:   store double 0.000000e+00, double* %[[ipc6]]
; SHARED-NEXT:   call void bitcast (i32 (...)* @free to void (i8*)*)(i8* nonnull %[[alloc1]])
; SHARED-NEXT:   call void bitcast (i32 (...)* @free to void (i8*)*)(i8* nonnull %[[alloc2]])
; SHARED-NEXT:   call void bitcast (i32 (...)* @free to void (i8*)*)(i8* nonnull %[[alloc3]])
; SHARED-NEXT:   call void bitcast (i32 (...)* @free to void (i8*)*)(i8* %call)
; SHARED-NEXT:   %[[ret:.+]] = insertvalue [3 x double] {{(undef|poison)}}, double %[[res10]], 0
; SHARED-NEXT:   %[[ret2:.+]] = insertvalue [3 x double] %[[ret]], double %[[res11]], 1
; SHARED-NEXT:   %[[ret3:.+]] = insertvalue [3 x double] %[[ret2]], double %[[res12]], 2
; SHARED-NEXT:   %[[ret4:.+]] = insertvalue { [3 x double] } {{(undef|poison)}}, [3 x double] %[[ret3]], 0
; SHARED-NEXT:   ret { [3 x double] } %[[ret4]]

; SHARED: define internal void @diffe3f(double* nocapture readonly %x, [3 x double*] %"x'", [3 x double] %differeturn)
; SHARED-NEXT: entry:
; SHARED-NEXT:   %[[i0:.+]] = extractvalue [3 x double*] %"x'", 0
; SHARED-NEXT:   %[[i6:.+]] = extractvalue [3 x double] %differeturn, 0
; SHARED-NEXT:   %[[i1:.+]] = load double, double* %[[i0]]
; SHARED-NEXT:   %[[i7:.+]] = fadd fast double %[[i1]], %[[i6]]
; SHARED-NEXT:   store double %[[i7]], double* %[[i0]]
; SHARED-NEXT:   %[[i2:.+]] = extractvalue [3 x double*] %"x'", 1
; SHARED-NEXT:   %[[i8:.+]] = extractvalue [3 x double] %differeturn, 1
; SHARED-NEXT:   %[[i3:.+]] = load double, double* %[[i2]]
; SHARED-NEXT:   %[[i9:.+]] = fadd fast double %[[i3]], %[[i8]]
; SHARED-NEXT:   store double %[[i9]], double* %[[i2]]
; SHARED-NEXT:   %[[i4:.+]] = extractvalue [3 x double*] %"x'", 2
; SHARED-NEXT:   %[[i10:.+]] = extractvalue [3 x double] %differeturn, 2
; SHARED-NEXT:   %[[i5:.+]] = load double, double* %[[i4]]
; SHARED-NEXT:   %[[i11:.+]] = fadd fast double %[[i5]], %[[i10]]
; SHARED-NEXT:   store double %[[i11]], double* %[[i4]]
; SHARED-NEXT:   ret void
; SHARED-NEXT: }
