; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -gvn -dse -dse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,gvn,dse,dse)" -enzyme-preopt=false -S | FileCheck %s

define void @addOneMem(double* nocapture %x) {
entry:
  %0 = load double, double* %x
  %add = fadd double %0, 1.000000e+00
  store double %add, double* %x
  ret void
}

define void @test_derivative(double* %x, double* %xp1, double* %xp2, double* %xp3) {
entry:
  call void (void (double*)*, ...) @__enzyme_autodiff(void (double*)* nonnull @addOneMem, metadata !"enzyme_width", i64 3, double* %x, double* %xp1, double* %xp2, double* %xp3)
  ret void
}

declare void @__enzyme_autodiff(void (double*)*, ...)

; CHECK: define internal void @diffe3addOneMem(double* nocapture %x, [3 x double*] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"add'de" = alloca [3 x double]
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"add'de"
; CHECK-NEXT:   %"'de" = alloca [3 x double]
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"'de"
; CHECK-NEXT:   %[[i2:.+]] = load double, double* %x
; CHECK-NEXT:   %add = fadd double %[[i2]], 1.000000e+00
; CHECK-NEXT:   store double %add, double* %x
; CHECK-NEXT:   %[[xp1:.+]] = extractvalue [3 x double*] %"x'", 0
; CHECK-NEXT:   %[[i3:.+]] = load double, double* %[[xp1]]
; CHECK-NEXT:   %[[xp2:.+]] = extractvalue [3 x double*] %"x'", 1
; CHECK-NEXT:   %[[i4:.+]] = load double, double* %[[xp2]]
; CHECK-NEXT:   %[[xp3:.+]] = extractvalue [3 x double*] %"x'", 2
; CHECK-NEXT:   %[[i5:.+]] = load double, double* %[[xp3]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[xp1]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[xp2]]
; CHECK-NEXT:   store double 0.000000e+00, double* %[[xp3]]
; CHECK-NEXT:   %[[i6:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"add'de", i32 0, i32 0
; CHECK-NEXT:   %[[i7:.+]] = load double, double* %[[i6]]
; CHECK-NEXT:   %[[i8:.+]] = fadd fast double %[[i7]], %[[i3]]
; CHECK-NEXT:   store double %[[i8]], double* %[[i6]]
; CHECK-NEXT:   %[[i9:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"add'de", i32 0, i32 1
; CHECK-NEXT:   %[[i10:.+]] = load double, double* %[[i9]]
; CHECK-NEXT:   %[[i11:.+]] = fadd fast double %[[i10]], %[[i4]]
; CHECK-NEXT:   store double %[[i11]], double* %[[i9]]
; CHECK-NEXT:   %[[i12:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"add'de", i32 0, i32 2
; CHECK-NEXT:   %[[i13:.+]] = load double, double* %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = fadd fast double %[[i13]], %[[i5]]
; CHECK-NEXT:   store double %[[i14]], double* %[[i12]]
; CHECK-NEXT:   %[[i15:.+]] = load [3 x double], [3 x double]* %"add'de"
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"add'de"
; CHECK-NEXT:   %[[i16:.+]] = extractvalue [3 x double] %[[i15]], 0
; CHECK-NEXT:   %[[i20:.+]] = extractvalue [3 x double] %[[i15]], 1
; CHECK-NEXT:   %[[i24:.+]] = extractvalue [3 x double] %[[i15]], 2
; CHECK-NEXT:   %[[i17:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"'de", i32 0, i32 0
; CHECK-NEXT:   %[[i18:.+]] = load double, double* %[[i17]]
; CHECK-NEXT:   %[[i19:.+]] = fadd fast double %[[i18]], %[[i16]]
; CHECK-NEXT:   store double %[[i19]], double* %[[i17]]
; CHECK-NEXT:   %[[i21:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"'de", i32 0, i32 1
; CHECK-NEXT:   %[[i22:.+]] = load double, double* %[[i21]]
; CHECK-NEXT:   %[[i23:.+]] = fadd fast double %[[i22]], %[[i20]]
; CHECK-NEXT:   store double %[[i23]], double* %[[i21:.+]]
; CHECK-NEXT:   %[[i25:.+]] = getelementptr inbounds [3 x double], [3 x double]* %"'de", i32 0, i32 2
; CHECK-NEXT:   %[[i26:.+]] = load double, double* %[[i25]]
; CHECK-NEXT:   %[[i27:.+]] = fadd fast double %[[i26]], %[[i24]]
; CHECK-NEXT:   store double %[[i27]], double* %[[i25]]
; CHECK-NEXT:   %[[i28:.+]] = load [3 x double], [3 x double]* %"'de"
; CHECK-NEXT:   store [3 x double] zeroinitializer, [3 x double]* %"'de"
; CHECK-NEXT:   %[[i31:.+]] = extractvalue [3 x double] %[[i28]], 0
; CHECK-NEXT:   %[[i29:.+]] = load double, double* %[[xp1]]
; CHECK-NEXT:   %[[i32:.+]] = fadd fast double %[[i29]], %[[i31]]
; CHECK-NEXT:   store double %[[i32]], double* %[[xp1]]
; CHECK-NEXT:   %[[i33:.+]] = extractvalue [3 x double] %[[i28]], 1
; CHECK-NEXT:   %[[i30:.+]] = load double, double* %[[xp2]]
; CHECK-NEXT:   %[[i34:.+]] = fadd fast double %[[i30]], %[[i33]]
; CHECK-NEXT:   store double %[[i34]], double* %[[xp2]]
; CHECK-NEXT:   %[[i35:.+]] = extractvalue [3 x double] %[[i28]], 2
; CHECK-NEXT:   %[[i36:.+]] = load double, double* %[[xp3]]
; CHECK-NEXT:   %[[i37:.+]] = fadd fast double %[[i36]], %[[i35]]
; CHECK-NEXT:   store double %[[i37]], double* %[[xp3]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
