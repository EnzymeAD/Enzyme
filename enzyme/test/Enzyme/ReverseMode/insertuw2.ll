; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg)" -S | FileCheck %s

define void @tester(double* %in0, double* %in1, i1 %c) {
entry:
  br i1 %c, label %trueb, label %exit

trueb:
  %pre_x0 = load double, double* %in0
  store double 0.000000e+00, double* %in0
  %x0 = insertvalue {double, double, double*} undef, double %pre_x0, 0

  %pre_x1 = load double, double* %in1
  store double 0.000000e+00, double* %in1
  %x1 = insertvalue {double, double, double*} %x0, double %pre_x1, 1

  %out1 = insertvalue {double, double, double*} %x1, double* %in0, 2
  %out2 = insertvalue {double, double, double*} %out1, double 0.000000e+00, 1
  
  %post_x0 = extractvalue {double, double, double*} %out2, 0
  %post_x1 = extractvalue {double, double, double*} %x1, 1
  
  %mul0 = fmul double %post_x0, %post_x1
  store double %mul0, double* %in0   
  
  br label %exit

exit:
  ret void
}

define void @test_derivative(double* %x, double* %dx, double* %y, double* %dy) {
entry:
  tail call void (...) @__enzyme_autodiff(void (double*, double*, i1)* nonnull @tester, double* %x, double* %dx, double* %y, double* %dy, i1 true)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffetester(double* %in0, double* %"in0'", double* %in1, double* %"in1'", i1 %c)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x1'de" = alloca { double, double, double* }
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   %"out2'de" = alloca { double, double, double* }
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"out2'de"
; CHECK-NEXT:   %"out1'de" = alloca { double, double, double* }
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"out1'de"
; CHECK-NEXT:   %"x0'de" = alloca { double, double, double* }
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"x0'de"
; CHECK-NEXT:   br i1 %c, label %trueb, label %exit

; CHECK: trueb:                                            ; preds = %entry
; CHECK-NEXT:   %pre_x0 = load double, double* %in0
; CHECK-NEXT:   store double 0.000000e+00, double* %in0
; CHECK-NEXT:   %x0 = insertvalue { double, double, double* } undef, double %pre_x0, 0
; CHECK-NEXT:   %pre_x1 = load double, double* %in1
; CHECK-NEXT:   store double 0.000000e+00, double* %in1
; CHECK-NEXT:   %x1 = insertvalue { double, double, double* } %x0, double %pre_x1, 1
; CHECK-NEXT:   %out1 = insertvalue { double, double, double* } %x1, double* %in0, 2
; CHECK-NEXT:   %out2 = insertvalue { double, double, double* } %out1, double 0.000000e+00, 1
; CHECK-NEXT:   %post_x0 = extractvalue { double, double, double* } %out2, 0
; CHECK-NEXT:   %post_x1 = extractvalue { double, double, double* } %x1, 1
; CHECK-NEXT:   %mul0 = fmul double %post_x0, %post_x1
; CHECK-NEXT:   store double %mul0, double* %in0
; CHECK-NEXT:   br label %exit

; CHECK: exit:                                             ; preds = %trueb, %entry
; CHECK-NEXT:   %x1_cache.0 = phi { double, double, double* } [ %x1, %trueb ], [ undef, %entry ]
; CHECK-NEXT:   br label %invertexit

; CHECK: invertentry:                                      ; preds = %invertexit, %inverttrueb
; CHECK-NEXT:   ret void

; CHECK: inverttrueb:                                      ; preds = %invertexit
; CHECK-NEXT:   %0 = load double, double* %"in0'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"in0'"
; CHECK-NEXT:   %[[i1:.+]] = fadd fast double 0.000000e+00, %0
; CHECK-NEXT:   %post_x1_unwrap = extractvalue { double, double, double* } %x1_cache.0, 1
; CHECK-NEXT:   %[[m0diffepost_x0:.+]] = fmul fast double %[[i1]], %post_x1_unwrap
; CHECK-NEXT:   %[[i2:.+]] = fadd fast double 0.000000e+00, %[[m0diffepost_x0]]
; CHECK-NEXT:   %out1_unwrap = insertvalue { double, double, double* } %x1_cache.0, double* %in0, 2
; CHECK-NEXT:   %out2_unwrap = insertvalue { double, double, double* } %out1_unwrap, double 0.000000e+00, 1
; CHECK-NEXT:   %post_x0_unwrap = extractvalue { double, double, double* } %out2_unwrap, 0
; CHECK-NEXT:   %[[m1diffepost_x1:.+]] = fmul fast double %[[i1]], %post_x0_unwrap
; CHECK-NEXT:   %[[i3:.+]] = fadd fast double 0.000000e+00, %[[m1diffepost_x1]]
; CHECK-NEXT:   %[[i4:.+]] = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x1'de", i32 0, i32 1
; CHECK-NEXT:   %[[i5:.+]] = load double, double* %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = fadd fast double %[[i5]], %[[i3]]
; CHECK-NEXT:   store double %[[i6]], double* %[[i4]]
; CHECK-NEXT:   %[[i7:.+]] = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"out2'de", i32 0, i32 0
; CHECK-NEXT:   %[[i8:.+]] = load double, double* %[[i7]]
; CHECK-NEXT:   %[[i9:.+]] = fadd fast double %[[i8]], %[[i2]]
; CHECK-NEXT:   store double %[[i9]], double* %[[i7]]
; CHECK-NEXT:   %[[i10:.+]] = load { double, double, double* }, { double, double, double* }* %"out2'de"
; CHECK-NEXT:   %[[i11:.+]] = insertvalue { double, double, double* } %[[i10]], double 0.000000e+00, 1
; CHECK-NEXT:   %[[i12:.+]] = load { double, double, double* }, { double, double, double* }* %"out1'de"
; CHECK-NEXT:   %[[i13:.+]] = extractvalue { double, double, double* } %[[i10]], 0
; CHECK-NEXT:   %[[i14:.+]] = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"out1'de", i32 0, i32 0
; CHECK-NEXT:   %[[i15:.+]] = load double, double* %[[i14]]
; CHECK-NEXT:   %[[i16:.+]] = fadd fast double %[[i15]], %[[i13]]
; CHECK-NEXT:   store double %[[i16]], double* %[[i14]]
; CHECK-NEXT:   %[[i17:.+]] = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"out1'de", i32 0, i32 1
; CHECK-NEXT:   %[[i18:.+]] = load double, double* %[[i17]]
; CHECK-NEXT:   %[[i19:.+]] = fadd fast double %[[i18]], 0.000000e+00
; CHECK-NEXT:   store double %[[i19]], double* %[[i17]]
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"out2'de"
; CHECK-NEXT:   %[[i20:.+]] = load { double, double, double* }, { double, double, double* }* %"out1'de"
; CHECK-NEXT:   %[[i21:.+]] = insertvalue { double, double, double* } %[[i20]], double* null, 2
; CHECK-NEXT:   %[[i22:.+]] = load { double, double, double* }, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   %[[i23:.+]] = extractvalue { double, double, double* } %[[i20]], 0
; CHECK-NEXT:   %[[i24:.+]] = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x1'de", i32 0, i32 0
; CHECK-NEXT:   %[[i25:.+]] = load double, double* %[[i24]]
; CHECK-NEXT:   %[[i26:.+]] = fadd fast double %[[i25]], %[[i23]]
; CHECK-NEXT:   store double %[[i26]], double* %[[i24]]
; CHECK-NEXT:   %[[i27:.+]] = extractvalue { double, double, double* } %[[i20]], 1
; CHECK-NEXT:   %[[i28:.+]] = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x1'de", i32 0, i32 1
; CHECK-NEXT:   %[[i29:.+]] = load double, double* %[[i28]]
; CHECK-NEXT:   %[[i30:.+]] = fadd fast double %[[i29]], %[[i27]]
; CHECK-NEXT:   store double %[[i30]], double* %[[i28]]
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"out1'de"
; CHECK-NEXT:   %[[i31:.+]] = load { double, double, double* }, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   %[[i32:.+]] = extractvalue { double, double, double* } %[[i31]], 1
; CHECK-NEXT:   %[[i33:.+]] = fadd fast double 0.000000e+00, %[[i32]]
; CHECK-NEXT:   %[[i34:.+]] = load { double, double, double* }, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   %[[i35:.+]] = insertvalue { double, double, double* } %[[i34]], double 0.000000e+00, 1
; CHECK-NEXT:   %[[i36:.+]] = load { double, double, double* }, { double, double, double* }* %"x0'de"
; CHECK-NEXT:   %[[i37:.+]] = extractvalue { double, double, double* } %[[i34]], 0
; CHECK-NEXT:   %[[i38:.+]] = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x0'de", i32 0, i32 0
; CHECK-NEXT:   %[[i39:.+]] = load double, double* %[[i38]]
; CHECK-NEXT:   %[[i40:.+]] = fadd fast double %[[i39]], %[[i37]]
; CHECK-NEXT:   store double %[[i40]], double* %[[i38]]
; CHECK-NEXT:   %[[i41:.+]] = getelementptr inbounds { double, double, double* }, { double, double, double* }* %"x0'de", i32 0, i32 1
; CHECK-NEXT:   %[[i42:.+]] = load double, double* %[[i41]]
; CHECK-NEXT:   %[[i43:.+]] = fadd fast double %[[i42]], 0.000000e+00
; CHECK-NEXT:   store double %[[i43]], double* %[[i41]]
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"x1'de"
; CHECK-NEXT:   store double 0.000000e+00, double* %"in1'"
; CHECK-NEXT:   %[[i44:.+]] = load double, double* %"in1'"
; CHECK-NEXT:   %[[i45:.+]] = fadd fast double %[[i44]], %[[i33]]
; CHECK-NEXT:   store double %[[i45]], double* %"in1'"
; CHECK-NEXT:   %[[i46:.+]] = load { double, double, double* }, { double, double, double* }* %"x0'de"
; CHECK-NEXT:   %[[i47:.+]] = extractvalue { double, double, double* } %[[i46]], 0
; CHECK-NEXT:   %[[i48:.+]] = fadd fast double 0.000000e+00, %[[i47]]
; CHECK-NEXT:   store { double, double, double* } zeroinitializer, { double, double, double* }* %"x0'de"
; CHECK-NEXT:   store double 0.000000e+00, double* %"in0'"
; CHECK-NEXT:   %[[i49:.+]] = load double, double* %"in0'"
; CHECK-NEXT:   %[[i50:.+]] = fadd fast double %[[i49]], %[[i48]]
; CHECK-NEXT:   store double %[[i50]], double* %"in0'"
; CHECK-NEXT:   br label %invertentry

; CHECK: invertexit:                                       ; preds = %exit
; CHECK-NEXT:   br i1 %c, label %inverttrueb, label %invertentry
; CHECK-NEXT: }
