; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

declare { double, double } @Faddeeva_erfi({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfi({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define { double, double } @test_derivative({ double, double } %x) {
entry:
  %0 = tail call { double, double } ({ double, double } ({ double, double })*, ...) @__enzyme_autodiff({ double, double } ({ double, double })* nonnull @tester, metadata !"enzyme_out", { double, double } %x)
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_autodiff({ double, double } ({ double, double })*, ...)

; CHECK: define internal { { double, double } } @diffetester({ double, double } %in, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %[[a1:.+]] = extractvalue { double, double } %in, 1
; CHECK-NEXT:   %[[a3:.+]] = fmul fast double %[[a0]], %[[a0]]
; CHECK-NEXT:   %[[a2:.+]] = fmul fast double %[[a1]], %[[a1]]
; CHECK-NEXT:   %[[a4:.+]] = fsub fast double %[[a3]], %[[a2]]
; CHECK-NEXT:   %[[a5:.+]] = fmul fast double %[[a0]], %[[a1]]
; CHECK-NEXT:   %[[a6:.+]] = fadd fast double %[[a5]], %[[a5]]
; CHECK-NEXT:   %[[i9:.+]] = call fast double @llvm.exp.f64(double %[[a4]])
; CHECK-NEXT:   %[[i10:.+]] = call fast double @llvm.cos.f64(double %[[a6]])
; CHECK-NEXT:   %[[i11:.+]] = fmul fast double %[[i9]], %[[i10]]
; CHECK-NEXT:   %[[i12:.+]] = call fast double @llvm.sin.f64(double %[[a6]])
; CHECK-NEXT:   %[[i13:.+]] = fmul fast double %[[i9]], %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[i11]]
; CHECK-NEXT:   %[[i15:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[i13]]
; CHECK-NEXT:   %[[i16:.+]] = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %[[i17:.+]] = extractvalue { double, double } %differeturn, 1
; CHECK-NEXT:   %[[i19:.+]] = fmul fast double %[[i16]], %[[i14]]
; CHECK-NEXT:   %[[i18:.+]] = fmul fast double %[[i17]], %[[i15]]
; CHECK-NEXT:   %[[i20:.+]] = fsub fast double %[[i19]], %[[i18]]
; CHECK-NEXT:   %[[i22:.+]] = fmul fast double %[[i16]], %[[i15]]
; CHECK-NEXT:   %[[i21:.+]] = fmul fast double %[[i14]], %[[i17]]
; CHECK-NEXT:   %[[i23:.+]] = fadd fast double %[[i22]], %[[i21]]
; CHECK-NEXT:   %[[insert5:.+]] = insertvalue { double, double } {{(undef|poison)}}, double %[[i20]], 0
; CHECK-NEXT:   %[[insert8:.+]] = insertvalue { double, double } %[[insert5]], double %[[i23]], 1
; CHECK-NEXT:   %[[i24:.+]] = insertvalue { { double, double } } undef, { double, double } %[[insert8]], 0
; CHECK-NEXT:   ret { { double, double } } %[[i24]]
; CHECK-NEXT: }
