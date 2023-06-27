; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -early-cse -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

declare { double, double } @Faddeeva_erfi({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfi({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define { double, double } @test_derivative({ double, double } %x) {
entry:
  %0 = tail call { double, double } ({ double, double } ({ double, double })*, ...) @__enzyme_fwdsplit({ double, double } ({ double, double })* nonnull @tester, { double, double } %x, { double, double } { double 1.0, double 1.0 }, i8* null)
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_fwdsplit({ double, double } ({ double, double })*, ...)


; CHECK: define internal { double, double } @fwddiffetester({ double, double } %in, { double, double } %"in'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %[[a1:.+]] = extractvalue { double, double } %in, 1
; CHECK-NEXT:   %[[a2:.+]] = fmul fast double %[[a0]], %[[a0]]
; CHECK-NEXT:   %[[a3:.+]] = fmul fast double %[[a1]], %[[a1]]
; CHECK-NEXT:   %[[a4:.+]] = fsub fast double %[[a2]], %[[a3]]
; CHECK-NEXT:   %[[a5:.+]] = fmul fast double %[[a0]], %[[a1]]
; CHECK-NEXT:   %[[a6:.+]] = fadd fast double %[[a5]], %[[a5]]
; CHECK-NEXT:   %[[a7:.+]] = call fast double @llvm.exp.f64(double %[[a4]])
; CHECK-NEXT:   %[[a8:.+]] = call fast double @llvm.cos.f64(double %[[a6]])
; CHECK-NEXT:   %[[a9:.+]] = fmul fast double %[[a7]], %[[a8]]
; CHECK-NEXT:   %[[a10:.+]] = call fast double @llvm.sin.f64(double %[[a6]])
; CHECK-NEXT:   %[[a11:.+]] = fmul fast double %[[a7]], %[[a10]]
; CHECK-NEXT:   %[[a12:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a9]]
; CHECK-NEXT:   %[[a14:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a11]]
; CHECK-NEXT:   %[[a16:.+]] = extractvalue { double, double } %"in'", 0
; CHECK-NEXT:   %[[a17:.+]] = extractvalue { double, double } %"in'", 1
; CHECK-NEXT:   %[[a19:.+]] = fmul fast double %[[a16]], %[[a12]]
; CHECK-NEXT:   %[[a18:.+]] = fmul fast double %[[a17]], %[[a14]]
; CHECK-NEXT:   %[[a20:.+]] = fsub fast double %[[a19]], %[[a18]]
; CHECK-NEXT:   %[[a23:.+]] = fmul fast double %[[a16]], %[[a14]]
; CHECK-NEXT:   %[[a22:.+]] = fmul fast double %[[a12]], %[[a17]]
; CHECK-NEXT:   %[[a24:.+]] = fadd fast double %[[a23]], %[[a22]]
; CHECK-NEXT:   %[[a21:.+]] = insertvalue { double, double } undef, double %[[a20]], 0
; CHECK-NEXT:   %[[a25:.+]] = insertvalue { double, double } %[[a21]], double %[[a24]], 1
; CHECK-NEXT:   ret { double, double } %[[a25:.+]]
; CHECK-NEXT: }
