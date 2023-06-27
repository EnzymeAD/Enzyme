; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

declare { double, double } @Faddeeva_erf({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erf({ double, double } %in, double 0.000000e+00)
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
; CHECK-DAG:    %[[a2:.+]] = fmul fast double %[[a1]], %[[a1]]
; CHECK-DAG:    %[[a3:.+]] = fmul fast double %[[a0]], %[[a0]]
; CHECK-NEXT:   %[[a4:.+]] = fsub fast double %[[a3]], %[[a2]]
; CHECK-NEXT:   %[[a5:.+]] = fmul fast double %[[a0]], %[[a1]]
; CHECK-NEXT:   %[[a6:.+]] = fadd fast double %[[a5]], %[[a5]]
; CHECK-NEXT:   %[[a7:.+]] = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %[[a4]]
; CHECK-NEXT:   %[[a8:.+]] = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %[[a6]]
; CHECK-NEXT:   %[[a9:.+]] = call fast double @llvm.exp.f64(double %[[a7]])
; CHECK-NEXT:   %[[a10:.+]] = call fast double @llvm.cos.f64(double %[[a8]])
; CHECK-NEXT:   %[[a11:.+]] = fmul fast double %[[a9]], %[[a10]]
; CHECK-NEXT:   %[[a12:.+]] = call fast double @llvm.sin.f64(double %[[a8]])
; CHECK-NEXT:   %[[a13:.+]] = fmul fast double %[[a9]], %[[a12]]
; CHECK-NEXT:   %[[a14:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a11]]
; CHECK-NEXT:   %[[a15:.+]] = fmul fast double 0x3FF20DD750429B6D, %[[a13]]
; CHECK-NEXT:   %[[a16:.+]] = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %[[a17:.+]] = extractvalue { double, double } %differeturn, 1
; CHECK-DAG:    %[[a19:.+]] = fmul fast double %[[a16]], %[[a14]]
; CHECK-DAG:    %[[a18:.+]] = fmul fast double %[[a17]], %[[a15]]
; CHECK-NEXT:   %[[a20:.+]] = fsub fast double %[[a19]], %[[a18]]
; CHECK-DAG:    %[[a22:.+]] = fmul fast double %[[a16]], %[[a15]]
; CHECK-DAG:    %[[a21:.+]] = fmul fast double %[[a14]], %[[a17]]
; CHECK-NEXT:   %[[a23:.+]] = fadd fast double %[[a22]], %[[a21]]
; CHECK-NEXT:   %[[insert5:.+]] = insertvalue { double, double } {{(undef|poison)}}, double %[[a20]], 0
; CHECK-NEXT:   %[[insert8:.+]] = insertvalue { double, double } %[[insert5]], double %[[a23]], 1
; CHECK-NEXT:   %[[a24:.+]] = insertvalue { { double, double } } undef, { double, double } %[[insert8]], 0
; CHECK-NEXT:   ret { { double, double } } %[[a24]]
; CHECK-NEXT: }
