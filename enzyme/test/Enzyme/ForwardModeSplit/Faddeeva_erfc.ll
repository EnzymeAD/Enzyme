; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -early-cse -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

declare { double, double } @Faddeeva_erfc({ double, double }, double)

define { double, double } @tester({ double, double } %in) {
entry:
  %call = call { double, double } @Faddeeva_erfc({ double, double } %in, double 0.000000e+00)
  ret { double, double } %call
}

define { double, double } @test_derivative({ double, double } %x) {
entry:
  %0 = tail call { double, double } ({ double, double } ({ double, double })*, ...) @__enzyme_fwdsplit({ double, double } ({ double, double })* @tester, { double, double } %x, { double, double } { double 1.0, double 1.0 }, i8* null)
  ret { double, double } %0
}

; Function Attrs: nounwind
declare { double, double } @__enzyme_fwdsplit({ double, double } ({ double, double })*, ...)


; CHECK: define internal { double, double } @fwddiffetester({ double, double } %in, { double, double } %"in'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a0:.+]] = extractvalue { double, double } %in, 0
; CHECK-NEXT:   %[[a1:.+]] = extractvalue { double, double } %in, 1
; CHECK-DAG:    %[[a2:.+]] = fmul fast double %[[a0]], %[[a0]]
; CHECK-DAG:    %[[a3:.+]] = fmul fast double %[[a1]], %[[a1]]
; CHECK-NEXT:   %[[a4:.+]] = fsub fast double %[[a2]], %[[a3]]
; CHECK-NEXT:   %[[a5:.+]] = fmul fast double %[[a0]], %[[a1]]
; CHECK-NEXT:   %[[a6:.+]] = fadd fast double %[[a5]], %[[a5]]
; CHECK-NEXT:   %[[a7:.+]] = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %[[a4]]
; CHECK-NEXT:   %[[a8:.+]] = {{(fsub fast double \-0.000000e\+00,|fneg fast double)}} %[[a6]]
; CHECK-NEXT:   %[[a9:.+]] = call fast double @llvm.exp.f64(double %[[a7]])
; CHECK-NEXT:   %[[a10:.+]] = call fast double @llvm.cos.f64(double %[[a8]])
; CHECK-NEXT:   %[[a11:.+]] = fmul fast double %[[a9]], %[[a10]]
; CHECK-NEXT:   %[[a12:.+]] = call fast double @llvm.sin.f64(double %[[a8]])
; CHECK-NEXT:   %[[a13:.+]] = fmul fast double %[[a9]], %[[a12]]
; CHECK-NEXT:   %[[a14:.+]] = fmul fast double 0xBFF20DD750429B6D, %[[a11]]
; CHECK-NEXT:   %[[a16:.+]] = fmul fast double 0xBFF20DD750429B6D, %[[a13]]
; CHECK-NEXT:   %[[a18:.+]] = extractvalue { double, double } %"in'", 0
; CHECK-NEXT:   %[[a19:.+]] = extractvalue { double, double } %"in'", 1
; CHECK-DAG:    %[[a20:.+]] = fmul fast double %[[a18]], %[[a14]]
; CHECK-DAG:    %[[a21:.+]] = fmul fast double %[[a19]], %[[a16]]
; CHECK-NEXT:   %[[a22:.+]] = fsub fast double %[[a20]], %[[a21]]
; CHECK-DAG:    %[[a24:.+]] = fmul fast double %[[a18]], %[[a16]]
; CHECK-DAG:    %[[a25:.+]] = fmul fast double %[[a14]], %[[a19]]
; CHECK-NEXT:   %[[a26:.+]] = fadd fast double %[[a24]], %[[a25]]
; CHECK-NEXT:   %[[a23:.+]] = insertvalue { double, double } undef, double %[[a22]], 0
; CHECK-NEXT:   %[[a27:.+]] = insertvalue { double, double } %[[a23]], double %[[a26]], 1
; CHECK-NEXT:   ret { double, double } %[[a27]]
; CHECK-NEXT: }
