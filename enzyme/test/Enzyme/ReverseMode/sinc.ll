; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @sinc(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @sinc(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[abs:.+]] = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT:   %[[small:.+]] = fcmp fast olt double %[[abs]], 4.400000e-01
; CHECK-NEXT:   %[[usq:.+]] = fmul fast double %x, %x
; CHECK-NEXT:   %[[p13:.+]] = fmul fast double %[[usq]], 0xBDA78AF56A4D411B
; CHECK-NEXT:   %[[s11:.+]] = fadd fast double 0x3E208DB48EBE51C7, %[[p13]]
; CHECK-NEXT:   %[[p11:.+]] = fmul fast double %[[usq]], %[[s11]]
; CHECK-NEXT:   %[[s9:.+]] = fadd fast double 0xBE90CFEB60F94B0E, %[[p11]]
; CHECK-NEXT:   %[[p9:.+]] = fmul fast double %[[usq]], %[[s9]]
; CHECK-NEXT:   %[[s7:.+]] = fadd fast double 0x3EF71DE3A556C734, %[[p9]]
; CHECK-NEXT:   %[[p7:.+]] = fmul fast double %[[usq]], %[[s7]]
; CHECK-NEXT:   %[[s5:.+]] = fadd fast double 0xBF53813813813814, %[[p7]]
; CHECK-NEXT:   %[[p5:.+]] = fmul fast double %[[usq]], %[[s5]]
; CHECK-NEXT:   %[[s3:.+]] = fadd fast double 0x3FA1111111111111, %[[p5]]
; CHECK-NEXT:   %[[p3:.+]] = fmul fast double %[[usq]], %[[s3]]
; CHECK-NEXT:   %[[s1:.+]] = fadd fast double 0xBFD5555555555555, %[[p3]]
; CHECK-NEXT:   %[[series:.+]] = fmul fast double %x, %[[s1]]
; CHECK-NEXT:   %[[safe:.+]] = select {{(fast )?}}i1 %[[small]], double 1.000000e+00, double %x
; CHECK-NEXT:   %[[cos:.+]] = call fast double @llvm.cos.f64(double %[[safe]])
; CHECK-NEXT:   %[[call:.+]] = call fast double @sinc(double %[[safe]])
; CHECK-NEXT:   %[[num:.+]] = fsub fast double %[[cos]], %[[call]]
; CHECK-NEXT:   %[[closed:.+]] = fdiv fast double %[[num]], %[[safe]]
; CHECK-NEXT:   %[[deriv:.+]] = select {{(fast )?}}i1 %[[small]], double %[[series]], double %[[closed]]
; CHECK-NEXT:   %[[res:.+]] = fmul fast double %differeturn, %[[deriv]]
; CHECK-NEXT:   %[[iv:.+]] = insertvalue { double } undef, double %[[res]], 0
; CHECK-NEXT:   ret { double } %[[iv]]
; CHECK-NEXT: }
