; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, i32 %y) {
entry:
  %0 = tail call fast double @llvm.powi.f64.i32(double %x, i32 %y)
  ret double %0
}

define double @test_derivative(double %x, i32 %y) {
entry:
  %0 = tail call double (double (double, i32)*, ...) @__enzyme_autodiff(double (double, i32)* nonnull @tester, double %x, i32 %y)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.powi.f64.i32(double, i32)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i32)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double } @diffetester(double %x, i32 %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[cmp:.+]] = icmp eq i32 %y, 0
; CHECK-NEXT:    %[[sitofp:.+]] = sitofp i32 %y to double
; CHECK-NEXT:   %[[ym1:.+]] = sub i32 %y, 1
; CHECK-NEXT:   %[[newpow:.+]] = call fast double @llvm.powi.f64{{(\.i32)?}}(double %x, i32 %[[ym1]])
; CHECK-DAG:    %[[newpowdret:.+]] = fmul fast double %[[sitofp]], %[[newpow]]
; CHECK-NEXT:   %[[dx:.+]] = fmul fast double %differeturn, %[[newpowdret]]
; CHECK-NEXT:   %[[res:.+]] = select {{(fast )?}}i1 %[[cmp]], double 0.000000e+00, double %[[dx]]
; CHECK-NEXT:   %[[interres:.+]] = insertvalue { double } undef, double %[[res]], 0
; CHECK-NEXT:   ret { double } %[[interres]]
; CHECK-NEXT: }
