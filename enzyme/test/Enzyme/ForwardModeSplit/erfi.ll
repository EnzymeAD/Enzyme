; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

declare double @erfi(double)

define double @tester(double %x) {
entry:
  %call = call double @erfi(double %x)
  ret double %call
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwdsplit(double (double)* nonnull @tester, double %x, double 1.0, i8* null)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:    %0 = fmul fast double %x, %x
; CHECK-NEXT:    %1 = call fast double @llvm.exp.f64(double %0)
; CHECK-NEXT:    %2 = fmul fast double 0x3FF20DD750429B6D, %1
; CHECK-NEXT:    %3 = fmul fast double %"x'", %2
; CHECK-NEXT:    ret double %3
; CHECK-NEXT: }
