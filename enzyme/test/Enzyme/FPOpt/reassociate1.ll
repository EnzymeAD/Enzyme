; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -enzyme-print-fpopt -enzyme-print-herbie -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  %1 = fadd fast double %0, %x
  ret double %1
}

; CHECK: define double @tester(double %x, double %y)
; CHECK: entry:
; CHECK-NEXT:   %[[i0:.+]] = call fast double @llvm.fmuladd.f64(double %x, double 2.000000e+00, double %y)
; CHECK-NEXT:   ret double %[[i0]]

