; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -enzyme-print-fpopt -enzyme-print-herbie -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  %1 = fsub fast double %0, %x
  ret double %1
}

; CHECK: define double @tester(double %x, double %y)
; CHECK: entry:
; CHECK-NEXT:   ret double %y
