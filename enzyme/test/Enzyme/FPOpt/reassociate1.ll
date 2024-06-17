; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -enzyme-preopt=false -S | FileCheck %s; fi
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
; CHECK-NEXT:   %[[i0:.+]] = fmul fast double %x, 2.000000e+00
; CHECK-NEXT:   %[[i1:.+]] = fadd fast double %y, %[[i0]]
; CHECK-NEXT:   ret double %[[i1]]

