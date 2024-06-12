; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -O3 -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt,default<O3>" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

; CHECK: define double @tester(double %x, double %y)
; CHECK: entry:
; CHECK-NEXT:   %[[i0:.+]] = fadd fast double %y, %x
; CHECK-NEXT:   ret double %[[i0]]
