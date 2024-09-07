; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -fpopt-enable-herbie=0 -fpopt-enable-pt=1 -enzyme-print-fpopt -enzyme-print-herbie -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt" -fpopt-enable-herbie=0 -fpopt-enable-pt=1 -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = call fast double @llvm.cos.f64(double %x)
  %1 = fmul fast double %0, %0
  %2 = fsub fast double 1.000000e+00, %1
  ret double %2
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)

; CHECK: define double @tester(double %x)
; CHECK: entry:
; CHECK-NEXT:   %[[i0:.+]] = fptrunc double %x to half
; CHECK-NEXT:   %[[i1:.+]] = call fast half @llvm.cos.f16(half %[[i0]])
; CHECK-NEXT:   %[[i2:.+]] = fmul fast half %[[i1]], %[[i1]]
; CHECK-NEXT:   %[[i3:.+]] = fsub fast half 0xH3C00, %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = fpext half %[[i3]] to double
; CHECK-NEXT:   ret double %[[i4]]
