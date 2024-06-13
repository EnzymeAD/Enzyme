; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -enzyme-preopt=false -S | %opt -O3 -S - | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt,default<O3>" -enzyme-preopt=false -S | FileCheck %s

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
; CHECK-NEXT:   %[[i0:.+]] = tail call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %[[i0]], %[[i0]]
; CHECK-NEXT:   ret double %[[i1]]
