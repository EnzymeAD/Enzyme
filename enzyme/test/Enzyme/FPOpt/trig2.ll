; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -O3 -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt,default<O3>" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %cmp = fcmp olt double %x, 0.0        ; Compare x < 0.0
  br i1 %cmp, label %less, label %notless

less:                                     ; If x < 0
  %sin = call fast double @llvm.sin.f64(double %x)
  %squared1 = fmul fast double %sin, %sin
  %0 = fsub fast double 1.000000e+00, %squared1
  br label %merge

notless:                                  ; If x >= 0
  %cos = call fast double @llvm.cos.f64(double %x)
  %squared2 = fmul fast double %cos, %cos
  %1 = fsub fast double 1.000000e+00, %squared2
  br label %merge

merge:                                    ; Merge point, use of phi node
  %result = phi double [ %0, %less ], [ %1, %notless ]
  ret double %result
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)

; CHECK: define double @tester(double %x)
; CHECK: entry:
; CHECK-NEXT:   %[[i0:.+]] = fcmp olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[i1:.+]] = tail call fast double @llvm.cos.f64(double %x)
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %0, %0
; CHECK-NEXT:   %[[i3:.+]] = tail call fast double @llvm.sin.f64(double %x)
; CHECK-NEXT:   %[[i4:.+]] = fmul fast double %1, %1
; CHECK-NEXT:   %[[i5:.+]] = select i1 %cmp, double %square1, double %square
; CHECK-NEXT:   ret double %[[i5]]
