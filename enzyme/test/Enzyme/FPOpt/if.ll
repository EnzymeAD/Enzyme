; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -enzyme-print-fpopt -enzyme-print-herbie -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %a, double %b, double %c) {
entry:
  %0 = fmul double %a, %c
  %1 = fmul double 4.000000e+00, %0
  %2 = fmul double %b, %b
  %3 = fsub double %2, %1
  %4 = call double @llvm.sqrt.f64(double %3)
  %5 = fsub double 0.000000e+00, %b
  %6 = fsub double %5, %4
  %7 = fmul double 2.000000e+00, %a
  %8 = fdiv double %6, %7
  ret double %8
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double)

; CHECK: define double @tester(double %a, double %b, double %c)
; CHECK: entry:
; CHECK-NEXT:   %[[i0:.+]] = fcmp fast ole double %b, -6.800000e+00
; CHECK-NEXT:   br i1 %[[i0]], label %[[i1:.+]], label %[[i2:.+]]

; CHECK: [[i1]]:
; CHECK-NEXT:   %[[i3:.+]] = fdiv fast double %c, %b
; CHECK-NEXT:   %[[i4:.+]] = fneg fast double %c
; CHECK-NEXT:   %[[i5:.+]] = fdiv fast double %b, %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = fdiv fast double %[[i3]], %[[i5]]
; CHECK-NEXT:   %[[i7:.+]] = fmul fast double %a, %[[i6]]
; CHECK-NEXT:   %[[i8:.+]] = fsub fast double %[[i7]], %c
; CHECK-NEXT:   %[[i9:.+]] = fdiv fast double %[[i8]], %b
; CHECK-NEXT:   br label %[[i10:.+]]

; CHECK: [[i2]]:
; CHECK-NEXT:   %[[i11:.+]] = fcmp fast ole double %b, 0x52682111B1052222
; CHECK-NEXT:   br i1 %[[i11]], label %[[i12:.+]], label %[[i13:.+]]

; CHECK: [[i12]]:
; CHECK-NEXT:   %[[i14:.+]] = fmul fast double %c, -4.000000e+00
; CHECK-NEXT:   %[[i15:.+]] = fmul fast double %b, %b
; CHECK-NEXT:   %[[i16:.+]] = call fast double @llvm.fma.f64(double %a, double %[[i14]], double %[[i15]])
; CHECK-NEXT:   %[[i17:.+]] = call fast double @llvm.sqrt.f64(double %[[i16]])
; CHECK-NEXT:   %[[i18:.+]] = fadd fast double %b, %[[i17]]
; CHECK-NEXT:   %[[i19:.+]] = fneg fast double %a
; CHECK-NEXT:   %[[i20:.+]] = fmul fast double 2.000000e+00, %[[i19]]
; CHECK-NEXT:   %[[i21:.+]] = fdiv fast double %[[i18]], %[[i20]]
; CHECK-NEXT:   br label %[[i22:.+]]

; CHECK: [[i13]]:
; CHECK-NEXT:   %[[i23:.+]] = fdiv fast double %c, %b
; CHECK-NEXT:   %[[i24:.+]] = fdiv fast double %b, %a
; CHECK-NEXT:   %[[i25:.+]] = fsub fast double %[[i23]], %[[i24]]
; CHECK-NEXT:   br label %[[i26:.+]]

; CHECK: [[i26]]:
; CHECK-NEXT:   %[[i27:.+]] = phi fast double [ %[[i21]], %[[i12]] ], [ %[[i25]], %[[i13]] ]
; CHECK-NEXT:   br label %[[i28:.+]]

; CHECK: [[i28]]:
; CHECK-NEXT:   %[[i29:.+]] = phi fast double [ %[[i9]], %[[i1]] ], [ %[[i27]], %[[i22]] ]
; CHECK-NEXT:   ret double %[[i29]]
