; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -enzyme-print-fpopt -enzyme-print-herbie -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define void @tester() {
entry:
  %arr = alloca double, i64 5, align 8
  
  ; Compute addresses for array elements
  %ptr1 = getelementptr inbounds double, double* %arr, i64 0
  %ptr2 = getelementptr inbounds double, double* %arr, i64 1
  %ptr3 = getelementptr inbounds double, double* %arr, i64 2
  %ptr4 = getelementptr inbounds double, double* %arr, i64 3
  %ptr5 = getelementptr inbounds double, double* %arr, i64 4
  
  ; Store constants into the array
  store double 3.141592653589793, double* %ptr1, align 8
  store double 2.718281828459045, double* %ptr2, align 8
  store double 1.4142135623730951, double* %ptr3, align 8
  store double 0.5772156649015329, double* %ptr4, align 8
  store double 0.6931471805599453, double* %ptr5, align 8

  ; Load constants from the array
  %val1 = load double, double* %ptr1, align 8
  %val2 = load double, double* %ptr2, align 8
  %val3 = load double, double* %ptr3, align 8
  %val4 = load double, double* %ptr4, align 8
  %val5 = load double, double* %ptr5, align 8

  ; Perform computations
  %cos_val = call fast double @llvm.cos.f64(double %val1)
  %sin_val = call fast double @llvm.sin.f64(double %val2)
  %exp_val = call fast double @llvm.exp.f64(double %val3)
  %log_val = call fast double @llvm.log.f64(double %val4)
  %sum_val = fadd fast double %cos_val, %sin_val

  ; Store results back to the array
  store double %cos_val, double* %ptr1, align 8
  store double %sin_val, double* %ptr2, align 8
  store double %exp_val, double* %ptr3, align 8
  store double %log_val, double* %ptr4, align 8
  store double %sum_val, double* %ptr5, align 8

  ret void
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double)

; CHECK: define void @tester()
; CHECK: entry:
; CHECK-NEXT:   %arr = alloca double, i64 5, align 8
; CHECK-NEXT:   %ptr1 = getelementptr inbounds double, double* %arr, i64 0
; CHECK-NEXT:   %ptr2 = getelementptr inbounds double, double* %arr, i64 1
; CHECK-NEXT:   %ptr3 = getelementptr inbounds double, double* %arr, i64 2
; CHECK-NEXT:   %ptr4 = getelementptr inbounds double, double* %arr, i64 3
; CHECK-NEXT:   %ptr5 = getelementptr inbounds double, double* %arr, i64 4
; CHECK-NEXT:   store double 0x400921FB54442D18, double* %ptr1, align 8
; CHECK-NEXT:   store double 0x4005BF0A8B145769, double* %ptr2, align 8
; CHECK-NEXT:   store double 0x3FF6A09E667F3BCD, double* %ptr3, align 8
; CHECK-NEXT:   store double 0x3FE2788CFC6FB619, double* %ptr4, align 8
; CHECK-NEXT:   store double 0x3FE62E42FEFA39EF, double* %ptr5, align 8
; CHECK-NEXT:   %val1 = load double, double* %ptr1, align 8
; CHECK-NEXT:   %val2 = load double, double* %ptr2, align 8
; CHECK-NEXT:   %val3 = load double, double* %ptr3, align 8
; CHECK-NEXT:   %val4 = load double, double* %ptr4, align 8
; CHECK-NEXT:   %val5 = load double, double* %ptr5, align 8
; CHECK-NEXT:   %[[i0:.+]] = call fast double @llvm.cos.f64(double %val1)
; CHECK-NEXT:   %[[i1:.+]] = call fast double @llvm.sin.f64(double %val2)
; CHECK-NEXT:   %exp_val = call fast double @llvm.exp.f64(double %val3)
; CHECK-NEXT:   %log_val = call fast double @llvm.log.f64(double %val4)
; CHECK-NEXT:   %[[i2:.+]] = call fast double @llvm.cos.f64(double %val1)
; CHECK-NEXT:   %[[i3:.+]] = call fast double @llvm.sin.f64(double %val2)
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %[[i2]], %[[i3]]
; CHECK-NEXT:   store double %[[i0]], double* %ptr1, align 8
; CHECK-NEXT:   store double %[[i1]], double* %ptr2, align 8
; CHECK-NEXT:   store double %exp_val, double* %ptr3, align 8
; CHECK-NEXT:   store double %log_val, double* %ptr4, align 8
; CHECK-NEXT:   store double %[[i4]], double* %ptr5, align 8
; CHECK-NEXT:   ret void