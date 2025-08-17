; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define double @tester(double %x0) #0 {
entry:
  %x2 = alloca double, align 8
  %x3 = alloca x86_fp80, align 16
  store double %x0, double* %x2, align 8
  %x4 = load double, double* %x2, align 8
  %x5 = fpext double %x4 to x86_fp80
  store x86_fp80 %x5, x86_fp80* %x3, align 16
  %x6 = load x86_fp80, x86_fp80* %x3, align 16
  %x7 = fptrunc x86_fp80 %x6 to double
  ret double %x7
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, double %x, double 0.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double)*, ...)

; CHECK: define internal double @fwddiffetester(double %x0, double %"x0'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x5'ipc" = fpext double %"x0'" to x86_fp80
; CHECK-NEXT:   %"x7'ipc" = fptrunc x86_fp80 %"x5'ipc" to double
; CHECK-NEXT:   ret double %"x7'ipc"
; CHECK-NEXT: }

