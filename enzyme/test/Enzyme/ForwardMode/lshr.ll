; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

define double @tester(<2 x double> %x) {
entry:
  %x128 = bitcast <2 x double> %x to i128
  %shr = lshr i128 %x128, 64
  %trunc = trunc i128 %shr to i64
  %res = bitcast i64 %trunc to double
  ret double %res
}

define double @test_derivative(<2 x double> %x, <2 x double> %dx) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (<2 x double>)* nonnull @tester, metadata !"enzyme_dup", <2 x double> %x, <2 x double> %dx)
  ret double %0
}

declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffetester(<2 x double> %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x128'ipc" = bitcast <2 x double> %"x'" to i128
; CHECK-NEXT:   %x128 = bitcast <2 x double> %x to i128
; CHECK-NEXT:   %0 = lshr i128 %"x128'ipc", 64
; CHECK-NEXT:   %"trunc'ipc" = trunc i128 %0 to i64
; CHECK-NEXT:   %"res'ipc" = bitcast i64 %"trunc'ipc" to double
; CHECK-NEXT:   ret double %"res'ipc"
; CHECK-NEXT: }
