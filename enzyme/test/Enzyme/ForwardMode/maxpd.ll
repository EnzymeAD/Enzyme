; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>)

define dso_local <2 x double> @max(<2 x double> %x, <2 x double> %y) {
entry:
  %res = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %res
}

; Function Attrs: nounwind uwtable
define dso_local <2 x double> @test_derivative(<2 x double> %x, <2 x double> %y) local_unnamed_addr #1 {
entry:
  %0 = tail call <2 x double> (...) @__enzyme_fwddiff(<2 x double> (<2 x double>, <2 x double>)* nonnull @max, <2 x double> %x, <2 x double> %x, <2 x double> %y, <2 x double> %y)
  ret <2 x double> %0
}

; Function Attrs: nounwind
declare <2 x double> @__enzyme_fwddiff(...)

; CHECK: define internal <2 x double> @fwddiffemax(<2 x double> %x, <2 x double> %"x'", <2 x double> %y, <2 x double> %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fcmp{{( fast)?}} olt <2 x double> %x, %y
; CHECK-NEXT:   %1 = select{{( fast)?}} <2 x i1> %0, <2 x double> %"y'", <2 x double> %"x'"
; CHECK-NEXT:   ret <2 x double> %1
; CHECK-NEXT: }
