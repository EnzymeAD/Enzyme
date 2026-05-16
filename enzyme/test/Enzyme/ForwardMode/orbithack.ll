; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

define double @tester(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %normx = or <2 x i64> %cstx, <i64 4602678819172646912, i64 4602678819172646912>
  %csty = bitcast <2 x i64> %normx to <2 x double>
  %res = extractelement <2 x double> %csty, i32 0
  ret double %res
}

define double @test_derivative(<2 x double> %x, <2 x double> %dx) {
entry:
  %0 = tail call double (double (<2 x double>)*, ...) @__enzyme_fwddiff(double (<2 x double>)* nonnull @tester, <2 x double> %x, <2 x double> %dx)
  ret double %0
}

declare double @__enzyme_fwddiff(double (<2 x double>)*, ...)

; CHECK: define internal double @fwddiffetester(<2 x double> %x, <2 x double> %"x'")
; CHECK: or <2 x i64>
; CHECK: ret double
