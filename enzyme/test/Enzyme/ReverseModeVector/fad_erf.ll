; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

declare [3 x [2 x double]] @__enzyme_autodiff(...)

declare [2 x double] @Faddeeva_erf([2 x double], double)

define double @test([2 x double] %x) {
entry:
  %f = call [2 x double] @Faddeeva_erf([2 x double] %x, double noundef 0.000000e+00)
  %y = extractvalue [2 x double] %f, 1
  ret double %y
}

define [3 x [2 x double]] @test_derivative([2 x double] %x) {
entry:
  %call = call [3 x [2 x double]] (...) @__enzyme_autodiff(double ([2 x double])* @test, metadata !"enzyme_width", i64 3, [2 x double] %x)
  ret [3 x [2 x double]] %call
}


; CHECK: define internal { [3 x [2 x double]] } @diffe3test([2 x double] %x, [3 x double] %differeturn)