; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

declare [2 x double] @cmplx_sqrt([2 x double] %x)

define [2 x double] @tester([2 x double] %x) {
entry:
  %y = call [2 x double] @cmplx_sqrt([2 x double] %x)
  ret [2 x double] %y
}

define [2 x double] @test_derivative([2 x double] %x) {
entry:
  %0 = tail call [2 x double] (...) @__enzyme_autodiff([2 x double] ([2 x double])* nonnull @tester, metadata !"enzyme_active_return", [2 x double] %x)
  ret [2 x double] %0
}

declare [2 x double] @__enzyme_autodiff(...)

; CHECK: define internal { [2 x double] } @diffetester([2 x double] %x, [2 x double] %differeturn)
