; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare { double, double } @csin([2 x double])
declare { double, double } @cpow([2 x double], [2 x double])
declare double @carg([2 x double])

define double @test_packed([2 x double] %a, [2 x double] %b) {
entry:
  %s = call { double, double } @csin([2 x double] %a)
  %s0 = extractvalue { double, double } %s, 0
  %s1 = extractvalue { double, double } %s, 1
  %sa0 = insertvalue [2 x double] undef, double %s0, 0
  %sa1 = insertvalue [2 x double] %sa0, double %s1, 1
  %p = call { double, double } @cpow([2 x double] %sa1, [2 x double] %b)
  %p0 = extractvalue { double, double } %p, 0
  %g = call double @carg([2 x double] %a)
  %res = fadd double %p0, %g
  ret double %res
}

declare { [2 x double], [2 x double] } @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)

define { [2 x double], [2 x double] } @rev_packed([2 x double] %a, [2 x double] %b) {
  %r = call { [2 x double], [2 x double] } (...) @__enzyme_autodiff(double ([2 x double], [2 x double])* nonnull @test_packed, [2 x double] %a, [2 x double] %b)
  ret { [2 x double], [2 x double] } %r
}

define double @fwd_packed([2 x double] %a, [2 x double] %da, [2 x double] %b, [2 x double] %db) {
  %r = call double (...) @__enzyme_fwddiff(double ([2 x double], [2 x double])* nonnull @test_packed, [2 x double] %a, [2 x double] %da, [2 x double] %b, [2 x double] %db)
  ret double %r
}

; CHECK: define internal { [2 x double], [2 x double] } @diffetest_packed([2 x double] %a, [2 x double] %b, double %differeturn)
; CHECK: call { double, double } @cpow([2 x double]
; CHECK: call { double, double } @clog([2 x double]
; CHECK: call { double, double } @ccos([2 x double]
; CHECK: ret { [2 x double], [2 x double] }

; CHECK: define internal double @fwddiffetest_packed([2 x double] %a, [2 x double] %"a'", [2 x double] %b, [2 x double] %"b'")
; CHECK: call { double, double } @ccos([2 x double]
; CHECK: call { double, double } @clog([2 x double]
; CHECK: ret double
