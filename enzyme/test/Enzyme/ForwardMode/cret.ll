; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

define "enzyme_type"="{[-1]:Float@double}"  [2 x double] @G() {
entry:
  ret [2 x double] [double 0.000000e+00, double 1.000000e+00]
}

declare [2 x double] @__enzyme_fwddiff(...)


define [2 x double] @dsquare() {
entry:
  %0 = call [2 x double] (...) @__enzyme_fwddiff([2 x double] ()*  @G, metadata !"enzyme_dup_return")
  ret [2 x double] %0
}

; CHECK: define internal [2 x double] @fwddiffeG()
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret [2 x double] zeroinitializer
; CHECK-NEXT: }
