; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s

; A batch request that precedes a forward-mode request whose target is chosen
; by a select must be lowered exactly once even though splitting the select
; rescans the function and revisits the batch call.

declare [2 x double] @__enzyme_batch(...)
declare double @__enzyme_fwddiff(...)

define double @identity(double %x) {
entry:
  ret double %x
}

define double @zero(double %x) {
entry:
  ret double 0.000000e+00
}

; CHECK-LABEL: define [2 x double] @test(
define [2 x double] @test(i1 %c, double %x, double %y) {
entry:
  ; CHECK: %[[BATCH:.+]] = call [2 x double] @batch_identity(
  ; CHECK-NOT: call [2 x double] @batch_identity(
  %b = call [2 x double] (...) @__enzyme_batch(ptr @identity, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", double %x, double %y)
  %fn = select i1 %c, ptr @identity, ptr @zero
  ; CHECK: br i1 %c, label %sel1, label %sel2
  ; CHECK: ret [2 x double] %[[BATCH]]
  ; CHECK: sel1:
  ; CHECK-NEXT: call fast double @fwddiffeidentity(double %x, double 1.000000e+00)
  ; CHECK: sel2:
  ; CHECK-NEXT: call fast double @fwddiffezero(double %x, double 1.000000e+00)
  %unused = call double (...) @__enzyme_fwddiff(ptr %fn, double %x, double 1.000000e+00)
  ret [2 x double] %b
}

; CHECK-LABEL: define internal [2 x double] @batch_identity(
; CHECK: ret [2 x double]
