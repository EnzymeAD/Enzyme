; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme-summary" -enzyme-summary-out=%t.json -disable-output; FileCheck %s < %t.json; fi

; enzyme-summary: the facts a thin-link step needs to plan separate
; compilation, written as JSON.

@enzyme_strong_zero = external global i32
@__enzyme_inactivefn_log_msg = global ptr @log_msg
@__enzyme_register_gradient_sq = global { ptr, ptr, ptr } { ptr @sq, ptr @aug_sq, ptr @rev_sq }
@common_blk_ = common global [16 x i8] zeroinitializer

declare void @log_msg(ptr)
declare void @aug_sq(ptr)
declare void @rev_sq(ptr)
declare void @ext(ptr)
declare void @__enzyme_autodiff(...)

define void @sq(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %x) {
  call void @log_msg(ptr %x)
  %v = load double, ptr %x
  %m = fmul double %v, %v
  store double %m, ptr @common_blk_
  call void @ext(ptr %x)
  ret void
}

define void @caller(ptr %x, ptr %dx) {
  call void (...) @__enzyme_autodiff(ptr @sq, ptr @enzyme_strong_zero, ptr %x, ptr %dx)
  ret void
}

; CHECK-DAG: "fn": "sq"
; CHECK-DAG: "mode": "reverse"
; CHECK-DAG: "strong_zero": true
; CHECK-DAG: "inactive": [
; CHECK-DAG: "custom_rule": [
; CHECK-DAG: "common_blk_": {
; CHECK-DAG: "linkage": "common"
; CHECK-DAG: "size": 16
; CHECK-DAG: "{[-1]:Pointer, [-1,0]:Float@double}"
