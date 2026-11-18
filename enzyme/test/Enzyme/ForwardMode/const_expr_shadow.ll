; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Inverting a ConstantStruct initializer containing an inactive GlobalValue or
; ConstantExpr (such as flang's derived-type descriptors holding ptrtoint of
; globals) must keep the shadow as a Constant rather than falling into the
; runtime alloca/store/load path (which also crashed getNewFromOriginal).

@g = external constant i8
@desc_ptrtoint = global { i64, double } { i64 ptrtoint (ptr @g to i64), double 1.0 }
@desc_xor = global { i64, double } { i64 xor (i64 ptrtoint (ptr @g to i64), i64 1), double 2.0 }
@desc_gv = global { ptr, double } { ptr @g, double 3.0 }

declare void @__enzyme_fwddiff(...)

define void @f(ptr %dst1, ptr %dst2, ptr %dst3) {
entry:
  %v1 = load { i64, double }, ptr @desc_ptrtoint, align 8
  store { i64, double } %v1, ptr %dst1, align 8
  %v2 = load { i64, double }, ptr @desc_xor, align 8
  store { i64, double } %v2, ptr %dst2, align 8
  %v3 = load { ptr, double }, ptr @desc_gv, align 8
  store { ptr, double } %v3, ptr %dst3, align 8
  ret void
}

define void @df(ptr %dst1, ptr %ddst1, ptr %dst2, ptr %ddst2, ptr %dst3, ptr %ddst3) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_dup", ptr %dst1, ptr %ddst1, metadata !"enzyme_dup", ptr %dst2, ptr %ddst2, metadata !"enzyme_dup", ptr %dst3, ptr %ddst3)
  ret void
}

; CHECK: @desc_ptrtoint_shadow = global { i64, double } { i64 ptrtoint (ptr @g to i64), double 0.000000e+00 }
; CHECK: @desc_xor_shadow = global { i64, double } { i64 xor (i64 ptrtoint (ptr @g to i64), i64 1), double 0.000000e+00 }
; CHECK: @desc_gv_shadow = global { ptr, double } { ptr @g, double 0.000000e+00 }
; CHECK: define internal void @fwddiffef(
; CHECK:   load { i64, double }, ptr @desc_ptrtoint_shadow
; CHECK:   load { i64, double }, ptr @desc_xor_shadow
; CHECK:   load { ptr, double }, ptr @desc_gv_shadow
