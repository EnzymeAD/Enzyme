; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Reverse-mode counterpart: inverting a GlobalVariable initializer whose
; ConstantStruct holds an inactive GlobalValue or ConstantExpr must produce a
; Constant shadow initializer without crashing getNewFromOriginal or
; cast<Constant>.

@g = external constant i8
@desc_ptrtoint = global { i64, double } { i64 ptrtoint (ptr @g to i64), double 1.0 }
@desc_gv = global { ptr, double } { ptr @g, double 3.0 }

declare void @__enzyme_autodiff(...)

define void @f(ptr %dst1, ptr %dst2) {
entry:
  %v1 = load { i64, double }, ptr @desc_ptrtoint, align 8
  store { i64, double } %v1, ptr %dst1, align 8
  %v2 = load { ptr, double }, ptr @desc_gv, align 8
  store { ptr, double } %v2, ptr %dst2, align 8
  ret void
}

define void @df(ptr %dst1, ptr %ddst1, ptr %dst2, ptr %ddst2) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %dst1, ptr %ddst1, metadata !"enzyme_dup", ptr %dst2, ptr %ddst2)
  ret void
}

; CHECK: @desc_ptrtoint_shadow = global { i64, double } { i64 ptrtoint (ptr @g to i64), double 0.000000e+00 }
; CHECK: @desc_gv_shadow = global { ptr, double } { ptr @g, double 0.000000e+00 }
; CHECK: define internal void @diffef(
