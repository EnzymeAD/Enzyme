; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare void @julia.safepoint()

declare noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)*, {} addrspace(10)*)

; this came up as a bound to a loop in a broadcast enforcing really bad perf
; since we could no longer determine loop bound
define i64 @preprocess_julia_gelu_act_1883(i1 %c) {
top:
  %c2 = freeze i1 %c
  br i1 %c2, label %r1, label %r2

r1:
  ret i64 13

r2:
  ret i64 49
}

; CHECK: define i64 @preprocess_julia_gelu_act_1883(i1 %c) 
; CHECK-NEXT: top:
; CHECK-NEXT:   %c2 = freeze i1 %c
; CHECK-NEXT:   br i1 %c, label %r1, label %r2

; CHECK: r1:                                               ; preds = %top
; CHECK-NEXT:   ret i64 13

; CHECK: r2:                                               ; preds = %top
; CHECK-NEXT:   ret i64 49
; CHECK-NEXT: }
