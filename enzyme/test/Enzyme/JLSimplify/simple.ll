; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare i8 @jl_mightalias({} addrspace(11)*, {} addrspace(11)*)

declare noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)*, {} addrspace(10)*)

define i8 @preprocess_julia_gelu_act_1883({} addrspace(10)* %a0, {} addrspace(10)* %box) {
top:
  %a4 = addrspacecast {} addrspace(10)* %a0 to {} addrspace(11)*
  %a14 = call noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 125211990967184 to {}*) to {} addrspace(10)*), {} addrspace(10)* %box) 
  %a15 = addrspacecast {} addrspace(10)* %a14 to {} addrspace(11)*
  %.not165 = call i8 @jl_mightalias({} addrspace(11)* %a15, {} addrspace(11)* %a4)
  ret i8 %.not165
}


; CHECK: define i8 @preprocess_julia_gelu_act_1883({} addrspace(10)* %a0, {} addrspace(10)* %box) {
; CHECK-NEXT: top:
; CHECK-NEXT:   %a4 = addrspacecast {} addrspace(10)* %a0 to {} addrspace(11)*
; CHECK-NEXT:   %a14 = call noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 125211990967184 to {}*) to {} addrspace(10)*), {} addrspace(10)* %box)
; CHECK-NEXT:   %a15 = addrspacecast {} addrspace(10)* %a14 to {} addrspace(11)*
; CHECK-NEXT:   %.not165 = call i8 @jl_mightalias({} addrspace(11)* %a15, {} addrspace(11)* %a4)
; CHECK-NEXT:   ret i8 0
; CHECK-NEXT: }
