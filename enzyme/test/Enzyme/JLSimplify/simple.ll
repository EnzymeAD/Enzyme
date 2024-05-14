; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare void @julia.safepoint()

declare noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)*, {} addrspace(10)*)

define i1 @preprocess_julia_gelu_act_1883({} addrspace(10)* %a0, {} addrspace(10)* %box) {
top:
  call void @julia.safepoint()
  %a4 = addrspacecast {} addrspace(10)* %a0 to {} addrspace(11)*
  %a14 = call noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 125211990967184 to {}*) to {} addrspace(10)*), {} addrspace(10)* %box) 
  %a15 = addrspacecast {} addrspace(10)* %a14 to {} addrspace(11)*
  %.not165 = icmp eq {} addrspace(11)* %a15, %a4
  ret i1 %.not165
}


; CHECK: define i1 @preprocess_julia_gelu_act_1883({} addrspace(10)* %a0, {} addrspace(10)* %box)
; CHECK-NEXT: top:
; CHECK-NEXT:   call void @julia.safepoint()
; CHECK-NEXT:   %a4 = addrspacecast {} addrspace(10)* %a0 to {} addrspace(11)*
; CHECK-NEXT:   %a14 = call noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 125211990967184 to {}*) to {} addrspace(10)*), {} addrspace(10)* %box)
; CHECK-NEXT:   %a15 = addrspacecast {} addrspace(10)* %a14 to {} addrspace(11)*
; CHECK-NEXT:   %.not165 = icmp eq {} addrspace(11)* %a15, %a4
; CHECK-NEXT:   ret i1 false
; CHECK-NEXT: }
