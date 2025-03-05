; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

declare void @julia.safepoint()

declare noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)*, {} addrspace(10)*) "enzyme_allocation"

declare {}* @julia.pointer_from_objref({} addrspace(11)*) readnone

define i1 @preprocess_julia_gelu_act_1883({} addrspace(10)* %a0, {} addrspace(10)* %box) {
top:
  call void @julia.safepoint()
  %a4 = addrspacecast {} addrspace(10)* %a0 to {} addrspace(11)*
  %a14 = call noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 125211990967184 to {}*) to {} addrspace(10)*), {} addrspace(10)* %box) 
  %a15 = addrspacecast {} addrspace(10)* %a14 to {} addrspace(11)*

  %a32 = call noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 125211990967184 to {}*) to {} addrspace(10)*), {} addrspace(10)* noundef nonnull %box)
  %a33 = addrspacecast {} addrspace(10)* %a32 to {} addrspace(11)*

  %a40 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* %a33) 
  %a41 = bitcast {}* %a40 to i8**
  %arrayptr113 = load i8*, i8** %a41, align 8
  %a42 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* %a15)
  %a43 = bitcast {}* %a42 to i8**
  %arrayptr115 = load i8*, i8** %a43, align 8
  %.not170 = icmp eq i8* %arrayptr113, %arrayptr115
  ret i1 %.not170
}

; CHECK: define i1 @preprocess_julia_gelu_act_1883({} addrspace(10)* %a0, {} addrspace(10)* %box) 
; CHECK-NEXT: top:
; CHECK-NEXT:   call void @julia.safepoint()
; CHECK-NEXT:   %a4 = addrspacecast {} addrspace(10)* %a0 to {} addrspace(11)*
; CHECK-NEXT:   %a14 = call noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 125211990967184 to {}*) to {} addrspace(10)*), {} addrspace(10)* %box)
; CHECK-NEXT:   %a15 = addrspacecast {} addrspace(10)* %a14 to {} addrspace(11)*
; CHECK-NEXT:   %a32 = call noalias nonnull {} addrspace(10)* @ijl_new_array({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 125211990967184 to {}*) to {} addrspace(10)*), {} addrspace(10)* noundef nonnull %box)
; CHECK-NEXT:   %a33 = addrspacecast {} addrspace(10)* %a32 to {} addrspace(11)*
; CHECK-NEXT:   %a40 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* %a33)
; CHECK-NEXT:   %a41 = bitcast {}* %a40 to i8**
; CHECK-NEXT:   %arrayptr113 = load i8*, i8** %a41, align 8
; CHECK-NEXT:   %a42 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* %a15)
; CHECK-NEXT:   %a43 = bitcast {}* %a42 to i8**
; CHECK-NEXT:   %arrayptr115 = load i8*, i8** %a43, align 8
; CHECK-NEXT:   %.not170 = icmp eq i8* %arrayptr113, %arrayptr115
; CHECK-NEXT:   ret i1 false
; CHECK-NEXT: }
