; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -jl-inst-simplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s

; ModuleID = 'start'
source_filename = "start"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"
target triple = "arm64-apple-darwin22.4.0"

; Function Attrs: nofree nounwind readnone
declare nonnull {}* @julia.pointer_from_objref({} addrspace(11)*) nofree readnone

define i1 @julia_muladd_5091( { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 } addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(40) %a2) {
top:
  %a10 = call noalias nonnull {} addrspace(10)* @ijl_alloc_array_2d({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4510689616 to {}*) to {} addrspace(10)*), i64 8, i64 10) #30
  %a25 = addrspacecast {} addrspace(10)* %a10 to {} addrspace(11)*
  %a26 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* %a25) #33
  %a27 = bitcast {}* %a26 to i8**
  %arrayptr = load i8*, i8** %a27, align 8
  %getfield_addr = getelementptr inbounds { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 } addrspace(11)* %a2, i64 0, i32 0
  %getfield = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %getfield_addr unordered, align 8
  %a28 = addrspacecast {} addrspace(10)* %getfield to {} addrspace(11)*
  %a29 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* noundef %a28) #33
  %a30 = bitcast {}* %a29 to i8**
  %arrayptr23 = load i8*, i8** %a30, align 8
  %.not = icmp eq i8* %arrayptr, %arrayptr23
  ret i1 %.not
}

declare noalias {} addrspace(10)* @ijl_alloc_array_2d({} addrspace(10)*, i64, i64) "enzyme_allocation"

; CHECK:   ret i1 false
