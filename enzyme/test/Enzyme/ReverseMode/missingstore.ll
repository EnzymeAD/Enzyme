; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

source_filename = "text"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-pc-linux-gnu"

declare void @__enzyme_reverse(...)

define void @dsquare(double %x) {
entry:
  call void (...) @__enzyme_reverse({} addrspace(10)* ({ [1 x {} addrspace(10)*] } addrspace(11)*)* @julia__foldl_impl_3869, metadata !"enzyme_dup", metadata !"enzyme_not_overwritten", { [1 x {} addrspace(10)*] } addrspace(11)* null, { [1 x {} addrspace(10)*] } addrspace(11)* null, i8* null)
  ret void
}

define internal fastcc nonnull {} addrspace(10)* @julia__foldl_impl_3869({ [1 x {} addrspace(10)*] } addrspace(11)* nocapture nonnull readonly align 8 dereferenceable(8) %arg)  {
top:
  %i11 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj(i8* null, i64 8)
  %i12 = bitcast { [1 x {} addrspace(10)*] } addrspace(11)* %arg to i64 addrspace(11)*
  %i13 = bitcast {} addrspace(10)* %i11 to i64 addrspace(10)*
  %i14 = load i64, i64 addrspace(11)* %i12, align 8
  store i64 %i14, i64 addrspace(10)* %i13, align 8
  call void @jl_invoke({} addrspace(10)* nonnull %i11) argmemonly
  ret {} addrspace(10)* null
}

; Function Attrs: nofree
define void @jl_invoke({} addrspace(10)* nocapture readonly %y) {
bb:
  ret void
}

; Function Attrs: inaccessiblememonly allocsize(1)
declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj(i8*, i64)

; CHECK: define internal fastcc void @diffejulia__foldl_impl_3869({ [1 x {} addrspace(10)*] } addrspace(11)* nocapture readonly align 8 dereferenceable(8) %arg, { [1 x {} addrspace(10)*] } addrspace(11)* nocapture align 8 %"arg'", i8* %tapeArg)
; CHECK-NEXT: top:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %"i11'mi" = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj(i8* null, i64 8) 
; CHECK-NEXT:   %[[i1:.+]] = bitcast {} addrspace(10)* %"i11'mi" to i8 addrspace(10)*
; CHECK-NEXT:   call void @llvm.memset.p10i8.i64(i8 addrspace(10)* nonnull dereferenceable(8) dereferenceable_or_null(8) %[[i1]], i8 0, i64 8, i1 false)
; CHECK-NEXT:   %i11 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj(i8* null, i64 8) 
; CHECK-NEXT:   %"i12'ipc" = bitcast { [1 x {} addrspace(10)*] } addrspace(11)* %"arg'" to i64 addrspace(11)*
; CHECK-NEXT:   %i12 = bitcast { [1 x {} addrspace(10)*] } addrspace(11)* %arg to i64 addrspace(11)*
;; NOTE FOR ABOVE, the PRIMAL MUST COME AFTER THE SHADOW. This is because between the allocations there is otherwise undefined memory
;; that gc-alloc will error upon. But if the shadow comes first, it is at least null'd
; CHECK-NEXT:   %"i13'ipc" = bitcast {} addrspace(10)* %"i11'mi" to i64 addrspace(10)*
; CHECK-NEXT:   %i13 = bitcast {} addrspace(10)* %i11 to i64 addrspace(10)*
; CHECK-NEXT:   %"i14'ipl" = load i64, i64 addrspace(11)* %"i12'ipc"
; CHECK-NEXT:   %i14 = load i64, i64 addrspace(11)* %i12
; CHECK-NEXT:   store i64 %"i14'ipl", i64 addrspace(10)* %"i13'ipc", align 8
; THE CRITICAL PART OF THIS TEST IS ENSURING THIS STORE EXISTS
; CHECK-NEXT:   store i64 %i14, i64 addrspace(10)* %i13, align 8
; CHECK-NEXT:   call void @diffejl_invoke({} addrspace(10)* %i11, {} addrspace(10)* %"i11'mi")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }