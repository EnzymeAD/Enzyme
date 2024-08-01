; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify)" -S | FileCheck %s

source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

@ejl_jl_nothing = external addrspace(10) global {}, !enzyme_inactive !0, !enzyme_ta_norecur !0

declare void @__enzyme_reverse(...) local_unnamed_addr

define void @dsquare(double %arg) local_unnamed_addr {
bb:
  tail call void (...) @__enzyme_reverse(void ({} addrspace(10)*)* @wat, metadata !"enzyme_dup", {} addrspace(10)* undef, {} addrspace(10)* undef, i8* null)
  ret void
}

define internal fastcc void @setter({} addrspace(10)** noalias nocapture noundef nonnull writeonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" %arg, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40)  %arg2) {
bb:
  store {} addrspace(10)* @ejl_jl_nothing, {} addrspace(10)** %arg, align 8
  ret void
}

define internal void @inner({} addrspace(10)** nocapture noundef nonnull readonly align 8 dereferenceable(40) %arg3) {
bb:
  ret void
}

define internal void @wat({} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %arg2) {
bb:
  %stacked = alloca {} addrspace(10)*, align 8
  %outp = alloca {} addrspace(10)*, align 8

  call fastcc void @setter({} addrspace(10)** noalias nocapture noundef nonnull align 8 dereferenceable(24) %stacked, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %arg2)

  %getvalue = load {} addrspace(10)*, {} addrspace(10)** %stacked , align 8

  store {} addrspace(10)* %getvalue, {} addrspace(10)** %outp, align 8;, !noalias !51
  call void @inner({} addrspace(10)** nocapture noundef nonnull readonly align 8 dereferenceable(40) %outp)
  ret void
}

!0 = !{}

; CHECK: define internal void @diffewat({} addrspace(10)* align 16 dereferenceable(40) %arg2, {} addrspace(10)* align 16 %"arg2'", i8* %tapeArg) 
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { i8*, i8* }*
; CHECK-NEXT:   %truetape = load { i8*, i8* }, { i8*, i8* }* %0, align 8
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %outp = alloca {} addrspace(10)*, i64 1, align 8
; CHECK-NEXT:   %"malloccall'mi" = extractvalue { i8*, i8* } %truetape, 0
; CHECK-NEXT:   %malloccall = extractvalue { i8*, i8* } %truetape, 1
; CHECK-NEXT:   %"stacked'ipc" = bitcast i8* %"malloccall'mi" to {} addrspace(10)**
; CHECK-NEXT:   %stacked = bitcast i8* %malloccall to {} addrspace(10)**
; CHECK-NEXT:   %"outp'ai" = alloca {} addrspace(10)*, i64 1, align 8
; CHECK-NEXT:   %1 = bitcast {} addrspace(10)** %"outp'ai" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %1, i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"getvalue'ipl" = load {} addrspace(10)*, {} addrspace(10)** %"stacked'ipc", align 8
; CHECK-NEXT:   %getvalue = load {} addrspace(10)*, {} addrspace(10)** %stacked, align 8
; CHECK-NEXT:   store {} addrspace(10)* %"getvalue'ipl", {} addrspace(10)** %"outp'ai", align 8
; CHECK-NEXT:   store {} addrspace(10)* %getvalue, {} addrspace(10)** %outp, align 8
; CHECK-NEXT:   call void @diffeinner({} addrspace(10)** nocapture readonly align 8 %outp, {} addrspace(10)** nocapture align 8 %"outp'ai")
; CHECK-NEXT:   call fastcc void @diffesetter({} addrspace(10)** nocapture align 8 undef, {} addrspace(10)** nocapture align 8 undef, {} addrspace(10)* align 16 %arg2, {} addrspace(10)* align 16 %"arg2'")
; CHECK-NEXT:   call void @free(i8* nonnull %"malloccall'mi")
; CHECK-NEXT:   call void @free(i8* %malloccall)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
