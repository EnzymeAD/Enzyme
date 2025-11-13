; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-julia-addr-load -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -S -opaque-pointers | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false  -enzyme-julia-addr-load -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify)" -S -opaque-pointers | FileCheck %s


define ptr addrspace(10) @retv(double %arg, ptr addrspace(10) %arg1) {
bb:
  ret ptr addrspace(10) %arg1
}

; Function Attrs: nofree norecurse nosync nounwind speculatable willreturn memory(argmem: read)
declare noundef nonnull ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) nocapture noundef nonnull readnone, ptr noundef nonnull readnone) local_unnamed_addr #1

define  double @ifn(ptr addrspace(11) %arg, ptr addrspace(10) %arg1) {
bb:
  %i7 = load ptr addrspace(10), ptr addrspace(11) %arg, align 8
  %i8 = addrspacecast ptr addrspace(10) %i7 to ptr addrspace(11)
  %i31 = load ptr, ptr addrspace(11) %i8, align 8, !enzyme_nocache !14

  %i32 = getelementptr inbounds { ptr, ptr addrspace(10) }, ptr addrspace(11) %i8, i64 0, i32 1
  %i33 = load ptr addrspace(10), ptr addrspace(11) %i32, align 8

  %i50 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) noundef %i33, ptr noundef %i31)

  %i51 = load double, ptr addrspace(13) %i50, align 8

  %i52 = call ptr addrspace(10) @retv(double %i51, ptr addrspace(10) %arg1)

  ret double %i51
}

declare void @__enzyme_autodiff(...)

define void @dsquare(double %arg) {
bb:
  tail call void (...) @__enzyme_autodiff(ptr @ifn, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr addrspace(11) undef, ptr addrspace(11) undef, metadata !"enzyme_dup", ptr addrspace(10) undef, ptr addrspace(10) undef)
  ret void
}

attributes #1 = { nofree norecurse nosync nounwind speculatable willreturn memory(argmem: read) "enzyme_nocache" "enzyme_shouldrecompute" "enzymejl_world"="26726" }

!14 = !{}

; CHECK: define internal void @diffeifn(ptr addrspace(11) %arg, ptr addrspace(11) %"arg'", ptr addrspace(10) %arg1, ptr addrspace(10) %"arg1'", double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"i7'ipl" = load ptr addrspace(10), ptr addrspace(11) %"arg'", align 8
; CHECK-NEXT:   %i7 = load ptr addrspace(10), ptr addrspace(11) %arg, align 8
; CHECK-NEXT:   %"i8'ipc" = addrspacecast ptr addrspace(10) %"i7'ipl" to ptr addrspace(11)
; CHECK-NEXT:   %i8 = addrspacecast ptr addrspace(10) %i7 to ptr addrspace(11)
; CHECK-NEXT:   %"i31'ipl" = load ptr, ptr addrspace(11) %"i8'ipc", align 8
; CHECK-NEXT:   %i31 = load ptr, ptr addrspace(11) %i8, align 8
; CHECK-NEXT:   %"i32'ipg" = getelementptr inbounds { ptr, ptr addrspace(10) }, ptr addrspace(11) %"i8'ipc", i64 0, i32 1
; CHECK-NEXT:   %i32 = getelementptr inbounds { ptr, ptr addrspace(10) }, ptr addrspace(11) %i8, i64 0, i32 1
; CHECK-NEXT:   %"i33'ipl" = load ptr addrspace(10), ptr addrspace(11) %"i32'ipg", align 8
; CHECK-NEXT:   %i33 = load ptr addrspace(10), ptr addrspace(11) %i32, align 8
; CHECK-NEXT:   %0 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %"i33'ipl", ptr %"i31'ipl")
; CHECK-NEXT:   %i50 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) noundef %i33, ptr noundef %i31)
; CHECK-NEXT:   %i51 = load double, ptr addrspace(13) %i50, align 8
; CHECK-NEXT:   %i52 = call ptr addrspace(10) @retv(double %i51, ptr addrspace(10) %arg1)
; CHECK-NEXT:   %1 = icmp ne ptr addrspace(13) %i50, %0
; CHECK-NEXT:   br i1 %1, label %invertbb_active, label %invertbb_amerge

; CHECK: invertbb_active:                                  ; preds = %bb
; CHECK-NEXT:   %2 = load double, ptr addrspace(13) %0, align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %differeturn
; CHECK-NEXT:   store double %3, ptr addrspace(13) %0, align 8
; CHECK-NEXT:   br label %invertbb_amerge

; CHECK: invertbb_amerge:                                  ; preds = %invertbb_active, %bb
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

