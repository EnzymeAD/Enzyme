; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false  -enzyme-julia-addr-load -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify)" -S -opaque-pointers | FileCheck %s; fi

; Function Attrs: nofree nosync nounwind willreturn memory(none)
declare ptr @julia.get_pgcstack() local_unnamed_addr #0

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite)
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10)) local_unnamed_addr #1

; Function Attrs: nofree norecurse nosync nounwind speculatable willreturn memory(argmem: read)
declare noundef nonnull ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) nocapture noundef nonnull readnone, ptr noundef nonnull readnone) local_unnamed_addr #2

declare void @__enzyme_autodiff(...) local_unnamed_addr

define void @dsquare(double %arg) local_unnamed_addr {
bb:
  tail call void (...) @__enzyme_autodiff(ptr @julia_sparse_eval_1010, metadata !"enzyme_dup", ptr addrspace(10) undef, ptr addrspace(10) undef, metadata !"enzyme_dup", ptr addrspace(10) undef, ptr addrspace(10) undef)
  ret void
}

define "enzyme_type"="{[-1]:Float@double}" double @julia_sparse_eval_1010(ptr addrspace(10) noundef nonnull align 8 dereferenceable(24) %arg, ptr addrspace(10) noundef nonnull align 8 dereferenceable(24) "enzyme_inactive" %arg1) local_unnamed_addr #6 {
bb:
  
  %i10 = addrspacecast ptr addrspace(10) %arg to ptr addrspace(11)
  %i208 = load ptr, ptr addrspace(11) %i10, align 8, !enzyme_type !16, !enzymejl_byref_BITS_VALUE !19, !enzymejl_source_type_Ptr\7BFloat64\7D !19, !enzyme_nocache !19 
  
  %i194 = addrspacecast ptr addrspace(10) %arg1 to ptr addrspace(11)
  %i195 = getelementptr inbounds { i64, ptr }, ptr addrspace(11) %i194, i64 0, i32 1
  %i196 = load ptr, ptr addrspace(11) %i195, align 8, !tbaa !37, !alias.scope !39, !noalias !40, !nonnull !19, !enzyme_type !16, !enzymejl_byref_BITS_VALUE !19, !enzymejl_source_type_Ptr\7BFloat64\7D !19
  
  %i309 = call fastcc double @perm(ptr %i208, ptr addrspace(10) noundef nonnull align 8 dereferenceable(24) %arg1, ptr %i196)
  store double %i309, ptr %i208, align 8, !tbaa !25, !alias.scope !28, !noalias !42
  
  %i6 = addrspacecast ptr addrspace(10) %arg to ptr addrspace(11)
  %i12 = load ptr, ptr addrspace(11) %i6, align 8, !tbaa !3, !alias.scope !8, !noalias !11, !enzyme_type !16, !enzymejl_byref_BITS_VALUE !19, !enzymejl_source_type_Ptr\7BFloat64\7D !19, !enzyme_nocache !19
  %i33 = load double, ptr %i12, align 8, !tbaa !25, !alias.scope !28, !noalias !29, !enzyme_type !30, !enzymejl_byref_BITS_VALUE !19, !enzymejl_source_type_Float64 !19
  ret double %i33
}

define noalias ptr addrspace(10) @fakecopy(ptr addrspace(10) %arg) {
bb:
  ret ptr addrspace(10) %arg
}

define internal fastcc double @perm(ptr %i208, ptr addrspace(10) noalias noundef nonnull align 8 dereferenceable(24) %arg1, ptr %i196) unnamed_addr #8 {
bb:
  %i197 = call noalias nonnull align 8 dereferenceable(24) ptr addrspace(10) @julia.gc_alloc_obj(ptr nonnull undef, i64 noundef 24, ptr addrspace(10) noundef addrspacecast (ptr inttoptr (i64 127227706936608 to ptr) to ptr addrspace(10))) #11
   
  %i193 = call ptr addrspace(10) @fakecopy(ptr addrspace(10) %arg1)
   
  store ptr addrspace(10) %i193, ptr addrspace(10) %i197, align 8, !tbaa !3, !alias.scope !8, !noalias !41

  %i223 = load ptr addrspace(10), ptr addrspace(10) %i197, align 8, !tbaa !3, !alias.scope !8, !noalias !11, !dereferenceable_or_null !20, !align !21, !enzyme_type !22, !enzymejl_byref_MUT_REF !19, !enzymejl_source_type_Memory\7BFloat64\7D !19
  
  %i307 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) noundef %i223, ptr noundef %i196)
  %i309 = load double, ptr addrspace(13) %i307, align 8, !tbaa !25, !alias.scope !28, !noalias !29
  ret double %i309
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #9

attributes #0 = { nofree nosync nounwind willreturn memory(none) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzyme_shouldrecompute" "enzymejl_world"="26724" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" "enzyme_no_escaping_allocation" "enzymejl_world"="26724" }
attributes #2 = { nofree norecurse nosync nounwind speculatable willreturn memory(argmem: read) "enzyme_nocache" "enzyme_shouldrecompute" "enzymejl_world"="26724" }
attributes #3 = { nofree norecurse nounwind memory(none) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzyme_nocache" "enzyme_shouldrecompute" "enzymejl_world"="26724" }
attributes #4 = { nofree nounwind memory(none) "enzyme_no_escaping_allocation" "enzymejl_world"="26724" }
attributes #5 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" "enzyme_no_escaping_allocation" "enzymejl_world"="26724" }
attributes #6 = { "enzyme_ta_norecur" "enzymejl_mi"="127227877936720" "enzymejl_rt"="127227761848288" "enzymejl_world"="26724" "julia.fsig"="sparse_eval(Array{Float64, 1}, Array{Int64, 1})" }
attributes #7 = { mustprogress nofree nounwind willreturn memory(argmem: read, inaccessiblemem: readwrite) "enzyme_LocalReadOnlyOrThrow" "enzyme_no_escaping_allocation" }
attributes #8 = { "enzyme_retremove" "enzyme_ta_norecur" "enzymejl_mi"="127227877939024" "enzymejl_rt"="127227706936608" "enzymejl_world"="26724" "julia.fsig"="permute!(Array{Float64, 1}, Array{Int64, 1})" }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #10 = { nounwind memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #11 = { nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #12 = { nounwind memory(readwrite) "enzyme_no_escaping_allocation" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"stack-protector-guard", !"global"}
!3 = !{!4, !4, i64 0}
!4 = !{!"jtbaa_arrayptr", !5, i64 0}
!5 = !{!"jtbaa_array", !6, i64 0}
!6 = !{!"jtbaa", !7, i64 0}
!7 = !{!"jtbaa"}
!8 = !{!9}
!9 = !{!"jnoalias_typemd", !10}
!10 = !{!"jnoalias"}
!11 = !{!12, !13, !14, !15}
!12 = !{!"jnoalias_gcframe", !10}
!13 = !{!"jnoalias_stack", !10}
!14 = !{!"jnoalias_data", !10}
!15 = !{!"jnoalias_const", !10}
!16 = !{!"Unknown", i32 -1, !17}
!17 = !{!"Pointer", i32 -1, !18}
!18 = !{!"Float@double"}
!19 = !{}
!20 = !{i64 16}
!21 = !{i64 8}
!22 = !{!"Unknown", i32 -1, !23}
!23 = !{!"Pointer", i32 0, !24, i32 1, !24, i32 2, !24, i32 3, !24, i32 4, !24, i32 5, !24, i32 6, !24, i32 7, !24, i32 8, !17}
!24 = !{!"Integer"}
!25 = !{!26, !26, i64 0}
!26 = !{!"jtbaa_arraybuf", !27, i64 0}
!27 = !{!"jtbaa_data", !6, i64 0}
!28 = !{!14}
!29 = !{!12, !13, !9, !15}
!30 = !{!"Unknown", i32 -1, !18}
!31 = !{!6, !6, i64 0}
!32 = !{!9, !13}
!33 = !{!34, !12, !14, !15}
!34 = distinct !{!34, !35, !"na_addr13"}
!35 = distinct !{!35, !"addr13"}
!36 = !{!"Unknown", i32 -1, !24}
!37 = !{!38, !38, i64 0, i64 0}
!38 = !{!"jtbaa_const", !6, i64 0}
!39 = !{!15}
!40 = !{!12, !13, !14, !9}
!41 = !{!34, !12, !13, !14, !15}
!42 = !{!34, !12, !13, !9, !15}

; CHECK: define internal fastcc { ptr addrspace(10), double } @augmented_perm(ptr nocapture readnone %i208, ptr nocapture readnone %"i208'", ptr addrspace(10) noalias noundef nonnull align 8 dereferenceable(24) %arg1, ptr addrspace(10) align 8 %"arg1'", ptr nocapture readonly %i196, ptr nocapture %"i196'")
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = alloca { ptr addrspace(10), double }, align 8
; CHECK-NEXT:   %"i197'mi" = call noalias nonnull align 8 dereferenceable(24) ptr addrspace(10) @julia.gc_alloc_obj(ptr nonnull undef, i64 noundef 24, ptr addrspace(10) noundef addrspacecast (ptr inttoptr (i64 127227706936608 to ptr) to ptr addrspace(10)))
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull dereferenceable(24) dereferenceable_or_null(24) %"i197'mi", i8 0, i64 24, i1 false)
; CHECK-NEXT:   %i197 = call noalias nonnull align 8 dereferenceable(24) ptr addrspace(10) @julia.gc_alloc_obj(ptr nonnull undef, i64 noundef 24, ptr addrspace(10) noundef addrspacecast (ptr inttoptr (i64 127227706936608 to ptr) to ptr addrspace(10)))
; CHECK-NEXT:   %i193_augmented = call { ptr addrspace(10), ptr addrspace(10) } @augmented_fakecopy(ptr addrspace(10) %arg1, ptr addrspace(10) %"arg1'")
; CHECK-NEXT:   %i193 = extractvalue { ptr addrspace(10), ptr addrspace(10) } %i193_augmented, 0
; CHECK-NEXT:   %"i193'ac" = extractvalue { ptr addrspace(10), ptr addrspace(10) } %i193_augmented, 1
; CHECK-NEXT:   store ptr addrspace(10) %"i193'ac", ptr %0, align 8
; CHECK-NEXT:   store ptr addrspace(10) %"i193'ac", ptr addrspace(10) %"i197'mi", align 8
; CHECK-NEXT:   store ptr addrspace(10) %i193, ptr addrspace(10) %i197, align 8
; CHECK-NEXT:   %i223 = load ptr addrspace(10), ptr addrspace(10) %i197, align 8
; CHECK-NEXT:   %i307 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) noundef %i223, ptr noundef %i196)
; CHECK-NEXT:   %i309 = load double, ptr addrspace(13) %i307, align 8
; CHECK-NEXT:   %1 = getelementptr inbounds { ptr addrspace(10), double }, ptr %0, i32 0, i32 1
; CHECK-NEXT:   store double %i309, ptr %1, align 8
; CHECK-NEXT:   %2 = load { ptr addrspace(10), double }, ptr %0, align 8
; CHECK-NEXT:   ret { ptr addrspace(10), double } %2
; CHECK-NEXT: }

; CHECK: define internal fastcc void @diffeperm(ptr nocapture readnone %i208, ptr nocapture readnone %"i208'", ptr addrspace(10) noalias align 8 dereferenceable(24) %arg1, ptr addrspace(10) align 8 %"arg1'", ptr nocapture readonly %i196, ptr nocapture %"i196'", double %differeturn, ptr addrspace(10) %"i193'ip_phi")
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"i197'mi" = call noalias nonnull align 8 dereferenceable(24) ptr addrspace(10) @julia.gc_alloc_obj(ptr nonnull undef, i64 noundef 24, ptr addrspace(10) noundef addrspacecast (ptr inttoptr (i64 127227706936608 to ptr) to ptr addrspace(10)))
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull dereferenceable(24) dereferenceable_or_null(24) %"i197'mi", i8 0, i64 24, i1 false)
; CHECK-NEXT:   store ptr addrspace(10) %"i193'ip_phi", ptr addrspace(10) %"i197'mi", align 8
; CHECK-NEXT:   %"i223'ipl" = load ptr addrspace(10), ptr addrspace(10) %"i197'mi", align 8
; CHECK-NEXT:   %0 = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %"i223'ipl", ptr %"i196'")
; CHECK-NEXT:   %1 = load double, ptr addrspace(13) %0, align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   store double %2, ptr addrspace(13) %0, align 8
; CHECK-NEXT:   call void @diffefakecopy(ptr addrspace(10) %arg1, ptr addrspace(10) undef)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

