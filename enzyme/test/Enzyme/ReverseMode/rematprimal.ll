; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite)
declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}*, i64, {} addrspace(10)*) local_unnamed_addr #0

declare void @__enzyme_reverse(...) local_unnamed_addr

define void @dsquare({} addrspace(10)* %arg, {} addrspace(10)* %arg1, {} addrspace(10)* %arg2, {} addrspace(10)* %arg3) local_unnamed_addr {
bb:
  call void (...) @__enzyme_reverse(void ({} addrspace(10)*, {} addrspace(10)*, i1)* nonnull @julia_set_params_870, metadata !"enzyme_dup", {} addrspace(10)* %arg, {} addrspace(10)* %arg1, metadata !"enzyme_dup", {} addrspace(10)* %arg2, {} addrspace(10)* %arg3, i1 false, i8* null)
  ret void
}

define void @julia_set_params_870({} addrspace(10)* nocapture noundef nonnull readonly align 8 %arg, {} addrspace(10)* noundef nonnull align 8 dereferenceable(24) %arg1, i1 %arg2) {
bb:
  %i90 = call nonnull {} addrspace(10)* @f({} addrspace(10)* noundef %arg1)
  br label %bb3

bb3:                                              ; preds = %bb12, %bb
  %i = phi i64 [ 0, %bb ], [ %i4, %bb12 ]
  %i4 = add i64 %i, 1
  %i5 = icmp eq i64 %i, 10
  br i1 %arg2, label %bb6, label %bb12

bb6:                                              ; preds = %bb3
  %i8 = call noalias nonnull align 8 dereferenceable(24) {} addrspace(10)* @julia.gc_alloc_obj({}* null, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 137917234968496 to {}*) to {} addrspace(10)*)) #4
  %i9 = bitcast {} addrspace(10)* %i8 to {} addrspace(10)* addrspace(10)*
  store {} addrspace(10)* %i90, {} addrspace(10)* addrspace(10)* %i9, align 8, !tbaa !3, !alias.scope !8, !noalias !11
  br label %bb12

bb12:                                             ; preds = %bb6, %bb3
  %i13 = phi {} addrspace(10)* [ %i8, %bb6 ], [ %arg1, %bb3 ]
  %i14 = addrspacecast {} addrspace(10)* %i13 to {} addrspace(11)*
  %i15 = bitcast {} addrspace(11)* %i14 to double addrspace(13)* addrspace(11)*
  %i16 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %i15, align 8, !tbaa !3, !alias.scope !8, !noalias !18, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22
  %i17 = bitcast {} addrspace(10)* %arg to double addrspace(13)* addrspace(10)*
  %i18 = load double addrspace(13)*, double addrspace(13)* addrspace(10)* %i17, align 8, !tbaa !3, !alias.scope !8, !noalias !18, !enzyme_nocache !22
  %i19 = load double, double addrspace(13)* %i16, align 8, !tbaa !23, !alias.scope !26, !noalias !27
  store double %i19, double addrspace(13)* %i18, align 8, !tbaa !23, !alias.scope !26, !noalias !28
  br i1 %i5, label %bb20, label %bb3

bb20:                                             ; preds = %bb12
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read, inaccessiblemem: readwrite)
define noalias nonnull {} addrspace(10)* @f({} addrspace(10)* %arg) local_unnamed_addr #2 {
bb:
  %i15 = bitcast {} addrspace(10)* %arg to {} addrspace(10)* addrspace(10)*
  %l = load {} addrspace(10)*, {} addrspace(10)* addrspace(10)* %i15
  ret {} addrspace(10)* %l
}

attributes #0 = { mustprogress nofree nounwind willreturn "enzyme_no_escaping_allocation" "enzymejl_world"="26725" }
attributes #2 = { mustprogress nofree nounwind willreturn "enzyme_no_escaping_allocation" }
attributes #4 = { nounwind willreturn "enzyme_no_escaping_allocation" }

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
!11 = !{!12, !14, !15, !16, !17}
!12 = distinct !{!12, !13, !"na_addr13"}
!13 = distinct !{!13, !"addr13"}
!14 = !{!"jnoalias_gcframe", !10}
!15 = !{!"jnoalias_stack", !10}
!16 = !{!"jnoalias_data", !10}
!17 = !{!"jnoalias_const", !10}
!18 = !{!14, !15, !16, !17}
!19 = !{!"Unknown", i32 -1, !20}
!20 = !{!"Pointer", i32 -1, !21}
!21 = !{!"Float@double"}
!22 = !{}
!23 = !{!24, !24, i64 0}
!24 = !{!"jtbaa_arraybuf", !25, i64 0}
!25 = !{!"jtbaa_data", !6, i64 0}
!26 = !{!16}
!27 = !{!14, !15, !9, !17}
!28 = !{!12, !14, !15, !9, !17}

; CHECK: define internal void @diffejulia_set_params_870({} addrspace(10)* nocapture readonly align 8 %arg, {} addrspace(10)* nocapture align 8 %"arg'", {} addrspace(10)* align 8 dereferenceable(24) %arg1, {} addrspace(10)* align 8 %"arg1'", i1 %arg2, i8* %tapeArg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to { {} addrspace(10)*, {} addrspace(10)* }*
; CHECK-NEXT:   %truetape = load { {} addrspace(10)*, {} addrspace(10)* }, { {} addrspace(10)*, {} addrspace(10)* }* %0, align 8, !enzyme_mustcache !
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %i901 = extractvalue { {} addrspace(10)*, {} addrspace(10)* } %truetape, 0
; CHECK-NEXT:   %"i90'ip_phi" = extractvalue { {} addrspace(10)*, {} addrspace(10)* } %truetape, 1
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(88) dereferenceable_or_null(88) i8* @malloc(i64 88)
; CHECK-NEXT:   %"i8'mi_malloccache" = bitcast i8* %malloccall to {} addrspace(10)**
; CHECK-NEXT:   br label %bb3

; CHECK: bb3:                                              ; preds = %bb12, %bb
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %bb12 ], [ 0, %bb ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %i5 = icmp eq i64 %iv, 10
; CHECK-NEXT:   br i1 %arg2, label %bb6, label %bb12

; CHECK: bb6:                                              ; preds = %bb3
; CHECK-NEXT:   %"i8'mi" = call noalias nonnull align 8 dereferenceable(24) {} addrspace(10)* @julia.gc_alloc_obj({}* null, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 137917234968496 to {}*) to {} addrspace(10)*)) 
; CHECK-NEXT:   %"i9'ipc" = bitcast {} addrspace(10)* %"i8'mi" to {} addrspace(10)* addrspace(10)*
; CHECK-NEXT:   store {} addrspace(10)* %"i90'ip_phi", {} addrspace(10)* addrspace(10)* %"i9'ipc", align 8
; CHECK-NEXT:   %1 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)** %"i8'mi_malloccache", i64 %iv
; CHECK-NEXT:   store {} addrspace(10)* %"i8'mi", {} addrspace(10)** %1, align 8
; CHECK-NEXT:   br label %bb12

; CHECK: bb12:                                             ; preds = %bb6, %bb3
; CHECK-NEXT:   br i1 %i5, label %remat_enter, label %bb3

; CHECK: invertbb:                                         ; preds = %invertbb12_phimerge
; CHECK-NEXT:   call void @diffef({} addrspace(10)* %arg1, {} addrspace(10)* {{(undef|poison)}})
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void

; CHECK: incinvertbb3:                                     ; preds = %invertbb12_phimerge
; CHECK-NEXT:   %2 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: invertbb12_phirc:                                 ; preds = %remat_bb3_bb12
; CHECK-NEXT:   %3 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)** %"i8'mi_malloccache", i64 %"iv'ac.0"
; CHECK-NEXT:   %4 = load {} addrspace(10)*, {} addrspace(10)** %3, align 8
; CHECK-NEXT:   br label %invertbb12_phimerge

; CHECK: invertbb12_phimerge:                              ; preds = %remat_bb3_bb12, %invertbb12_phirc
; CHECK-NEXT:   %5 = phi {} addrspace(10)* [ %4, %invertbb12_phirc ], [ %"arg1'", %remat_bb3_bb12 ]
; CHECK-NEXT:   %"i14'ipc_unwrap" = addrspacecast {} addrspace(10)* %5 to {} addrspace(11)*
; CHECK-NEXT:   %"i15'ipc_unwrap" = bitcast {} addrspace(11)* %"i14'ipc_unwrap" to double addrspace(13)* addrspace(11)*
; CHECK-NEXT:   %"i16'il_phi_unwrap" = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %"i15'ipc_unwrap", align 8
; CHECK-NEXT:   %6 = load double, double addrspace(13)* %"i16'il_phi_unwrap", align 8
; CHECK-NEXT:   %7 = fadd fast double %6, %9
; CHECK-NEXT:   store double %7, double addrspace(13)* %"i16'il_phi_unwrap", align 8
; CHECK-NEXT:   %8 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %8, label %invertbb, label %incinvertbb3

; CHECK: remat_enter:                                      ; preds = %bb12, %incinvertbb3
; CHECK-NEXT:   %i8_cache.0 = phi {} addrspace(10)* [ %i8_cache.1, %incinvertbb3 ], [ undef, %bb12 ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %2, %incinvertbb3 ], [ 10, %bb12 ]
; CHECK-NEXT:   br i1 %arg2, label %remat_bb3_bb6, label %remat_bb3_bb12

; CHECK: remat_bb3_bb6:                                    ; preds = %remat_enter
; CHECK-NEXT:   %remat_i8 = call noalias nonnull align 8 dereferenceable(24) {} addrspace(10)* @julia.gc_alloc_obj({}* null, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 137917234968496 to {}*) to {} addrspace(10)*)) 
; CHECK-NEXT:   %i9_unwrap = bitcast {} addrspace(10)* %remat_i8 to {} addrspace(10)* addrspace(10)*
; CHECK-NEXT:   store {} addrspace(10)* %i901, {} addrspace(10)* addrspace(10)* %i9_unwrap, align 8
; CHECK-NEXT:   br label %remat_bb3_bb12

; CHECK: remat_bb3_bb12:                                   ; preds = %remat_bb3_bb6, %remat_enter
; CHECK-NEXT:   %i8_cache.1 = phi {} addrspace(10)* [ %remat_i8, %remat_bb3_bb6 ], [ %i8_cache.0, %remat_enter ]
; CHECK-NEXT:   %"i17'ipc_unwrap" = bitcast {} addrspace(10)* %"arg'" to double addrspace(13)* addrspace(10)*
; CHECK-NEXT:   %"i18'il_phi_unwrap" = load double addrspace(13)*, double addrspace(13)* addrspace(10)* %"i17'ipc_unwrap", align 8
; CHECK-NEXT:   %9 = load double, double addrspace(13)* %"i18'il_phi_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double addrspace(13)* %"i18'il_phi_unwrap", align 8
; CHECK-NEXT:   br i1 %arg2, label %invertbb12_phirc, label %invertbb12_phimerge
; CHECK-NEXT: }


