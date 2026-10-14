; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Reduced from Enzyme.jl on Julia 1.10:
;   mid(x, w) = (t = x .- 1.0; sum(t[i] * t[i] / w[i] for i in eachindex(t)))
; called from a function that uses its result again, so that mid is
; differentiated in split mode.
;
; The temporary array t is allocated in mid, its data pointer (an addrspace(13)
; pointer the allocator set up) is loaded from the object, and the data is
; filled through that pointer. The reverse pass needs t[i].
;
; The allocation is rematerialized in the reverse pass. Since the fill stores go
; through the loaded data pointer rather than into the object itself, they must
; be replayed too, or the reverse pass reads an unfilled buffer.
source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

; Function Attrs: mustprogress nofree nounwind willreturn
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10)) #0

declare double @__enzyme_autodiff(...)

define double @dtop(ptr addrspace(10) %x, ptr addrspace(10) %dx, ptr addrspace(10) %w, ptr addrspace(10) %dw) {
  %r = call double (...) @__enzyme_autodiff(ptr @julia_top, metadata !"enzyme_dup", ptr addrspace(10) %x, ptr addrspace(10) %dx, metadata !"enzyme_dup", ptr addrspace(10) %w, ptr addrspace(10) %dw)
  ret double %r
}

define double @julia_top(ptr addrspace(10) %x, ptr addrspace(10) %w) {
  %r = call fastcc double @julia_mid(ptr addrspace(10) %x, ptr addrspace(10) %w)
  %r2 = fmul double %r, %r
  ret double %r2
}

; Function Attrs: noinline
define internal fastcc double @julia_mid(ptr addrspace(10) nocapture nofree noundef nonnull readonly align 16 dereferenceable(40) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer}" %x, ptr addrspace(10) nocapture nofree noundef nonnull readonly align 16 dereferenceable(40) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer}" %w) unnamed_addr #1 {
top:
  %v5 = addrspacecast ptr addrspace(10) %x to ptr addrspace(11)
  %arraylen_ptr = getelementptr inbounds { ptr addrspace(13), i64, i16, i16, i32 }, ptr addrspace(11) %v5, i64 0, i32 1
  %arraylen = load i64, ptr addrspace(11) %arraylen_ptr, align 8, !tbaa !0, !enzyme_inactive !5, !enzyme_type !6
  %v6 = call noalias nonnull "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer}" ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 40, ptr addrspace(10) null)
  %.not = icmp eq i64 %arraylen, 0
  br i1 %.not, label %L42, label %top.L24_crit_edge

top.L24_crit_edge:                                ; preds = %top
  %v7 = addrspacecast ptr addrspace(10) %x to ptr addrspace(11)
  %arrayptr.pre51 = load ptr addrspace(13), ptr addrspace(11) %v7, align 16, !tbaa !8, !enzyme_type !10
  %v8 = addrspacecast ptr addrspace(10) %v6 to ptr addrspace(11)
  %arrayptr11.pre52 = load ptr addrspace(13), ptr addrspace(11) %v8, align 8, !tbaa !8, !enzyme_type !10
  br label %L24

L24:                                              ; preds = %L24, %top.L24_crit_edge
  %iv = phi i64 [ %iv.next, %L24 ], [ 0, %top.L24_crit_edge ]
  %iv.next = add nuw nsw i64 %iv, 1
  %v9 = add nsw i64 %iv.next, -1
  %v10 = getelementptr inbounds double, ptr addrspace(13) %arrayptr.pre51, i64 %v9
  %arrayref = load double, ptr addrspace(13) %v10, align 8, !tbaa !13, !enzyme_type !16
  %v11 = fadd double %arrayref, -1.000000e+00
  %v12 = getelementptr inbounds double, ptr addrspace(13) %arrayptr11.pre52, i64 %v9
  store double %v11, ptr addrspace(13) %v12, align 8, !tbaa !13
  %.not53 = icmp eq i64 %iv.next, %arraylen
  %v13 = add nuw nsw i64 %iv.next, 1
  br i1 %.not53, label %L42.loopexit, label %L24

L42.loopexit:                                     ; preds = %L24
  br label %L42

L42:                                              ; preds = %L42.loopexit, %top
  %.pre-phi46 = addrspacecast ptr addrspace(10) %v6 to ptr addrspace(11)
  %arraylen_ptr15 = getelementptr inbounds { ptr addrspace(13), i64, i16, i16, i32 }, ptr addrspace(11) %.pre-phi46, i64 0, i32 1
  %arraylen16 = load i64, ptr addrspace(11) %arraylen_ptr15, align 8, !tbaa !0, !enzyme_inactive !5, !enzyme_type !6
  %.not54 = icmp eq i64 %arraylen16, 0
  br i1 %.not54, label %L74, label %L42.L54_crit_edge

L42.L54_crit_edge:                                ; preds = %L42
  %v14 = addrspacecast ptr addrspace(10) %v6 to ptr addrspace(11)
  %arrayptr25.pre55 = load ptr addrspace(13), ptr addrspace(11) %v14, align 8, !tbaa !8, !enzyme_type !10
  %v15 = addrspacecast ptr addrspace(10) %w to ptr addrspace(11)
  %arrayptr31.pre56 = load ptr addrspace(13), ptr addrspace(11) %v15, align 16, !tbaa !8, !enzyme_type !10
  br label %L54

L54:                                              ; preds = %L54, %L42.L54_crit_edge
  %iv1 = phi i64 [ %iv.next2, %L54 ], [ 0, %L42.L54_crit_edge ]
  %value_phi23 = phi double [ 0.000000e+00, %L42.L54_crit_edge ], [ %v21, %L54 ]
  %iv.next2 = add nuw nsw i64 %iv1, 1
  %v16 = add nsw i64 %iv.next2, -1
  %v17 = getelementptr inbounds double, ptr addrspace(13) %arrayptr25.pre55, i64 %v16
  %arrayref26 = load double, ptr addrspace(13) %v17, align 8, !tbaa !13, !enzyme_type !16
  %v18 = fmul double %arrayref26, %arrayref26
  %v19 = getelementptr inbounds double, ptr addrspace(13) %arrayptr31.pre56, i64 %v16
  %arrayref32 = load double, ptr addrspace(13) %v19, align 8, !tbaa !13, !enzyme_type !16
  %v20 = fdiv double %v18, %arrayref32
  %v21 = fadd double %value_phi23, %v20
  %.not57 = icmp eq i64 %iv.next2, %arraylen16
  %v22 = add nuw nsw i64 %iv.next2, 1
  br i1 %.not57, label %L74.loopexit, label %L54

L74.loopexit:                                     ; preds = %L54
  br label %L74

L74:                                              ; preds = %L74.loopexit, %L42
  %value_phi36 = phi double [ 0.000000e+00, %L42 ], [ %v21, %L74.loopexit ]
  ret double %value_phi36
}

attributes #0 = { mustprogress nofree nounwind willreturn "enzyme_no_escaping_allocation" }
attributes #1 = { noinline }

!0 = !{!1, !1, i64 0}
!1 = !{!"jtbaa_arraylen", !2, i64 0}
!2 = !{!"jtbaa_array", !3, i64 0}
!3 = !{!"jtbaa", !4, i64 0}
!4 = !{!"jtbaa"}
!5 = !{}
!6 = !{!"Unknown", i32 -1, !7}
!7 = !{!"Integer"}
!8 = !{!9, !9, i64 0}
!9 = !{!"jtbaa_arrayptr", !2, i64 0}
!10 = !{!"Unknown", i32 -1, !11}
!11 = !{!"Pointer", i32 -1, !12}
!12 = !{!"Float@double"}
!13 = !{!14, !14, i64 0}
!14 = !{!"jtbaa_arraybuf", !15, i64 0}
!15 = !{!"jtbaa_data", !3, i64 0}
!16 = !{!"Unknown", i32 -1, !12}

; CHECK: define internal fastcc void @diffejulia_mid(ptr addrspace(10) {{.*}} %x, ptr addrspace(10) {{.*}} %"x'", ptr addrspace(10) {{.*}} %w, ptr addrspace(10) {{.*}} %"w'", double %differeturn)

; The rematerialized allocation and the replayed fill loop.
; CHECK: %v6 = call noalias nonnull {{.*}}ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 40, ptr addrspace(10) null)

; CHECK: L24:
; CHECK-NEXT:   %iv2 = phi i64 [ %iv.next3, %L24 ], [ 0, %top.L24_crit_edge ]
; CHECK-NEXT:   %iv.next3 = add nuw nsw i64 %iv2, 1
; CHECK-NEXT:   %v9 = add nsw i64 %iv.next3, -1
; CHECK-NEXT:   %"v10'ipg" = getelementptr inbounds double, ptr addrspace(13) %"arrayptr.pre51'ipl", i64 %v9
; CHECK-NEXT:   %v10 = getelementptr inbounds double, ptr addrspace(13) %arrayptr.pre51, i64 %v9
; CHECK-NEXT:   %arrayref = load double, ptr addrspace(13) %v10, align 8
; CHECK-NEXT:   %v11 = fadd double %arrayref, -1.000000e+00
; CHECK-NEXT:   %v12 = getelementptr inbounds double, ptr addrspace(13) %arrayptr11.pre52, i64 %v9
; CHECK-NEXT:   store double %v11, ptr addrspace(13) %v12, align 8
; CHECK-NEXT:   %.not53 = icmp eq i64 %iv.next3, %arraylen
; CHECK-NEXT:   br i1 %.not53, label %L42.loopexit, label %L24
