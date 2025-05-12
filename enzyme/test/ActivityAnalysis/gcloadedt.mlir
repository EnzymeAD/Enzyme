; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=julia_f_8240 -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=julia_f_8240 -S | FileCheck %s

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=julia_f_8240 -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=julia_f_8240 -S | FileCheck %s

source_filename = "start"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"
target triple = "arm64-apple-darwin24.0.0"

declare {}*** @julia.get_pgcstack() local_unnamed_addr #1

declare void @julia.safepoint(i64*) local_unnamed_addr #2

declare nonnull {} addrspace(10)* @ijl_invoke({} addrspace(10)*, {} addrspace(10)** nocapture readonly, i32, {} addrspace(10)*) #3

declare nonnull {} addrspace(10)* @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) local_unnamed_addr #3

declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}*, i64, {} addrspace(10)*) local_unnamed_addr #4

declare void @ijl_throw({} addrspace(12)*) local_unnamed_addr #5

declare void @llvm.trap() #6

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #7

declare noundef nonnull {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* nocapture noundef nonnull readnone, {} addrspace(10)** noundef nonnull readnone) local_unnamed_addr #8

define "enzyme_type"="{[-1]:Float@double}" double @julia_f_8240({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4698642384" "enzymejl_parmtype_ref"="2" %arg, {} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(24) "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4523794896" "enzymejl_parmtype_ref"="2" %arg1) local_unnamed_addr #9 {
bb:
  %i = alloca [1 x i64], align 8
  %i2 = call {}*** @julia.get_pgcstack()
  %i3 = getelementptr inbounds {}**, {}*** %i2, i64 -14
  %i4 = bitcast {}*** %i3 to {}*
  %i5 = getelementptr inbounds {}**, {}*** %i2, i64 2
  %i6 = bitcast {}*** %i5 to i64***
  %i7 = load i64**, i64*** %i6, align 8, !tbaa !2
  %i8 = getelementptr inbounds i64*, i64** %i7, i64 2
  %i9 = load i64*, i64** %i8, align 8, !tbaa !6
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %i9)
  fence syncscope("singlethread") seq_cst
  %i10 = bitcast {} addrspace(10)* %arg to { i8*, {} addrspace(10)* } addrspace(10)*
  %i11 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i10 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i12 = bitcast {} addrspace(10)* %arg to {} addrspace(10)** addrspace(10)*
  %i13 = addrspacecast {} addrspace(10)** addrspace(10)* %i12 to {} addrspace(10)** addrspace(11)*
  %i14 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i13, align 8, !tbaa !27, !alias.scope !30, !noalias !31, !enzyme_type !32, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BFloat64\7D !8, !enzyme_nocache !8
  %i15 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i11, i64 0, i32 1
  %i16 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i15, align 8, !tbaa !27, !alias.scope !30, !noalias !31, !dereferenceable_or_null !35, !align !36, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_MUT_REF !8
  %i17 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i16, {} addrspace(10)** noundef %i14)
  %i18 = bitcast {} addrspace(10)* addrspace(13)* %i17 to double addrspace(13)*
  %i19 = load double, double addrspace(13)* %i18, align 8, !tbaa !39, !alias.scope !23, !noalias !41, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i20 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4752515456 to {} addrspace(10)**) unordered, align 128, !tbaa !43, !alias.scope !44, !noalias !45, !nonnull !8, !enzyme_inactive !8, !enzyme_type !46, !enzymejl_source_type_Memory\7BUInt8\7D !8, !enzymejl_byref_BITS_REF !8
  %i21 = icmp eq {} addrspace(10)* %i20, null
  br i1 %i21, label %bb98, label %bb100

bb23:                                             ; preds = %bb100
  unreachable

bb28:                                             ; preds = %bb100
  %i29 = bitcast {} addrspace(10)* %i20 to i8 addrspace(10)*, !enzyme_inactive !8
  %i30 = addrspacecast i8 addrspace(10)* %i29 to i8 addrspace(11)*, !enzyme_inactive !8
  %i31 = getelementptr inbounds i8, i8 addrspace(11)* %i30, i64 8
  %i32 = bitcast i8 addrspace(11)* %i31 to i64 addrspace(11)*
  %i33 = load i64, i64 addrspace(11)* %i32, align 8, !tbaa !50, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !51, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BUInt8\7D !8
  %i34 = inttoptr i64 %i33 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 1 %i34, i8 noundef 0, i64 %i104, i1 noundef false)
  call void @llvm.julia.gc_preserve_end(token %i101)
  %i35 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4696681696 to {} addrspace(10)**) unordered, align 32, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_inactive !8, !enzyme_type !52, !enzymejl_byref_BITS_REF !8, !enzymejl_source_type_Memory\7BSymbol\7D !8
  %i36 = icmp eq {} addrspace(10)* %i35, null
  br i1 %i36, label %bb106, label %bb108

bb37:                                             ; preds = %bb113
  ; call fastcc void @a3({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) %i114, i64 noundef signext 16)
  br label %bb38

bb38:                                             ; preds = %bb113, %bb37
  ; call fastcc void @a1({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) %i114, double %i19)
  %i39 = bitcast {} addrspace(10)* %arg1 to i8 addrspace(10)*, !enzyme_inactive !8
  %i40 = addrspacecast i8 addrspace(10)* %i39 to i8 addrspace(11)*, !enzyme_inactive !8
  %i41 = getelementptr inbounds i8, i8 addrspace(11)* %i40, i64 16
  %i42 = bitcast i8 addrspace(11)* %i41 to i64 addrspace(11)*
  %i43 = load i64, i64 addrspace(11)* %i42, align 8, !tbaa !43, !alias.scope !55, !noalias !56, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  call void @jl_({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) %i114)
  %i44 = icmp eq i64 %i43, 0
  br i1 %i44, label %bb45, label %bb48

bb45:                                             ; preds = %bb38
  %i46 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_BITS_REF !8
  %i47 = icmp eq {} addrspace(10)* %i46, null
  br i1 %i47, label %bb132, label %bb50

bb48:                                             ; preds = %bb38
  %i49 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4697777888 to {}*) to {} addrspace(10)*), i64 %i43) #27
  br label %bb50

bb50:                                             ; preds = %bb48, %bb45
  %i51 = phi {} addrspace(10)* [ %i49, %bb48 ], [ %i46, %bb45 ]
  %i52 = bitcast {} addrspace(10)* %i51 to { i64, {} addrspace(10)** } addrspace(10)*
  %i53 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i52 to { i64, {} addrspace(10)** } addrspace(11)*
  %i54 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i53, i64 0, i32 1
  %i55 = bitcast {} addrspace(10)** addrspace(11)* %i54 to i8* addrspace(11)*
  %i56 = load i8*, i8* addrspace(11)* %i55, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !32, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BFloat64\7D !8, !enzyme_nocache !8
  %i57 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i4, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4698642384 to {}*) to {} addrspace(10)*)) #23
  %i58 = bitcast {} addrspace(10)* %i57 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i59 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i58 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i60 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i59, i64 0, i32 0
  store i8* %i56, i8* addrspace(11)* %i60, align 8, !tbaa !27, !alias.scope !30, !noalias !57
  %i61 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i59, i64 0, i32 1
  store {} addrspace(10)* %i51, {} addrspace(10)* addrspace(11)* %i61, align 8, !tbaa !27, !alias.scope !30, !noalias !57
  %i62 = bitcast {} addrspace(10)* %i57 to i8 addrspace(10)*
  %i63 = addrspacecast i8 addrspace(10)* %i62 to i8 addrspace(11)*
  %i64 = getelementptr inbounds i8, i8 addrspace(11)* %i63, i64 16
  %i65 = bitcast i8 addrspace(11)* %i64 to i64 addrspace(11)*
  store i64 %i43, i64 addrspace(11)* %i65, align 8, !tbaa !58, !alias.scope !30, !noalias !57
  %i66 = icmp slt i64 %i43, 1
  %i67 = bitcast i8* %i56 to {} addrspace(10)**
  br i1 %i66, label %bb96, label %bb68

bb68:                                             ; preds = %bb50
  %i69 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i51, {} addrspace(10)** noundef %i67)
  %i70 = bitcast {} addrspace(10)* addrspace(13)* %i69 to double addrspace(13)*
  br label %bb71

bb71:                                             ; preds = %bb71, %bb68
  %i72 = phi i64 [ %i75, %bb71 ], [ 1, %bb68 ]
  %i73 = call fastcc double @a2({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(64) %i114)
  store double %i73, double addrspace(13)* %i70, align 8, !tbaa !39, !alias.scope !23, !noalias !24
  %i74 = icmp eq i64 %i72, %i43
  %i75 = add nuw i64 %i72, 1
  br i1 %i74, label %bb76, label %bb71

bb76:                                             ; preds = %bb71
  %i77 = call i64 @llvm.smax.i64(i64 %i43, i64 noundef 0)
  %i78 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i61, align 8, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_MUT_REF !8
  %i79 = add nsw i64 %i77, -1
  %i80 = icmp ugt i64 %i43, %i79
  %i81 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i78, {} addrspace(10)** noundef %i67)
  br i1 %i80, label %bb82, label %bb92

bb82:                                             ; preds = %bb82, %bb76
  %i83 = phi i64 [ %i91, %bb82 ], [ 1, %bb76 ]
  %i84 = phi double [ %i89, %bb82 ], [ 0.000000e+00, %bb76 ]
  %i85 = add nsw i64 %i83, -1
  %i86 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i81, i64 %i85
  %i87 = bitcast {} addrspace(10)* addrspace(13)* %i86 to double addrspace(13)*
  %i88 = load double, double addrspace(13)* %i87, align 8, !tbaa !39, !alias.scope !23, !noalias !41
  %i89 = fadd double %i84, %i88
  %i90 = icmp eq i64 %i83, %i77
  %i91 = add nuw i64 %i83, 1
  br i1 %i90, label %bb96, label %bb82

bb92:                                             ; preds = %bb76
  unreachable

bb96:                                             ; preds = %bb82, %bb50
  %i97 = phi double [ 0.000000e+00, %bb50 ], [ %i89, %bb82 ]
  ret double %i97

bb98:                                             ; preds = %bb
  unreachable

bb100:                                            ; preds = %bb
  %i101 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* nonnull %i20)
  %i102 = bitcast {} addrspace(10)* %i20 to i64 addrspace(10)*, !enzyme_inactive !8
  %i103 = addrspacecast i64 addrspace(10)* %i102 to i64 addrspace(11)*, !enzyme_inactive !8
  %i104 = load i64, i64 addrspace(11)* %i103, align 8, !tbaa !50, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i105 = icmp sgt i64 %i104, -1
  br i1 %i105, label %bb28, label %bb23

bb106:                                            ; preds = %bb28
  unreachable

bb108:                                            ; preds = %bb28
  %i109 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_BITS_REF !8
  %i110 = icmp eq {} addrspace(10)* %i109, null
  br i1 %i110, label %bb111, label %bb113

bb111:                                            ; preds = %bb108
  unreachable

bb113:                                            ; preds = %bb108
  %i114 = call noalias nonnull align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i4, i64 noundef 64, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 14415035984 to {}*) to {} addrspace(10)*)) #28
  %i115 = bitcast {} addrspace(10)* %i114 to {} addrspace(10)* addrspace(10)*
  %i116 = addrspacecast {} addrspace(10)* addrspace(10)* %i115 to {} addrspace(10)* addrspace(11)*
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i116, align 8, !tbaa !64, !alias.scope !23, !noalias !24
  %i117 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i116, i64 1
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i117, align 8, !tbaa !64, !alias.scope !23, !noalias !24
  %i118 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i116, i64 2
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i118, align 8, !tbaa !64, !alias.scope !23, !noalias !24
  store atomic {} addrspace(10)* %i20, {} addrspace(10)* addrspace(11)* %i116 release, align 8, !tbaa !64, !alias.scope !23, !noalias !24
  %i119 = bitcast {} addrspace(10)* %i114 to i8 addrspace(10)*
  %i120 = addrspacecast i8 addrspace(10)* %i119 to i8 addrspace(11)*
  %i121 = getelementptr inbounds i8, i8 addrspace(11)* %i120, i64 8
  %i122 = bitcast i8 addrspace(11)* %i121 to {} addrspace(10)* addrspace(11)*
  store atomic {} addrspace(10)* %i35, {} addrspace(10)* addrspace(11)* %i122 release, align 8, !tbaa !64, !alias.scope !23, !noalias !24
  %i123 = getelementptr inbounds i8, i8 addrspace(11)* %i120, i64 16
  %i124 = bitcast i8 addrspace(11)* %i123 to {} addrspace(10)* addrspace(11)*
  store atomic {} addrspace(10)* %i109, {} addrspace(10)* addrspace(11)* %i124 release, align 8, !tbaa !64, !alias.scope !23, !noalias !24
  %i125 = getelementptr inbounds i8, i8 addrspace(11)* %i120, i64 24
  %i126 = getelementptr inbounds i8, i8 addrspace(11)* %i120, i64 48
  %i127 = bitcast i8 addrspace(11)* %i126 to i64 addrspace(11)*
  call void @llvm.memset.p11i8.i64(i8 addrspace(11)* noundef align 8 dereferenceable(24) dereferenceable_or_null(40) %i125, i8 noundef 0, i64 noundef 24, i1 noundef false), !enzyme_truetype !66
  store i64 1, i64 addrspace(11)* %i127, align 8, !tbaa !64, !alias.scope !23, !noalias !24
  %i128 = getelementptr inbounds i8, i8 addrspace(11)* %i120, i64 56
  %i129 = bitcast i8 addrspace(11)* %i128 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i129, align 8, !tbaa !64, !alias.scope !23, !noalias !24
  %i130 = load i64, i64 addrspace(11)* %i103, align 8, !tbaa !67, !alias.scope !30, !noalias !31, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i131 = icmp eq i64 %i130, 16
  br i1 %i131, label %bb38, label %bb37

bb132:                                            ; preds = %bb45
  unreachable
}

declare token @llvm.julia.gc_preserve_begin(...) #10

declare noalias nonnull align 8 dereferenceable(8) {} addrspace(10)* @ijl_box_int64(i64 signext) local_unnamed_addr #11

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #12

declare void @llvm.julia.gc_preserve_end(token) #10

declare noalias nonnull align 16 dereferenceable(16) {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)*, i64) local_unnamed_addr #13

declare i64 @llvm.ctlz.i64(i64, i1 immarg) #14

declare void @julia.write_barrier({} addrspace(10)* readonly, ...) local_unnamed_addr #15

declare void @jl_({} addrspace(10)*) local_unnamed_addr #16

declare i64 @llvm.smax.i64(i64, i64) #14

declare void @llvm.memset.p11i8.i64(i8 addrspace(11)* nocapture writeonly, i8, i64, i1 immarg) #12

define internal fastcc void @a1({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" "enzymejl_parmtype"="14415035984" "enzymejl_parmtype_ref"="2" %arg, double "enzyme_type"="{[-1]:Float@double}" "enzymejl_parmtype"="4752516336" "enzymejl_parmtype_ref"="0" %arg1) unnamed_addr #17 {
bb:
  %i = alloca { i64, i8 }, align 8, !enzyme_inactive !8, !enzyme_type !69, !enzymejl_byref_MUT_REF !8, !enzymejl_allocart !71, !enzymejl_source_type_Tuple\7BInt64\2C\20UInt8\7D !8
  %i2 = call {}*** @julia.get_pgcstack()
  %i3 = getelementptr inbounds {}**, {}*** %i2, i64 2
  %i4 = bitcast {}*** %i3 to i64***
  %i5 = load i64**, i64*** %i4, align 8, !tbaa !2
  %i6 = getelementptr inbounds i64*, i64** %i5, i64 2
  %i7 = load i64*, i64** %i6, align 8, !tbaa !6
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %i7) #22
  fence syncscope("singlethread") seq_cst
  ; call fastcc void @a4({ i64, i8 }* noalias nocapture nofree noundef nonnull writeonly sret({ i64, i8 }) align 8 dereferenceable(16) %i, {} addrspace(10)* noundef nonnull align 8 dereferenceable(64) %arg)
  %i8 = getelementptr inbounds { i64, i8 }, { i64, i8 }* %i, i64 0, i32 0
  %i9 = load i64, i64* %i8, align 8, !tbaa !60, !alias.scope !62, !noalias !72, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i10 = icmp slt i64 %i9, 1
  br i1 %i10, label %bb38, label %bb11

bb11:                                             ; preds = %bb
  %i12 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i13 = addrspacecast i8 addrspace(10)* %i12 to i8 addrspace(11)*
  %i14 = getelementptr inbounds i8, i8 addrspace(11)* %i13, i64 40
  %i15 = bitcast i8 addrspace(11)* %i14 to i64 addrspace(11)*
  %i16 = load i64, i64 addrspace(11)* %i15, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_UInt64 !8
  %i17 = add i64 %i16, 1
  store i64 %i17, i64 addrspace(11)* %i15, align 8, !tbaa !64, !alias.scope !23, !noalias !73
  %i18 = getelementptr inbounds i8, i8 addrspace(11)* %i13, i64 8
  %i19 = bitcast i8 addrspace(11)* %i18 to {} addrspace(10)* addrspace(11)*
  %i20 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i19 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_inactive !8, !enzyme_type !52, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BSymbol\7D !8
  %i21 = bitcast {} addrspace(10)* %i20 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i22 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i21 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i23 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i22, i64 0, i32 1
  %i24 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i23, align 8, !tbaa !43, !alias.scope !76, !noalias !77, !nonnull !8, !enzyme_inactive !8, !enzyme_type !78, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BSymbol\7D !8
  %i25 = add nsw i64 %i9, -1
  %i26 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i20, {} addrspace(10)** noundef %i24)
  %i27 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i26, i64 %i25
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 4336030960 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(13)* %i27 release, align 8, !tbaa !79, !alias.scope !23, !noalias !73
  %i28 = getelementptr inbounds i8, i8 addrspace(11)* %i13, i64 16
  %i29 = bitcast i8 addrspace(11)* %i28 to {} addrspace(10)* addrspace(11)*
  %i30 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i29 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_MUT_REF !8
  %i31 = bitcast {} addrspace(10)* %i30 to { i64, {} addrspace(10)** } addrspace(10)*
  %i32 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i31 to { i64, {} addrspace(10)** } addrspace(11)*
  %i33 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i32, i64 0, i32 1
  %i34 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i33, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !32, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BFloat64\7D !8, !enzyme_nocache !8
  %i35 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i30, {} addrspace(10)** noundef %i34)
  %i36 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i35, i64 %i25
  %i37 = bitcast {} addrspace(10)* addrspace(13)* %i36 to double addrspace(13)*
  store double %arg1, double addrspace(13)* %i37, align 8, !tbaa !39, !alias.scope !23, !noalias !73
  br label %bb108

bb38:                                             ; preds = %bb
  %i39 = getelementptr inbounds { i64, i8 }, { i64, i8 }* %i, i64 0, i32 1
  %i40 = sub i64 0, %i9
  %i41 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i42 = addrspacecast i8 addrspace(10)* %i41 to i8 addrspace(11)*
  %i43 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 24
  %i44 = bitcast i8 addrspace(11)* %i43 to i64 addrspace(11)*
  %i45 = load i64, i64 addrspace(11)* %i44, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i46 = bitcast {} addrspace(10)* %arg to {} addrspace(10)* addrspace(10)*
  %i47 = addrspacecast {} addrspace(10)* addrspace(10)* %i46 to {} addrspace(10)* addrspace(11)*
  %i48 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i47 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !enzyme_inactive !8, !enzyme_type !46, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BUInt8\7D !8
  %i49 = bitcast {} addrspace(10)* %i48 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i50 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i49 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i51 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i50, i64 0, i32 1
  %i52 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i51, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_inactive !8, !enzyme_type !51, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BUInt8\7D !8
  %i53 = xor i64 %i9, -1
  %i54 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i48, {} addrspace(10)** noundef %i52)
  %i55 = bitcast {} addrspace(10)* addrspace(13)* %i54 to i8 addrspace(13)*, !enzyme_inactive !8
  %i56 = getelementptr inbounds i8, i8 addrspace(13)* %i55, i64 %i53
  %i57 = load i8, i8 addrspace(13)* %i56, align 1, !tbaa !39, !alias.scope !23, !noalias !41
  %i58 = icmp eq i8 %i57, 127
  %i59 = sext i1 %i58 to i64
  %i60 = add i64 %i45, %i59
  store i64 %i60, i64 addrspace(11)* %i44, align 8, !tbaa !64, !alias.scope !23, !noalias !73
  %i61 = load i8, i8* %i39, align 8, !tbaa !60, !alias.scope !62, !noalias !72, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_UInt8 !8
  store i8 %i61, i8 addrspace(13)* %i56, align 1, !tbaa !39, !alias.scope !23, !noalias !73
  %i62 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 8
  %i63 = bitcast i8 addrspace(11)* %i62 to {} addrspace(10)* addrspace(11)*
  %i64 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i63 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_inactive !8, !enzyme_type !52, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BSymbol\7D !8
  %i65 = bitcast {} addrspace(10)* %i64 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i66 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i65 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i67 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i66, i64 0, i32 1
  %i68 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i67, align 8, !tbaa !43, !alias.scope !76, !noalias !77, !nonnull !8, !enzyme_inactive !8, !enzyme_type !78, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BSymbol\7D !8
  %i69 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i64, {} addrspace(10)** noundef %i68)
  %i70 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i69, i64 %i53
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 4336030960 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(13)* %i70 release, align 8, !tbaa !79, !alias.scope !23, !noalias !73
  %i71 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 16
  %i72 = bitcast i8 addrspace(11)* %i71 to {} addrspace(10)* addrspace(11)*
  %i73 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i72 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_MUT_REF !8
  %i74 = bitcast {} addrspace(10)* %i73 to { i64, {} addrspace(10)** } addrspace(10)*
  %i75 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i74 to { i64, {} addrspace(10)** } addrspace(11)*
  %i76 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i75, i64 0, i32 1
  %i77 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i76, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !32, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BFloat64\7D !8, !enzyme_nocache !8
  %i78 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i73, {} addrspace(10)** noundef %i77)
  %i79 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i78, i64 %i53
  %i80 = bitcast {} addrspace(10)* addrspace(13)* %i79 to double addrspace(13)*
  store double %arg1, double addrspace(13)* %i80, align 8, !tbaa !39, !alias.scope !23, !noalias !73
  %i81 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 32
  %i82 = bitcast i8 addrspace(11)* %i81 to i64 addrspace(11)*
  %i83 = load i64, i64 addrspace(11)* %i82, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i84 = add i64 %i83, 1
  store i64 %i84, i64 addrspace(11)* %i82, align 8, !tbaa !64, !alias.scope !23, !noalias !73
  %i85 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 40
  %i86 = bitcast i8 addrspace(11)* %i85 to i64 addrspace(11)*
  %i87 = load i64, i64 addrspace(11)* %i86, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_UInt64 !8
  %i88 = add i64 %i87, 1
  store i64 %i88, i64 addrspace(11)* %i86, align 8, !tbaa !64, !alias.scope !23, !noalias !73
  %i89 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 48
  %i90 = bitcast i8 addrspace(11)* %i89 to i64 addrspace(11)*
  %i91 = load i64, i64 addrspace(11)* %i90, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i92 = icmp sgt i64 %i91, %i40
  br i1 %i92, label %bb93, label %bb94

bb93:                                             ; preds = %bb38
  store i64 %i40, i64 addrspace(11)* %i90, align 8, !tbaa !64, !alias.scope !23, !noalias !73
  br label %bb94

bb94:                                             ; preds = %bb93, %bb38
  %i95 = add i64 %i84, %i60
  %i96 = mul i64 %i95, 3
  %i97 = bitcast {} addrspace(10)* %i64 to i64 addrspace(10)*, !enzyme_inactive !8
  %i98 = addrspacecast i64 addrspace(10)* %i97 to i64 addrspace(11)*, !enzyme_inactive !8
  %i99 = load i64, i64 addrspace(11)* %i98, align 8, !tbaa !67, !alias.scope !30, !noalias !31, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i100 = shl i64 %i99, 1
  %i101 = icmp slt i64 %i100, %i96
  br i1 %i101, label %bb102, label %bb108

bb102:                                            ; preds = %bb94
  %i103 = icmp slt i64 %i84, 64001
  %i104 = shl i64 %i84, 1
  %i105 = shl i64 %i84, 2
  %i106 = call i64 @llvm.smax.i64(i64 %i105, i64 noundef 4)
  %i107 = select i1 %i103, i64 %i106, i64 %i104
  call fastcc void @a3({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) %arg, i64 signext %i107)
  br label %bb108

bb108:                                            ; preds = %bb102, %bb94, %bb11
  ret void
}

define internal fastcc "enzyme_type"="{[-1]:Float@double}" double @a2({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" "enzymejl_parmtype"="14415035984" "enzymejl_parmtype_ref"="2" %arg) unnamed_addr #18 {
bb:
  %i = call {}*** @julia.get_pgcstack()
  %i1 = getelementptr inbounds {}**, {}*** %i, i64 -14
  %i2 = bitcast {}*** %i1 to {}*
  %i3 = getelementptr inbounds {}**, {}*** %i, i64 2
  %i4 = bitcast {}*** %i3 to i64***
  %i5 = load i64**, i64*** %i4, align 8, !tbaa !2
  %i6 = getelementptr inbounds i64*, i64** %i5, i64 2
  %i7 = load i64*, i64** %i6, align 8, !tbaa !6
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %i7) #22
  fence syncscope("singlethread") seq_cst
  %i8 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i9 = addrspacecast i8 addrspace(10)* %i8 to i8 addrspace(11)*
  %i10 = getelementptr inbounds i8, i8 addrspace(11)* %i9, i64 32
  %i11 = bitcast i8 addrspace(11)* %i10 to i64 addrspace(11)*
  %i12 = load i64, i64 addrspace(11)* %i11, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i13 = icmp eq i64 %i12, 0
  br i1 %i13, label %bb69, label %bb14

bb14:                                             ; preds = %bb
  %i15 = getelementptr inbounds i8, i8 addrspace(11)* %i9, i64 8
  %i16 = bitcast i8 addrspace(11)* %i15 to {} addrspace(10)* addrspace(11)*
  %i17 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i16 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_inactive !8, !enzyme_type !52, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BSymbol\7D !8
  %i18 = getelementptr inbounds i8, i8 addrspace(11)* %i9, i64 56
  %i19 = bitcast i8 addrspace(11)* %i18 to i64 addrspace(11)*
  %i20 = load i64, i64 addrspace(11)* %i19, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i21 = bitcast {} addrspace(10)* %i17 to i64 addrspace(10)*, !enzyme_inactive !8
  %i22 = addrspacecast i64 addrspace(10)* %i21 to i64 addrspace(11)*, !enzyme_inactive !8
  %i23 = load i64, i64 addrspace(11)* %i22, align 8, !tbaa !67, !alias.scope !30, !noalias !31, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i24 = icmp slt i64 %i20, %i23
  br i1 %i24, label %bb25, label %bb62

bb25:                                             ; preds = %bb14
  %i26 = load i64, i64 addrspace(11)* getelementptr inbounds (i64, i64 addrspace(11)* addrspacecast (i64* inttoptr (i64 4336030960 to i64*) to i64 addrspace(11)*), i64 2), align 16, !tbaa !6, !alias.scope !9, !noalias !12
  %i27 = add nsw i64 %i23, -1
  %i28 = lshr i64 %i26, 57
  %i29 = trunc i64 %i28 to i8
  %i30 = or i8 %i29, -128
  %i31 = bitcast {} addrspace(10)* %arg to {} addrspace(10)* addrspace(10)*
  %i32 = addrspacecast {} addrspace(10)* addrspace(10)* %i31 to {} addrspace(10)* addrspace(11)*
  %i33 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i32 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !46, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BUInt8\7D !8
  %i34 = bitcast {} addrspace(10)* %i33 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i35 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i34 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i36 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i35, i64 0, i32 1
  %i37 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i36, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_inactive !8, !enzyme_type !51, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BUInt8\7D !8
  %i38 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i33, {} addrspace(10)** noundef %i37)
  %i39 = bitcast {} addrspace(10)* addrspace(13)* %i38 to i8 addrspace(13)*, !enzyme_inactive !8
  %i40 = bitcast {} addrspace(10)* %i17 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i41 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i40 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i42 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i41, i64 0, i32 1
  %i43 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i42, align 8, !enzyme_inactive !8, !enzyme_type !78, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BSymbol\7D !8
  %i44 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i17, {} addrspace(10)** noundef %i43)
  br label %bb45

bb45:                                             ; preds = %bb59, %bb25
  %i46 = phi i64 [ %i26, %bb25 ], [ %i49, %bb59 ]
  %i47 = phi i64 [ 0, %bb25 ], [ %i60, %bb59 ]
  %i48 = and i64 %i46, %i27
  %i49 = add i64 %i48, 1
  %i50 = getelementptr inbounds i8, i8 addrspace(13)* %i39, i64 %i48
  %i51 = load i8, i8 addrspace(13)* %i50, align 1, !tbaa !39, !alias.scope !23, !noalias !41
  %i52 = icmp eq i8 %i51, 0
  br i1 %i52, label %bb69, label %bb53

bb53:                                             ; preds = %bb45
  %i54 = icmp eq i8 %i30, %i51
  br i1 %i54, label %bb55, label %bb59

bb55:                                             ; preds = %bb53
  %i56 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i44, i64 %i48
  %i57 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i56 unordered, align 8, !tbaa !79, !alias.scope !23, !noalias !41, !enzyme_type !51
  %i58 = icmp eq {} addrspace(10)* %i57, null
  br i1 %i58, label %bb86, label %bb89

bb59:                                             ; preds = %bb89, %bb53
  %i60 = add i64 %i47, 1
  %i61 = icmp slt i64 %i20, %i60
  br i1 %i61, label %bb69, label %bb45

bb62:                                             ; preds = %bb14
  %i63 = call noalias nonnull align 8 dereferenceable(8) "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,-1]:Pointer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i2, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4739065968 to {}*) to {} addrspace(10)*)) #23
  %i64 = bitcast {} addrspace(10)* %i63 to [1 x {} addrspace(10)*] addrspace(10)*, !enzyme_inactive !8
  %i65 = getelementptr [1 x {} addrspace(10)*], [1 x {} addrspace(10)*] addrspace(10)* %i64, i64 0, i64 0
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 4815715632 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %i65, align 8, !tbaa !19, !alias.scope !23, !noalias !81
  %i66 = addrspacecast {} addrspace(10)* %i63 to {} addrspace(12)*, !enzyme_inactive !8
  call void @ijl_throw({} addrspace(12)* %i66) #25
  unreachable

bb67:                                             ; preds = %bb89
  %i68 = icmp sgt i64 %i49, -1
  br i1 %i68, label %bb74, label %bb69

bb69:                                             ; preds = %bb67, %bb59, %bb45, %bb
  %i70 = call noalias nonnull align 8 dereferenceable(8) "enzyme_type"="{[-1]:Pointer, [-1,-1]:Pointer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i2, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4735480912 to {}*) to {} addrspace(10)*)) #23
  %i71 = bitcast {} addrspace(10)* %i70 to [1 x {} addrspace(10)*] addrspace(10)*
  %i72 = getelementptr [1 x {} addrspace(10)*], [1 x {} addrspace(10)*] addrspace(10)* %i71, i64 0, i64 0
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 4336030960 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %i72, align 8, !tbaa !19, !alias.scope !23, !noalias !81
  %i73 = addrspacecast {} addrspace(10)* %i70 to {} addrspace(12)*
  call void @ijl_throw({} addrspace(12)* %i73) #25
  unreachable

bb74:                                             ; preds = %bb67
  %i75 = getelementptr inbounds i8, i8 addrspace(11)* %i9, i64 16
  %i76 = bitcast i8 addrspace(11)* %i75 to {} addrspace(10)* addrspace(11)*
  %i77 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i76 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_MUT_REF !8
  %i78 = bitcast {} addrspace(10)* %i77 to { i64, {} addrspace(10)** } addrspace(10)*
  %i79 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i78 to { i64, {} addrspace(10)** } addrspace(11)*
  %i80 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i79, i64 0, i32 1
  %i81 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i80, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !32, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BFloat64\7D !8, !enzyme_nocache !8
  %i82 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i77, {} addrspace(10)** noundef %i81)
  %i83 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i82, i64 %i48
  %i84 = bitcast {} addrspace(10)* addrspace(13)* %i83 to double addrspace(13)*
  %i85 = load double, double addrspace(13)* %i84, align 8, !tbaa !39, !alias.scope !23, !noalias !41
  ret double %i85

bb86:                                             ; preds = %bb55
  unreachable

bb89:                                             ; preds = %bb55
  %i90 = addrspacecast {} addrspace(10)* %i57 to {} addrspace(11)*
  %i91 = icmp eq {} addrspace(11)* %i90, addrspacecast ({}* inttoptr (i64 4336030960 to {}*) to {} addrspace(11)*)
  br i1 %i91, label %bb67, label %bb59
}

define internal fastcc void @a3({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" "enzymejl_parmtype"="14415035984" "enzymejl_parmtype_ref"="2" %arg, i64 signext "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" "enzymejl_parmtype"="4752517136" "enzymejl_parmtype_ref"="0" %arg1) unnamed_addr #19 {
bb:
  %i = call {}*** @julia.get_pgcstack()
  %i2 = getelementptr inbounds {}**, {}*** %i, i64 2
  %i3 = bitcast {}*** %i2 to i64***
  %i4 = load i64**, i64*** %i3, align 8, !tbaa !2
  %i5 = getelementptr inbounds i64*, i64** %i4, i64 2
  %i6 = load i64*, i64** %i5, align 8, !tbaa !6
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %i6) #22
  fence syncscope("singlethread") seq_cst
  %i7 = bitcast {} addrspace(10)* %arg to {} addrspace(10)* addrspace(10)*
  %i8 = addrspacecast {} addrspace(10)* addrspace(10)* %i7 to {} addrspace(10)* addrspace(11)*
  %i9 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i8 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_inactive !8, !enzyme_type !46, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BUInt8\7D !8
  %i10 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i11 = addrspacecast i8 addrspace(10)* %i10 to i8 addrspace(11)*
  %i12 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 8
  %i13 = bitcast i8 addrspace(11)* %i12 to {} addrspace(10)* addrspace(11)*
  %i14 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i13 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_inactive !8, !enzyme_type !52, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BSymbol\7D !8
  %i15 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 16
  %i16 = bitcast i8 addrspace(11)* %i15 to {} addrspace(10)* addrspace(11)*
  %i17 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i16 unordered, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !nonnull !8, !dereferenceable !35, !align !36, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_MUT_REF !8
  %i18 = icmp sgt i64 %arg1, 15
  br i1 %i18, label %bb19, label %bb26

bb19:                                             ; preds = %bb
  %i20 = add nsw i64 %arg1, -1
  %i21 = call i64 @llvm.ctlz.i64(i64 %i20, i1 noundef false), !range !84
  %i22 = sub nuw nsw i64 64, %i21
  %i23 = shl nuw i64 1, %i22
  %i24 = icmp eq i64 %i21, 0
  %i25 = select i1 %i24, i64 0, i64 %i23
  br label %bb26

bb26:                                             ; preds = %bb19, %bb
  %i27 = phi i64 [ %i25, %bb19 ], [ 16, %bb ]
  %i28 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 40
  %i29 = bitcast i8 addrspace(11)* %i28 to i64 addrspace(11)*
  %i30 = load i64, i64 addrspace(11)* %i29, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_UInt64 !8
  %i31 = add i64 %i30, 1
  store i64 %i31, i64 addrspace(11)* %i29, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  %i32 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 48
  %i33 = bitcast i8 addrspace(11)* %i32 to i64 addrspace(11)*
  store i64 1, i64 addrspace(11)* %i33, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  %i34 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 32
  %i35 = bitcast i8 addrspace(11)* %i34 to i64 addrspace(11)*
  %i36 = load i64, i64 addrspace(11)* %i35, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i37 = icmp eq i64 %i36, 0
  br i1 %i37, label %bb38, label %bb80

bb38:                                             ; preds = %bb26
  %i39 = icmp eq i64 %i27, 0
  br i1 %i39, label %bb40, label %bb43

bb40:                                             ; preds = %bb38
  %i41 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4752515456 to {} addrspace(10)**) unordered, align 128, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_inactive !8, !enzyme_type !46, !enzymejl_source_type_Memory\7BUInt8\7D !8, !enzymejl_byref_BITS_REF !8
  %i42 = icmp eq {} addrspace(10)* %i41, null
  br i1 %i42, label %bb196, label %bb45

bb43:                                             ; preds = %bb38
  %i44 = call noalias "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4752515424 to {}*) to {} addrspace(10)*), i64 %i27) #27
  br label %bb45

bb45:                                             ; preds = %bb43, %bb40
  %i46 = phi {} addrspace(10)* [ %i44, %bb43 ], [ %i41, %bb40 ], !enzyme_inactive !8
  store atomic {} addrspace(10)* %i46, {} addrspace(10)* addrspace(11)* %i8 release, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i46) #29
  %i47 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* nonnull %i46)
  %i48 = bitcast {} addrspace(10)* %i46 to i64 addrspace(10)*, !enzyme_inactive !8
  %i49 = addrspacecast i64 addrspace(10)* %i48 to i64 addrspace(11)*, !enzyme_inactive !8
  %i50 = load i64, i64 addrspace(11)* %i49, align 8, !tbaa !67, !alias.scope !30, !noalias !31, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i51 = icmp sgt i64 %i50, -1
  br i1 %i51, label %bb57, label %bb52

bb52:                                             ; preds = %bb45
  unreachable

bb57:                                             ; preds = %bb45
  %i58 = bitcast {} addrspace(10)* %i46 to i8 addrspace(10)*, !enzyme_inactive !8
  %i59 = addrspacecast i8 addrspace(10)* %i58 to i8 addrspace(11)*, !enzyme_inactive !8
  %i60 = getelementptr inbounds i8, i8 addrspace(11)* %i59, i64 8
  %i61 = bitcast i8 addrspace(11)* %i60 to i64 addrspace(11)*
  %i62 = load i64, i64 addrspace(11)* %i61, align 8, !tbaa !88, !alias.scope !30, !noalias !31, !enzyme_inactive !8, !enzyme_type !51, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BUInt8\7D !8
  %i63 = inttoptr i64 %i62 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 1 %i63, i8 noundef 0, i64 %i50, i1 noundef false)
  call void @llvm.julia.gc_preserve_end(token %i47)
  br i1 %i39, label %bb64, label %bb70

bb64:                                             ; preds = %bb57
  %i65 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4696681696 to {} addrspace(10)**) unordered, align 32, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_inactive !8, !enzyme_type !52, !enzymejl_byref_BITS_REF !8, !enzymejl_source_type_Memory\7BSymbol\7D !8
  %i66 = icmp eq {} addrspace(10)* %i65, null
  br i1 %i66, label %bb199, label %bb67

bb67:                                             ; preds = %bb64
  store atomic {} addrspace(10)* %i65, {} addrspace(10)* addrspace(11)* %i13 release, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i65) #29
  %i68 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_BITS_REF !8
  %i69 = icmp eq {} addrspace(10)* %i68, null
  br i1 %i69, label %bb202, label %bb74

bb70:                                             ; preds = %bb57
  %i71 = call noalias "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,0]:Pointer}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4696681664 to {}*) to {} addrspace(10)*), i64 %i27) #27
  store atomic {} addrspace(10)* %i71, {} addrspace(10)* addrspace(11)* %i13 release, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i71) #29
  %i72 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4697777888 to {}*) to {} addrspace(10)*), i64 %i27) #27
  br label %bb74

bb73:                                             ; preds = %bb188, %bb74
  ret void

bb74:                                             ; preds = %bb70, %bb67
  %i75 = phi {} addrspace(10)* [ %i72, %bb70 ], [ %i68, %bb67 ]
  store atomic {} addrspace(10)* %i75, {} addrspace(10)* addrspace(11)* %i16 release, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i75) #29
  %i76 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 24
  %i77 = bitcast i8 addrspace(11)* %i76 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i77, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  %i78 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 56
  %i79 = bitcast i8 addrspace(11)* %i78 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i79, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  br label %bb73

bb80:                                             ; preds = %bb26
  %i81 = icmp eq i64 %i27, 0
  br i1 %i81, label %bb82, label %bb85

bb82:                                             ; preds = %bb80
  %i83 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4752515456 to {} addrspace(10)**) unordered, align 128, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_inactive !8, !enzyme_type !46, !enzymejl_source_type_Memory\7BUInt8\7D !8, !enzymejl_byref_BITS_REF !8
  %i84 = icmp eq {} addrspace(10)* %i83, null
  br i1 %i84, label %bb205, label %bb87

bb85:                                             ; preds = %bb80
  %i86 = call noalias "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4752515424 to {}*) to {} addrspace(10)*), i64 %i27) #27
  br label %bb87

bb87:                                             ; preds = %bb85, %bb82
  %i88 = phi {} addrspace(10)* [ %i86, %bb85 ], [ %i83, %bb82 ], !enzyme_inactive !8
  %i89 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* nonnull %i88)
  %i90 = bitcast {} addrspace(10)* %i88 to i64 addrspace(10)*, !enzyme_inactive !8
  %i91 = addrspacecast i64 addrspace(10)* %i90 to i64 addrspace(11)*, !enzyme_inactive !8
  %i92 = load i64, i64 addrspace(11)* %i91, align 8, !tbaa !67, !alias.scope !30, !noalias !31, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i93 = icmp sgt i64 %i92, -1
  br i1 %i93, label %bb99, label %bb94

bb94:                                             ; preds = %bb87
  unreachable

bb99:                                             ; preds = %bb87
  %i100 = bitcast {} addrspace(10)* %i88 to i8 addrspace(10)*, !enzyme_inactive !8
  %i101 = addrspacecast i8 addrspace(10)* %i100 to i8 addrspace(11)*, !enzyme_inactive !8
  %i102 = getelementptr inbounds i8, i8 addrspace(11)* %i101, i64 8
  %i103 = bitcast i8 addrspace(11)* %i102 to i64 addrspace(11)*
  %i104 = load i64, i64 addrspace(11)* %i103, align 8, !tbaa !88, !alias.scope !30, !noalias !31, !enzyme_inactive !8, !enzyme_type !51, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BUInt8\7D !8
  %i105 = inttoptr i64 %i104 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 1 %i105, i8 noundef 0, i64 %i92, i1 noundef false)
  call void @llvm.julia.gc_preserve_end(token %i89)
  br i1 %i81, label %bb106, label %bb112

bb106:                                            ; preds = %bb99
  %i107 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4696681696 to {} addrspace(10)**) unordered, align 32, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_inactive !8, !enzyme_type !52, !enzymejl_byref_BITS_REF !8, !enzymejl_source_type_Memory\7BSymbol\7D !8
  %i108 = icmp eq {} addrspace(10)* %i107, null
  br i1 %i108, label %bb208, label %bb109

bb109:                                            ; preds = %bb106
  %i110 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !37, !enzymejl_source_type_Memory\7BFloat64\7D !8, !enzymejl_byref_BITS_REF !8
  %i111 = icmp eq {} addrspace(10)* %i110, null
  br i1 %i111, label %bb211, label %bb115

bb112:                                            ; preds = %bb99
  %i113 = call noalias "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,0]:Pointer}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4696681664 to {}*) to {} addrspace(10)*), i64 %i27) #27
  %i114 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4697777888 to {}*) to {} addrspace(10)*), i64 %i27) #27
  br label %bb115

bb115:                                            ; preds = %bb112, %bb109
  %i116 = phi {} addrspace(10)* [ %i113, %bb112 ], [ %i107, %bb109 ], !enzyme_inactive !8
  %i117 = phi {} addrspace(10)* [ %i114, %bb112 ], [ %i110, %bb109 ]
  %i118 = load i64, i64 addrspace(11)* %i29, align 8, !tbaa !64, !alias.scope !23, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_UInt64 !8
  %i119 = bitcast {} addrspace(10)* %i9 to i64 addrspace(10)*, !enzyme_inactive !8
  %i120 = addrspacecast i64 addrspace(10)* %i119 to i64 addrspace(11)*, !enzyme_inactive !8
  %i121 = load i64, i64 addrspace(11)* %i120, align 8, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i122 = call i64 @llvm.smax.i64(i64 %i121, i64 noundef 0)
  %i123 = icmp slt i64 %i121, 1
  br i1 %i123, label %bb188, label %bb124

bb124:                                            ; preds = %bb115
  %i125 = bitcast {} addrspace(10)* %i9 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i126 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i125 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i127 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i126, i64 0, i32 1
  %i128 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i127, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_inactive !8, !enzyme_type !51, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BUInt8\7D !8
  %i129 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i9, {} addrspace(10)** noundef %i128)
  %i130 = bitcast {} addrspace(10)* addrspace(13)* %i129 to i8 addrspace(13)*, !enzyme_inactive !8
  %i131 = bitcast {} addrspace(10)* %i14 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i132 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i131 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i133 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i132, i64 0, i32 1
  %i134 = bitcast {} addrspace(10)* %i17 to { i64, {} addrspace(10)** } addrspace(10)*
  %i135 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i134 to { i64, {} addrspace(10)** } addrspace(11)*
  %i136 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i135, i64 0, i32 1
  %i137 = add i64 %i27, -1
  %i138 = bitcast {} addrspace(10)* %i88 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i139 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i138 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i140 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i139, i64 0, i32 1
  %i141 = bitcast {} addrspace(10)* %i116 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !8
  %i142 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i141 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !8
  %i143 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i142, i64 0, i32 1
  %i144 = bitcast {} addrspace(10)* %i117 to { i64, {} addrspace(10)** } addrspace(10)*
  %i145 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i144 to { i64, {} addrspace(10)** } addrspace(11)*
  %i146 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i145, i64 0, i32 1
  br label %bb147

bb147:                                            ; preds = %bb183, %bb124
  %i148 = phi i64 [ 1, %bb124 ], [ %i187, %bb183 ]
  %i149 = phi i64 [ 0, %bb124 ], [ %i184, %bb183 ]
  %i150 = phi i64 [ 0, %bb124 ], [ %i185, %bb183 ]
  %i151 = add nsw i64 %i148, -1
  %i152 = getelementptr inbounds i8, i8 addrspace(13)* %i130, i64 %i151
  %i153 = load i8, i8 addrspace(13)* %i152, align 1, !tbaa !39, !alias.scope !23, !noalias !41
  %i154 = icmp sgt i8 %i153, -1
  br i1 %i154, label %bb183, label %bb155

bb155:                                            ; preds = %bb147
  %i156 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i133, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_inactive !8, !enzyme_type !78, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BSymbol\7D !8
  %i157 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i14, {} addrspace(10)** noundef %i156)
  %i158 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i157, i64 %i151
  %i159 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i158 unordered, align 8, !tbaa !79, !alias.scope !23, !noalias !41, !enzyme_type !51
  %i160 = icmp eq {} addrspace(10)* %i159, null
  br i1 %i160, label %bb214, label %bb217

bb161:                                            ; preds = %bb217, %bb161
  %i162 = phi i64 [ %i164, %bb161 ], [ %i228, %bb217 ]
  %i163 = and i64 %i162, %i137
  %i164 = add i64 %i163, 1
  %i165 = getelementptr inbounds i8, i8 addrspace(13)* %i231, i64 %i163
  %i166 = load i8, i8 addrspace(13)* %i165, align 1, !tbaa !39, !alias.scope !23, !noalias !41
  %i167 = icmp eq i8 %i166, 0
  br i1 %i167, label %bb168, label %bb161

bb168:                                            ; preds = %bb217, %bb161
  %i169 = phi i64 [ %i228, %bb217 ], [ %i164, %bb161 ]
  %i170 = phi i64 [ %i227, %bb217 ], [ %i163, %bb161 ]
  %i171 = phi i8 addrspace(13)* [ %i232, %bb217 ], [ %i165, %bb161 ]
  %i172 = sub i64 %i169, %i228
  %i173 = and i64 %i172, %i137
  %i174 = call i64 @llvm.smax.i64(i64 %i149, i64 %i173)
  store i8 %i153, i8 addrspace(13)* %i171, align 1, !tbaa !39, !alias.scope !23, !noalias !85
  %i175 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i143, align 8, !tbaa !43, !alias.scope !76, !noalias !77, !nonnull !8, !enzyme_inactive !8, !enzyme_type !78, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BSymbol\7D !8
  %i176 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i116, {} addrspace(10)** noundef %i175)
  %i177 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i176, i64 %i170
  store atomic {} addrspace(10)* %i159, {} addrspace(10)* addrspace(13)* %i177 release, align 8, !tbaa !79, !alias.scope !23, !noalias !85
  %i178 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i146, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !32, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BFloat64\7D !8, !enzyme_nocache !8
  %i179 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i117, {} addrspace(10)** noundef %i178)
  %i180 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i179, i64 %i170
  %i181 = bitcast {} addrspace(10)* addrspace(13)* %i180 to double addrspace(13)*
  store double %i222, double addrspace(13)* %i181, align 8, !tbaa !39, !alias.scope !23, !noalias !85
  %i182 = add i64 %i150, 1
  br label %bb183

bb183:                                            ; preds = %bb168, %bb147
  %i184 = phi i64 [ %i174, %bb168 ], [ %i149, %bb147 ]
  %i185 = phi i64 [ %i182, %bb168 ], [ %i150, %bb147 ]
  %i186 = icmp eq i64 %i148, %i122
  %i187 = add nuw i64 %i148, 1
  br i1 %i186, label %bb188, label %bb147

bb188:                                            ; preds = %bb183, %bb115
  %i189 = phi i64 [ 0, %bb115 ], [ %i184, %bb183 ]
  %i190 = phi i64 [ 0, %bb115 ], [ %i185, %bb183 ]
  %i191 = add i64 %i118, 1
  store i64 %i191, i64 addrspace(11)* %i29, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  store atomic {} addrspace(10)* %i88, {} addrspace(10)* addrspace(11)* %i8 release, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* %i88) #29
  store atomic {} addrspace(10)* %i116, {} addrspace(10)* addrspace(11)* %i13 release, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i116) #29
  store atomic {} addrspace(10)* %i117, {} addrspace(10)* addrspace(11)* %i16 release, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* %i117) #29
  store i64 %i190, i64 addrspace(11)* %i35, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  %i192 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 24
  %i193 = bitcast i8 addrspace(11)* %i192 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i193, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  %i194 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 56
  %i195 = bitcast i8 addrspace(11)* %i194 to i64 addrspace(11)*
  store i64 %i189, i64 addrspace(11)* %i195, align 8, !tbaa !64, !alias.scope !23, !noalias !85
  br label %bb73

bb196:                                            ; preds = %bb40
  unreachable

bb199:                                            ; preds = %bb64
  unreachable

bb202:                                            ; preds = %bb67
  unreachable

bb205:                                            ; preds = %bb82
  unreachable

bb208:                                            ; preds = %bb106
  unreachable

bb211:                                            ; preds = %bb109
  unreachable

bb214:                                            ; preds = %bb155
  unreachable

bb217:                                            ; preds = %bb155
  %i218 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i136, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !32, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BFloat64\7D !8, !enzyme_nocache !8
  %i219 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i17, {} addrspace(10)** noundef %i218)
  %i220 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i219, i64 %i151
  %i221 = bitcast {} addrspace(10)* addrspace(13)* %i220 to double addrspace(13)*
  %i222 = load double, double addrspace(13)* %i221, align 8, !tbaa !39, !alias.scope !23, !noalias !41
  %i223 = bitcast {} addrspace(10)* %i159 to i64 addrspace(10)*
  %i224 = addrspacecast i64 addrspace(10)* %i223 to i64 addrspace(11)*
  %i225 = getelementptr inbounds i64, i64 addrspace(11)* %i224, i64 2
  %i226 = load i64, i64 addrspace(11)* %i225, align 8, !tbaa !6, !alias.scope !9, !noalias !12
  %i227 = and i64 %i226, %i137
  %i228 = add i64 %i227, 1
  %i229 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i140, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_inactive !8, !enzyme_type !51, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BUInt8\7D !8
  %i230 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i88, {} addrspace(10)** noundef %i229)
  %i231 = bitcast {} addrspace(10)* addrspace(13)* %i230 to i8 addrspace(13)*, !enzyme_inactive !8
  %i232 = getelementptr inbounds i8, i8 addrspace(13)* %i231, i64 %i227
  %i233 = load i8, i8 addrspace(13)* %i232, align 1, !tbaa !39, !alias.scope !23, !noalias !41
  %i234 = icmp eq i8 %i233, 0
  br i1 %i234, label %bb168, label %bb161
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #21

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #21

attributes #0 = { noinline noreturn "enzyme_ta_norecur" "enzymejl_mi"="4732598432" "enzymejl_rt"="4815717520" "enzymejl_world"="26726" }
attributes #1 = { nofree memory(none) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzyme_shouldrecompute" "enzymejl_world"="26726" }
attributes #2 = { nofree memory(argmem: readwrite, inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #3 = { nofree "enzymejl_world"="26726" }
attributes #4 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #5 = { noreturn "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #6 = { cold noreturn nounwind }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #8 = { nofree norecurse nosync nounwind speculatable willreturn memory(none) "enzyme_nocache" "enzyme_shouldrecompute" "enzymejl_world"="26726" }
attributes #9 = { "enzyme_ta_norecur" "enzymejl_mi"="14418006736" "enzymejl_rt"="4752516336" "enzymejl_world"="26726" }
attributes #10 = { nofree memory(inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #11 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #12 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #13 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #14 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #15 = { nofree norecurse nounwind memory(inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #16 = { "enzyme_active" }
attributes #17 = { "enzyme_parmremove"="2" "enzyme_retremove" "enzyme_ta_norecur" "enzymejl_mi"="4485922576" "enzymejl_rt"="14415035984" "enzymejl_world"="26726" }
attributes #18 = { "enzyme_parmremove"="1" "enzyme_ta_norecur" "enzymejl_mi"="4473615312" "enzymejl_rt"="4752516336" "enzymejl_world"="26726" }
attributes #19 = { "enzyme_retremove" "enzyme_ta_norecur" "enzymejl_mi"="14400329744" "enzymejl_rt"="14415035984" "enzymejl_world"="26726" }
attributes #20 = { "enzyme_parmremove"="2" "enzyme_ta_norecur" "enzymejl_mi"="4485764240" "enzymejl_rt"="4697650384" "enzymejl_world"="26726" }
attributes #21 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #22 = { memory(readwrite) }
attributes #23 = { nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #24 = { nofree }
attributes #25 = { noreturn }
attributes #26 = { nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #27 = { nounwind memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #28 = { nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_active_val" "enzyme_no_escaping_allocation" }
attributes #29 = { nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{!3, !3, i64 0}
!3 = !{!"jtbaa_gcframe", !4, i64 0}
!4 = !{!"jtbaa", !5, i64 0}
!5 = !{!"jtbaa"}
!6 = !{!7, !7, i64 0, i64 0}
!7 = !{!"jtbaa_const", !4, i64 0}
!8 = !{}
!9 = !{!10}
!10 = !{!"jnoalias_const", !11}
!11 = !{!"jnoalias"}
!12 = !{!13, !14, !15, !16}
!13 = !{!"jnoalias_gcframe", !11}
!14 = !{!"jnoalias_stack", !11}
!15 = !{!"jnoalias_data", !11}
!16 = !{!"jnoalias_typemd", !11}
!17 = !{!"Unknown", i32 -1, !18}
!18 = !{!"Integer"}
!19 = !{!20, !20, i64 0}
!20 = !{!"jtbaa_immut", !21, i64 0}
!21 = !{!"jtbaa_value", !22, i64 0}
!22 = !{!"jtbaa_data", !4, i64 0}
!23 = !{!15}
!24 = !{!25, !13, !14, !16, !10}
!25 = distinct !{!25, !26, !"na_addr13"}
!26 = distinct !{!26, !"addr13"}
!27 = !{!28, !28, i64 0}
!28 = !{!"jtbaa_arrayptr", !29, i64 0}
!29 = !{!"jtbaa_array", !4, i64 0}
!30 = !{!16}
!31 = !{!13, !14, !15, !10}
!32 = !{!"Unknown", i32 -1, !33}
!33 = !{!"Pointer", i32 -1, !34}
!34 = !{!"Float@double"}
!35 = !{i64 16}
!36 = !{i64 8}
!37 = !{!"Unknown", i32 -1, !38}
!38 = !{!"Pointer", i32 0, !18, i32 1, !18, i32 2, !18, i32 3, !18, i32 4, !18, i32 5, !18, i32 6, !18, i32 7, !18, i32 8, !33}
!39 = !{!40, !40, i64 0}
!40 = !{!"jtbaa_arraybuf", !22, i64 0}
!41 = !{!13, !14, !16, !10}
!42 = !{!"Unknown", i32 -1, !34}
!43 = !{!4, !4, i64 0}
!44 = !{!10, !15}
!45 = !{!13, !14, !16}
!46 = !{!"Unknown", i32 -1, !47}
!47 = !{!"Pointer", i32 0, !18, i32 1, !18, i32 2, !18, i32 3, !18, i32 4, !18, i32 5, !18, i32 6, !18, i32 7, !18, i32 8, !48}
!48 = !{!"Pointer"}
!49 = !{i64 56}
!50 = !{!21, !21, i64 0}
!51 = !{!"Unknown", i32 -1, !48}
!52 = !{!"Unknown", i32 -1, !53}
!53 = !{!"Pointer", i32 0, !18, i32 1, !18, i32 2, !18, i32 3, !18, i32 4, !18, i32 5, !18, i32 6, !18, i32 7, !18, i32 8, !54}
!54 = !{!"Pointer", i32 0, !48}
!55 = !{!16, !14}
!56 = !{!13, !15, !10}
!57 = !{!25, !13, !14, !15, !10}
!58 = !{!59, !59, i64 0}
!59 = !{!"jtbaa_arraysize", !29, i64 0}
!60 = !{!61, !61, i64 0}
!61 = !{!"jtbaa_stack", !4, i64 0}
!62 = !{!14}
!63 = !{!25, !13, !15, !16, !10}
!64 = !{!65, !65, i64 0}
!65 = !{!"jtbaa_mutab", !21, i64 0}
!66 = !{!"Integer", i64 0, !"Integer", i64 8, !"Integer", i64 16}
!67 = !{!68, !68, i64 0, i64 0}
!68 = !{!"jtbaa_memorylen", !29, i64 0}
!69 = !{!"Unknown", i32 -1, !70}
!70 = !{!"Pointer", i32 0, !18, i32 1, !18, i32 2, !18, i32 3, !18, i32 4, !18, i32 5, !18, i32 6, !18, i32 7, !18, i32 8, !18}
!71 = !{!"4697650384"}
!72 = !{!13, !15, !16, !10}
!73 = !{!74, !13, !14, !16, !10}
!74 = distinct !{!74, !75, !"na_addr13"}
!75 = distinct !{!75, !"addr13"}
!76 = !{!10, !16}
!77 = !{!13, !14, !15}
!78 = !{!"Unknown", i32 -1, !54}
!79 = !{!80, !80, i64 0}
!80 = !{!"jtbaa_ptrarraybuf", !22, i64 0}
!81 = !{!82, !13, !14, !16, !10}
!82 = distinct !{!82, !83, !"na_addr13"}
!83 = distinct !{!83, !"addr13"}
!84 = !{i64 0, i64 65}
!85 = !{!86, !13, !14, !16, !10}
!86 = distinct !{!86, !87, !"na_addr13"}
!87 = distinct !{!87, !"addr13"}
!88 = !{!89, !89, i64 0, i64 0}
!89 = !{!"jtbaa_memoryptr", !29, i64 0}
!90 = !{!91}
!91 = distinct !{!91, !92, !"na_addr13"}
!92 = distinct !{!92, !"addr13"}
!93 = !{!91, !13, !14, !16, !10}