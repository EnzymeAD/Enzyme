; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=julia_f_8240 -S | FileCheck %s; fi

source_filename = "start"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"
target triple = "arm64-apple-darwin24.0.0"

declare {}*** @julia.get_pgcstack() local_unnamed_addr

declare nonnull {} addrspace(10)* @ijl_invoke({} addrspace(10)*, {} addrspace(10)** nocapture readonly, i32, {} addrspace(10)*) #1

declare nonnull {} addrspace(10)* @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) local_unnamed_addr #1

declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}*, i64, {} addrspace(10)*) local_unnamed_addr #2

declare void @ijl_throw({} addrspace(12)*) local_unnamed_addr #3

declare void @llvm.trap() #4

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

declare noundef nonnull {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* nocapture noundef nonnull readnone, {} addrspace(10)** noundef nonnull readnone) local_unnamed_addr #6

define "enzyme_type"="{[-1]:Float@double}" double @julia_f_8240({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4698642384" "enzymejl_parmtype_ref"="2" %arg, {} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(24) "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4523794896" "enzymejl_parmtype_ref"="2" %arg1) local_unnamed_addr #7 {
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
  %i10 = bitcast {} addrspace(10)* %arg to { i8*, {} addrspace(10)* } addrspace(10)*
  %i11 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i10 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i12 = bitcast {} addrspace(10)* %arg to {} addrspace(10)** addrspace(10)*
  %i13 = addrspacecast {} addrspace(10)** addrspace(10)* %i12 to {} addrspace(10)** addrspace(11)*
  %i14 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i13, align 8, !tbaa !8, !alias.scope !11, !noalias !14, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22
  %i15 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i11, i64 0, i32 1
  %i16 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i15, align 8, !tbaa !8, !alias.scope !11, !noalias !14, !dereferenceable_or_null !23, !align !24, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_MUT_REF !22
  %i17 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i16, {} addrspace(10)** noundef %i14)
  %i18 = bitcast {} addrspace(10)* addrspace(13)* %i17 to double addrspace(13)*
  %i19 = load double, double addrspace(13)* %i18, align 8, !tbaa !28, !alias.scope !31, !noalias !32, !enzyme_type !33, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Float64 !22
  %i20 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4752515456 to {} addrspace(10)**) unordered, align 128, !tbaa !34, !alias.scope !35, !noalias !36, !nonnull !22, !enzyme_type !37, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BUInt8\7D !22, !enzymejl_byref_BITS_REF !22
  %i21 = icmp eq {} addrspace(10)* %i20, null
  br i1 %i21, label %bb90, label %bb91

bb22:                                             ; preds = %bb91
  unreachable

bb23:                                             ; preds = %bb91
  %i24 = bitcast {} addrspace(10)* %i20 to i8 addrspace(10)*, !enzyme_inactive !22
  %i25 = addrspacecast i8 addrspace(10)* %i24 to i8 addrspace(11)*, !enzyme_inactive !22
  %i26 = getelementptr inbounds i8, i8 addrspace(11)* %i25, i64 8
  %i27 = bitcast i8 addrspace(11)* %i26 to i64 addrspace(11)*
  %i28 = load i64, i64 addrspace(11)* %i27, align 8, !tbaa !40, !alias.scope !31, !noalias !32, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BUInt8\7D !22
  %i29 = inttoptr i64 %i28 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 1 %i29, i8 noundef 0, i64 %i95, i1 noundef false)
  call void @llvm.julia.gc_preserve_end(token %i92)
  %i30 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4696681696 to {} addrspace(10)**) unordered, align 32, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !45, !enzyme_inactive !22, !enzymejl_byref_BITS_REF !22, !enzymejl_source_type_Memory\7BSymbol\7D !22
  %i31 = icmp eq {} addrspace(10)* %i30, null
  br i1 %i31, label %bb97, label %bb98

bb32:                                             ; preds = %bb102
  br label %bb33

bb33:                                             ; preds = %bb102, %bb32
  %i34 = bitcast {} addrspace(10)* %arg1 to i8 addrspace(10)*, !enzyme_inactive !22
  %i35 = addrspacecast i8 addrspace(10)* %i34 to i8 addrspace(11)*, !enzyme_inactive !22
  %i36 = getelementptr inbounds i8, i8 addrspace(11)* %i35, i64 16
  %i37 = bitcast i8 addrspace(11)* %i36 to i64 addrspace(11)*
  %i38 = load i64, i64 addrspace(11)* %i37, align 8, !tbaa !34, !alias.scope !48, !noalias !49, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  call void @jl_({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) %i103)
  %i39 = icmp eq i64 %i38, 0
  br i1 %i39, label %bb40, label %bb43

bb40:                                             ; preds = %bb33
  %i41 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_BITS_REF !22
  %i42 = icmp eq {} addrspace(10)* %i41, null
  br i1 %i42, label %bb121, label %bb45

bb43:                                             ; preds = %bb33
  %i44 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4697777888 to {}*) to {} addrspace(10)*), i64 %i38) #19
  br label %bb45

bb45:                                             ; preds = %bb43, %bb40
  %i46 = phi {} addrspace(10)* [ %i44, %bb43 ], [ %i41, %bb40 ]
  %i47 = bitcast {} addrspace(10)* %i46 to { i64, {} addrspace(10)** } addrspace(10)*
  %i48 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i47 to { i64, {} addrspace(10)** } addrspace(11)*
  %i49 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i48, i64 0, i32 1
  %i50 = bitcast {} addrspace(10)** addrspace(11)* %i49 to i8* addrspace(11)*
  %i51 = load i8*, i8* addrspace(11)* %i50, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22
  %i52 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i4, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4698642384 to {}*) to {} addrspace(10)*)) #20
  %i53 = bitcast {} addrspace(10)* %i52 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i54 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i53 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i55 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i54, i64 0, i32 0
  store i8* %i51, i8* addrspace(11)* %i55, align 8, !tbaa !8, !alias.scope !11, !noalias !51
  %i56 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i54, i64 0, i32 1
  store {} addrspace(10)* %i46, {} addrspace(10)* addrspace(11)* %i56, align 8, !tbaa !8, !alias.scope !11, !noalias !51
  %i57 = bitcast {} addrspace(10)* %i52 to i8 addrspace(10)*
  %i58 = addrspacecast i8 addrspace(10)* %i57 to i8 addrspace(11)*
  %i59 = getelementptr inbounds i8, i8 addrspace(11)* %i58, i64 16
  %i60 = bitcast i8 addrspace(11)* %i59 to i64 addrspace(11)*
  store i64 %i38, i64 addrspace(11)* %i60, align 8, !tbaa !54, !alias.scope !11, !noalias !51
  %i61 = icmp slt i64 %i38, 1
  %i62 = bitcast i8* %i51 to {} addrspace(10)**
  br i1 %i61, label %bb88, label %bb63

bb63:                                             ; preds = %bb45
  %i64 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i46, {} addrspace(10)** noundef %i62)
  %i65 = bitcast {} addrspace(10)* addrspace(13)* %i64 to double addrspace(13)*
  br label %bb66

bb66:                                             ; preds = %bb66, %bb63
  %i67 = phi i64 [ %i70, %bb66 ], [ 1, %bb63 ]
  %i68 = call fastcc double @a1({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(64) %i103)
  store double %i68, double addrspace(13)* %i65, align 8, !tbaa !28, !alias.scope !31, !noalias !56
  %i69 = icmp eq i64 %i67, %i38
  %i70 = add nuw i64 %i67, 1
  br i1 %i69, label %bb71, label %bb66

bb71:                                             ; preds = %bb66
  %i72 = call i64 @llvm.smax.i64(i64 %i38, i64 noundef 0)
  %i73 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i56, align 8, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_MUT_REF !22
  %i74 = add nsw i64 %i72, -1
  %i75 = icmp ugt i64 %i38, %i74
  %i76 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i73, {} addrspace(10)** noundef %i62)
  br i1 %i75, label %bb77, label %bb87

bb77:                                             ; preds = %bb77, %bb71
  %i78 = phi i64 [ %i86, %bb77 ], [ 1, %bb71 ]
  %i79 = phi double [ %i84, %bb77 ], [ 0.000000e+00, %bb71 ]
  %i80 = add nsw i64 %i78, -1
  %i81 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i76, i64 %i80
  %i82 = bitcast {} addrspace(10)* addrspace(13)* %i81 to double addrspace(13)*
  %i83 = load double, double addrspace(13)* %i82, align 8, !tbaa !28, !alias.scope !31, !noalias !32
  %i84 = fadd double %i79, %i83
  %i85 = icmp eq i64 %i78, %i72
  %i86 = add nuw i64 %i78, 1
  br i1 %i85, label %bb88, label %bb77

bb87:                                             ; preds = %bb71
  unreachable

bb88:                                             ; preds = %bb77, %bb45
  %i89 = phi double [ 0.000000e+00, %bb45 ], [ %i84, %bb77 ]
  ret double %i89

bb90:                                             ; preds = %bb
  unreachable

bb91:                                             ; preds = %bb
  %i92 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* nonnull %i20)
  %i93 = bitcast {} addrspace(10)* %i20 to i64 addrspace(10)*, !enzyme_inactive !22
  %i94 = addrspacecast i64 addrspace(10)* %i93 to i64 addrspace(11)*, !enzyme_inactive !22
  %i95 = load i64, i64 addrspace(11)* %i94, align 8, !tbaa !40, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i96 = icmp sgt i64 %i95, -1
  br i1 %i96, label %bb23, label %bb22

bb97:                                             ; preds = %bb23
  unreachable

bb98:                                             ; preds = %bb23
  %i99 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_BITS_REF !22
  %i100 = icmp eq {} addrspace(10)* %i99, null
  br i1 %i100, label %bb101, label %bb102

bb101:                                            ; preds = %bb98
  unreachable

bb102:                                            ; preds = %bb98
  %i103 = call noalias nonnull align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i4, i64 noundef 64, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 14415035984 to {}*) to {} addrspace(10)*)) #21
  %i104 = bitcast {} addrspace(10)* %i103 to {} addrspace(10)* addrspace(10)*
  %i105 = addrspacecast {} addrspace(10)* addrspace(10)* %i104 to {} addrspace(10)* addrspace(11)*
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i105, align 8, !tbaa !57, !alias.scope !31, !noalias !56
  %i106 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i105, i64 1
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i106, align 8, !tbaa !57, !alias.scope !31, !noalias !56
  %i107 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i105, i64 2
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i107, align 8, !tbaa !57, !alias.scope !31, !noalias !56
  store atomic {} addrspace(10)* %i20, {} addrspace(10)* addrspace(11)* %i105 release, align 8, !tbaa !57, !alias.scope !31, !noalias !56
  %i108 = bitcast {} addrspace(10)* %i103 to i8 addrspace(10)*
  %i109 = addrspacecast i8 addrspace(10)* %i108 to i8 addrspace(11)*
  %i110 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 8
  %i111 = bitcast i8 addrspace(11)* %i110 to {} addrspace(10)* addrspace(11)*
  store atomic {} addrspace(10)* %i30, {} addrspace(10)* addrspace(11)* %i111 release, align 8, !tbaa !57, !alias.scope !31, !noalias !56
  %i112 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 16
  %i113 = bitcast i8 addrspace(11)* %i112 to {} addrspace(10)* addrspace(11)*
  store atomic {} addrspace(10)* %i99, {} addrspace(10)* addrspace(11)* %i113 release, align 8, !tbaa !57, !alias.scope !31, !noalias !56
  %i114 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 24
  %i115 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 48
  %i116 = bitcast i8 addrspace(11)* %i115 to i64 addrspace(11)*
  call void @llvm.memset.p11i8.i64(i8 addrspace(11)* noundef align 8 dereferenceable(24) dereferenceable_or_null(40) %i114, i8 noundef 0, i64 noundef 24, i1 noundef false), !enzyme_truetype !59
  store i64 1, i64 addrspace(11)* %i116, align 8, !tbaa !57, !alias.scope !31, !noalias !56
  %i117 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 56
  %i118 = bitcast i8 addrspace(11)* %i117 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i118, align 8, !tbaa !57, !alias.scope !31, !noalias !56
  %i119 = load i64, i64 addrspace(11)* %i94, align 8, !tbaa !60, !alias.scope !11, !noalias !14, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i120 = icmp eq i64 %i119, 16
  br i1 %i120, label %bb33, label %bb32

bb121:                                            ; preds = %bb40
  unreachable
}

declare token @llvm.julia.gc_preserve_begin(...) #8

declare noalias nonnull align 8 dereferenceable(8) {} addrspace(10)* @ijl_box_int64(i64 signext) local_unnamed_addr #9

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #10

declare void @llvm.julia.gc_preserve_end(token) #8

declare noalias nonnull align 16 dereferenceable(16) {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)*, i64) local_unnamed_addr #11

declare i64 @llvm.ctlz.i64(i64, i1 immarg) #12

declare void @julia.write_barrier({} addrspace(10)* readonly, ...) local_unnamed_addr #13

declare void @jl_({} addrspace(10)*) local_unnamed_addr #14

declare i64 @llvm.smax.i64(i64, i64) #12

declare void @llvm.memset.p11i8.i64(i8 addrspace(11)* nocapture writeonly, i8, i64, i1 immarg) #10


declare fastcc "enzyme_type"="{[-1]:Float@double}" double @a1({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" "enzymejl_parmtype"="14415035984" "enzymejl_parmtype_ref"="2" %arg) unnamed_addr #16 

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #18

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #18

attributes #1 = { nofree "enzymejl_world"="26726" }
attributes #2 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #3 = { noreturn "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #4 = { cold noreturn nounwind }
attributes #6 = { nofree norecurse nosync nounwind speculatable willreturn memory(none) "enzyme_nocache" "enzyme_shouldrecompute" "enzymejl_world"="26726" }
attributes #7 = { "enzyme_ta_norecur" "enzymejl_mi"="14418006736" "enzymejl_rt"="4752516336" "enzymejl_world"="26726" }
attributes #8 = { nofree memory(inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #9 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #10 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #11 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #12 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #13 = { nofree norecurse nounwind memory(inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26726" }
attributes #14 = { "enzyme_active" }
attributes #15 = { "enzyme_parmremove"="2" "enzyme_retremove" "enzyme_ta_norecur" "enzymejl_mi"="4485922576" "enzymejl_rt"="14415035984" "enzymejl_world"="26726" }
attributes #16 = { "enzyme_parmremove"="1" "enzyme_ta_norecur" "enzymejl_mi"="4473615312" "enzymejl_rt"="4752516336" "enzymejl_world"="26726" }
attributes #17 = { "enzyme_retremove" "enzyme_ta_norecur" "enzymejl_mi"="14400329744" "enzymejl_rt"="14415035984" "enzymejl_world"="26726" }
attributes #18 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #19 = { nounwind memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #20 = { nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #21 = { nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_active_val" "enzyme_no_escaping_allocation" }
attributes #22 = { memory(readwrite) }
attributes #23 = { noreturn }
attributes #24 = { nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{!3, !3, i64 0}
!3 = !{!"jtbaa_gcframe", !4, i64 0}
!4 = !{!"jtbaa", !5, i64 0}
!5 = !{!"jtbaa"}
!6 = !{!7, !7, i64 0, i64 0}
!7 = !{!"jtbaa_const", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"jtbaa_arrayptr", !10, i64 0}
!10 = !{!"jtbaa_array", !4, i64 0}
!11 = !{!12}
!12 = !{!"jnoalias_typemd", !13}
!13 = !{!"jnoalias"}
!14 = !{!15, !16, !17, !18}
!15 = !{!"jnoalias_gcframe", !13}
!16 = !{!"jnoalias_stack", !13}
!17 = !{!"jnoalias_data", !13}
!18 = !{!"jnoalias_const", !13}
!19 = !{!"Unknown", i32 -1, !20}
!20 = !{!"Pointer", i32 -1, !21}
!21 = !{!"Float@double"}
!22 = !{}
!23 = !{i64 16}
!24 = !{i64 8}
!25 = !{!"Unknown", i32 -1, !26}
!26 = !{!"Pointer", i32 0, !27, i32 1, !27, i32 2, !27, i32 3, !27, i32 4, !27, i32 5, !27, i32 6, !27, i32 7, !27, i32 8, !20}
!27 = !{!"Integer"}
!28 = !{!29, !29, i64 0}
!29 = !{!"jtbaa_arraybuf", !30, i64 0}
!30 = !{!"jtbaa_data", !4, i64 0}
!31 = !{!17}
!32 = !{!15, !16, !12, !18}
!33 = !{!"Unknown", i32 -1, !21}
!34 = !{!4, !4, i64 0}
!35 = !{!18, !17}
!36 = !{!15, !16, !12}
!37 = !{!"Unknown", i32 -1, !38}
!38 = !{!"Pointer", i32 0, !27, i32 1, !27, i32 2, !27, i32 3, !27, i32 4, !27, i32 5, !27, i32 6, !27, i32 7, !27, i32 8, !39}
!39 = !{!"Pointer"}
!40 = !{!41, !41, i64 0}
!41 = !{!"jtbaa_value", !30, i64 0}
!42 = !{!"Unknown", i32 -1, !39}
!43 = !{!18}
!44 = !{!15, !16, !17, !12}
!45 = !{!"Unknown", i32 -1, !46}
!46 = !{!"Pointer", i32 0, !27, i32 1, !27, i32 2, !27, i32 3, !27, i32 4, !27, i32 5, !27, i32 6, !27, i32 7, !27, i32 8, !47}
!47 = !{!"Pointer", i32 0, !39}
!48 = !{!12, !16}
!49 = !{!15, !17, !18}
!50 = !{!"Unknown", i32 -1, !27}
!51 = !{!52, !15, !16, !17, !18}
!52 = distinct !{!52, !53, !"na_addr13"}
!53 = distinct !{!53, !"addr13"}
!54 = !{!55, !55, i64 0}
!55 = !{!"jtbaa_arraysize", !10, i64 0}
!56 = !{!52, !15, !16, !12, !18}
!57 = !{!58, !58, i64 0}
!58 = !{!"jtbaa_mutab", !41, i64 0}
!59 = !{!"Integer", i64 0, !"Integer", i64 8, !"Integer", i64 16}
!60 = !{!61, !61, i64 0, i64 0}
!61 = !{!"jtbaa_memorylen", !10, i64 0}
!62 = !{!"Unknown", i32 -1, !63}
!63 = !{!"Pointer", i32 0, !27, i32 1, !27, i32 2, !27, i32 3, !27, i32 4, !27, i32 5, !27, i32 6, !27, i32 7, !27, i32 8, !27}
!64 = !{!"4697650384"}
!65 = !{!66, !66, i64 0}
!66 = !{!"jtbaa_stack", !4, i64 0}
!67 = !{!16}
!68 = !{!15, !17, !12, !18}
!69 = !{!70, !15, !16, !12, !18}
!70 = distinct !{!70, !71, !"na_addr13"}
!71 = distinct !{!71, !"addr13"}
!72 = !{!18, !12}
!73 = !{!15, !16, !17}
!74 = !{!"Unknown", i32 -1, !47}
!75 = !{!76, !76, i64 0}
!76 = !{!"jtbaa_ptrarraybuf", !30, i64 0}
!77 = !{!78, !78, i64 0}
!78 = !{!"jtbaa_immut", !41, i64 0}
!79 = !{!80, !15, !16, !12, !18}
!80 = distinct !{!80, !81, !"na_addr13"}
!81 = distinct !{!81, !"addr13"}
!82 = !{i64 0, i64 65}
!83 = !{!84, !15, !16, !12, !18}
!84 = distinct !{!84, !85, !"na_addr13"}
!85 = distinct !{!85, !"addr13"}
!86 = !{!87, !87, i64 0, i64 0}
!87 = !{!"jtbaa_memoryptr", !10, i64 0}

; CHECK: {} addrspace(10)* %arg: icv:0
; CHECK: {} addrspace(10)* %arg1: icv:0
; CHECK: bb
; CHECK-NEXT:   %i = alloca [1 x i64], align 8: icv:1 ici:1
; CHECK-NEXT:   %i2 = call {}*** @julia.get_pgcstack(): icv:0 ici:1
; CHECK-NEXT:   %i3 = getelementptr inbounds {}**, {}*** %i2, i64 -14: icv:0 ici:1
; CHECK-NEXT:   %i4 = bitcast {}*** %i3 to {}*: icv:0 ici:1
; CHECK-NEXT:   %i5 = getelementptr inbounds {}**, {}*** %i2, i64 2: icv:0 ici:1
; CHECK-NEXT:   %i6 = bitcast {}*** %i5 to i64***: icv:0 ici:1
; CHECK-NEXT:   %i7 = load i64**, i64*** %i6, align 8, !tbaa !2: icv:0 ici:1
; CHECK-NEXT:   %i8 = getelementptr inbounds i64*, i64** %i7, i64 2: icv:0 ici:1
; CHECK-NEXT:   %i9 = load i64*, i64** %i8, align 8, !tbaa !6: icv:1 ici:1
; CHECK-NEXT:   %i10 = bitcast {} addrspace(10)* %arg to { i8*, {} addrspace(10)* } addrspace(10)*: icv:0 ici:1
; CHECK-NEXT:   %i11 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i10 to { i8*, {} addrspace(10)* } addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   %i12 = bitcast {} addrspace(10)* %arg to {} addrspace(10)** addrspace(10)*: icv:0 ici:1
; CHECK-NEXT:   %i13 = addrspacecast {} addrspace(10)** addrspace(10)* %i12 to {} addrspace(10)** addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   %i14 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i13, align 8, !tbaa !8, !alias.scope !11, !noalias !14, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22: icv:0 ici:1
; CHECK-NEXT:   %i15 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i11, i64 0, i32 1: icv:0 ici:1
; CHECK-NEXT:   %i16 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i15, align 8, !tbaa !8, !alias.scope !11, !noalias !14, !dereferenceable_or_null !23, !align !24, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_MUT_REF !22: icv:0 ici:1
; CHECK-NEXT:   %i17 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i16, {} addrspace(10)** noundef %i14): icv:0 ici:1
; CHECK-NEXT:   %i18 = bitcast {} addrspace(10)* addrspace(13)* %i17 to double addrspace(13)*: icv:0 ici:1
; CHECK-NEXT:   %i19 = load double, double addrspace(13)* %i18, align 8, !tbaa !28, !alias.scope !31, !noalias !32, !enzyme_type !33, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Float64 !22: icv:1 ici:1
; CHECK-NEXT:   %i20 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4752515456 to {} addrspace(10)**) unordered, align 128, !tbaa !34, !alias.scope !35, !noalias !36, !nonnull !22, !enzyme_type !37, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BUInt8\7D !22, !enzymejl_byref_BITS_REF !22: icv:1 ici:1
; CHECK-NEXT:   %i21 = icmp eq {} addrspace(10)* %i20, null: icv:1 ici:1
; CHECK-NEXT:   br i1 %i21, label %bb90, label %bb91: icv:1 ici:1
; CHECK-NEXT: bb22
; CHECK-NEXT:   unreachable: icv:1 ici:1
; CHECK-NEXT: bb23
; CHECK-NEXT:   %i24 = bitcast {} addrspace(10)* %i20 to i8 addrspace(10)*, !enzyme_inactive !22: icv:1 ici:1
; CHECK-NEXT:   %i25 = addrspacecast i8 addrspace(10)* %i24 to i8 addrspace(11)*, !enzyme_inactive !22: icv:1 ici:1
; CHECK-NEXT:   %i26 = getelementptr inbounds i8, i8 addrspace(11)* %i25, i64 8: icv:1 ici:1
; CHECK-NEXT:   %i27 = bitcast i8 addrspace(11)* %i26 to i64 addrspace(11)*: icv:1 ici:1
; CHECK-NEXT:   %i28 = load i64, i64 addrspace(11)* %i27, align 8, !tbaa !40, !alias.scope !31, !noalias !32, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BUInt8\7D !22: icv:1 ici:1
; CHECK-NEXT:   %i29 = inttoptr i64 %i28 to i8*: icv:1 ici:1
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* align 1 %i29, i8 noundef 0, i64 %i95, i1 noundef false): icv:1 ici:1
; CHECK-NEXT:   call void @llvm.julia.gc_preserve_end(token %i92): icv:1 ici:1
; CHECK-NEXT:   %i30 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4696681696 to {} addrspace(10)**) unordered, align 32, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !45, !enzyme_inactive !22, !enzymejl_byref_BITS_REF !22, !enzymejl_source_type_Memory\7BSymbol\7D !22: icv:1 ici:1
; CHECK-NEXT:   %i31 = icmp eq {} addrspace(10)* %i30, null: icv:1 ici:1
; CHECK-NEXT:   br i1 %i31, label %bb97, label %bb98: icv:1 ici:1
; CHECK-NEXT: bb32
; CHECK-NEXT:   br label %bb33: icv:1 ici:1
; CHECK-NEXT: bb33
; CHECK-NEXT:   %i34 = bitcast {} addrspace(10)* %arg1 to i8 addrspace(10)*, !enzyme_inactive !22: icv:1 ici:1
; CHECK-NEXT:   %i35 = addrspacecast i8 addrspace(10)* %i34 to i8 addrspace(11)*, !enzyme_inactive !22: icv:1 ici:1
; CHECK-NEXT:   %i36 = getelementptr inbounds i8, i8 addrspace(11)* %i35, i64 16: icv:0 ici:1
; CHECK-NEXT:   %i37 = bitcast i8 addrspace(11)* %i36 to i64 addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   %i38 = load i64, i64 addrspace(11)* %i37, align 8, !tbaa !34, !alias.scope !48, !noalias !49, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22: icv:1 ici:1
; CHECK-NEXT:   call void @jl_({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) %i103): icv:1 ici:0
; CHECK-NEXT:   %i39 = icmp eq i64 %i38, 0: icv:1 ici:1
; CHECK-NEXT:   br i1 %i39, label %bb40, label %bb43: icv:1 ici:1
; CHECK-NEXT: bb40
; CHECK-NEXT:   %i41 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_BITS_REF !22: icv:1 ici:1
; CHECK-NEXT:   %i42 = icmp eq {} addrspace(10)* %i41, null: icv:1 ici:1
; CHECK-NEXT:   br i1 %i42, label %bb121, label %bb45: icv:1 ici:1
; CHECK-NEXT: bb43
; CHECK-NEXT:   %i44 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4697777888 to {}*) to {} addrspace(10)*), i64 %i38) #16: icv:0 ici:1
; CHECK-NEXT:   br label %bb45: icv:1 ici:1
; CHECK-NEXT: bb45
; CHECK-NEXT:   %i46 = phi {} addrspace(10)* [ %i44, %bb43 ], [ %i41, %bb40 ]: icv:0 ici:1
; CHECK-NEXT:   %i47 = bitcast {} addrspace(10)* %i46 to { i64, {} addrspace(10)** } addrspace(10)*: icv:0 ici:1
; CHECK-NEXT:   %i48 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i47 to { i64, {} addrspace(10)** } addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   %i49 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i48, i64 0, i32 1: icv:0 ici:1
; CHECK-NEXT:   %i50 = bitcast {} addrspace(10)** addrspace(11)* %i49 to i8* addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   %i51 = load i8*, i8* addrspace(11)* %i50, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22: icv:0 ici:1
; CHECK-NEXT:   %i52 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i4, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4698642384 to {}*) to {} addrspace(10)*)) #17: icv:0 ici:1
; CHECK-NEXT:   %i53 = bitcast {} addrspace(10)* %i52 to { i8*, {} addrspace(10)* } addrspace(10)*: icv:0 ici:1
; CHECK-NEXT:   %i54 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i53 to { i8*, {} addrspace(10)* } addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   %i55 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i54, i64 0, i32 0: icv:0 ici:1
; CHECK-NEXT:   store i8* %i51, i8* addrspace(11)* %i55, align 8, !tbaa !8, !alias.scope !11, !noalias !51: icv:1 ici:0
; CHECK-NEXT:   %i56 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i54, i64 0, i32 1: icv:0 ici:1
; CHECK-NEXT:   store {} addrspace(10)* %i46, {} addrspace(10)* addrspace(11)* %i56, align 8, !tbaa !8, !alias.scope !11, !noalias !51: icv:1 ici:0
; CHECK-NEXT:   %i57 = bitcast {} addrspace(10)* %i52 to i8 addrspace(10)*: icv:0 ici:1
; CHECK-NEXT:   %i58 = addrspacecast i8 addrspace(10)* %i57 to i8 addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   %i59 = getelementptr inbounds i8, i8 addrspace(11)* %i58, i64 16: icv:0 ici:1
; CHECK-NEXT:   %i60 = bitcast i8 addrspace(11)* %i59 to i64 addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   store i64 %i38, i64 addrspace(11)* %i60, align 8, !tbaa !54, !alias.scope !11, !noalias !51: icv:1 ici:1
; CHECK-NEXT:   %i61 = icmp slt i64 %i38, 1: icv:1 ici:1
; CHECK-NEXT:   %i62 = bitcast i8* %i51 to {} addrspace(10)**: icv:0 ici:1
; CHECK-NEXT:   br i1 %i61, label %bb88, label %bb63: icv:1 ici:1
; CHECK-NEXT: bb63
; CHECK-NEXT:   %i64 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i46, {} addrspace(10)** noundef %i62): icv:0 ici:1
; CHECK-NEXT:   %i65 = bitcast {} addrspace(10)* addrspace(13)* %i64 to double addrspace(13)*: icv:0 ici:1
; CHECK-NEXT:   br label %bb66: icv:1 ici:1
; CHECK-NEXT: bb66
; CHECK-NEXT:   %i67 = phi i64 [ %i70, %bb66 ], [ 1, %bb63 ]: icv:1 ici:1
; CHECK-NEXT:   %i68 = call fastcc double @a1({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(64) %i103): icv:0 ici:0
; CHECK-NEXT:   store double %i68, double addrspace(13)* %i65, align 8, !tbaa !28, !alias.scope !31, !noalias !56: icv:1 ici:0
; CHECK-NEXT:   %i69 = icmp eq i64 %i67, %i38: icv:1 ici:1
; CHECK-NEXT:   %i70 = add nuw i64 %i67, 1: icv:1 ici:1
; CHECK-NEXT:   br i1 %i69, label %bb71, label %bb66: icv:1 ici:1
; CHECK-NEXT: bb71
; CHECK-NEXT:   %i72 = call i64 @llvm.smax.i64(i64 %i38, i64 noundef 0): icv:1 ici:1
; CHECK-NEXT:   %i73 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i56, align 8, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_MUT_REF !22: icv:0 ici:1
; CHECK-NEXT:   %i74 = add nsw i64 %i72, -1: icv:1 ici:1
; CHECK-NEXT:   %i75 = icmp ugt i64 %i38, %i74: icv:1 ici:1
; CHECK-NEXT:   %i76 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i73, {} addrspace(10)** noundef %i62): icv:0 ici:1
; CHECK-NEXT:   br i1 %i75, label %bb77, label %bb87: icv:1 ici:1
; CHECK-NEXT: bb77
; CHECK-NEXT:   %i78 = phi i64 [ %i86, %bb77 ], [ 1, %bb71 ]: icv:1 ici:1
; CHECK-NEXT:   %i79 = phi double [ %i84, %bb77 ], [ 0.000000e+00, %bb71 ]: icv:0 ici:0
; CHECK-NEXT:   %i80 = add nsw i64 %i78, -1: icv:1 ici:1
; CHECK-NEXT:   %i81 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i76, i64 %i80: icv:0 ici:1
; CHECK-NEXT:   %i82 = bitcast {} addrspace(10)* addrspace(13)* %i81 to double addrspace(13)*: icv:0 ici:1
; CHECK-NEXT:   %i83 = load double, double addrspace(13)* %i82, align 8, !tbaa !28, !alias.scope !31, !noalias !32: icv:0 ici:0
; CHECK-NEXT:   %i84 = fadd double %i79, %i83: icv:0 ici:0
; CHECK-NEXT:   %i85 = icmp eq i64 %i78, %i72: icv:1 ici:1
; CHECK-NEXT:   %i86 = add nuw i64 %i78, 1: icv:1 ici:1
; CHECK-NEXT:   br i1 %i85, label %bb88, label %bb77: icv:1 ici:1
; CHECK-NEXT: bb87
; CHECK-NEXT:   unreachable: icv:1 ici:1
; CHECK-NEXT: bb88
; CHECK-NEXT:   %i89 = phi double [ 0.000000e+00, %bb45 ], [ %i84, %bb77 ]: icv:0 ici:0
; CHECK-NEXT:   ret double %i89: icv:1 ici:1
; CHECK-NEXT: bb90
; CHECK-NEXT:   unreachable: icv:1 ici:1
; CHECK-NEXT: bb91
; CHECK-NEXT:   %i92 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* nonnull %i20): icv:1 ici:1
; CHECK-NEXT:   %i93 = bitcast {} addrspace(10)* %i20 to i64 addrspace(10)*, !enzyme_inactive !22: icv:1 ici:1
; CHECK-NEXT:   %i94 = addrspacecast i64 addrspace(10)* %i93 to i64 addrspace(11)*, !enzyme_inactive !22: icv:1 ici:1
; CHECK-NEXT:   %i95 = load i64, i64 addrspace(11)* %i94, align 8, !tbaa !40, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22: icv:1 ici:1
; CHECK-NEXT:   %i96 = icmp sgt i64 %i95, -1: icv:1 ici:1
; CHECK-NEXT:   br i1 %i96, label %bb23, label %bb22: icv:1 ici:1
; CHECK-NEXT: bb97
; CHECK-NEXT:   unreachable: icv:1 ici:1
; CHECK-NEXT: bb98
; CHECK-NEXT:   %i99 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_BITS_REF !22: icv:1 ici:1
; CHECK-NEXT:   %i100 = icmp eq {} addrspace(10)* %i99, null: icv:1 ici:1
; CHECK-NEXT:   br i1 %i100, label %bb101, label %bb102: icv:1 ici:1
; CHECK-NEXT: bb101
; CHECK-NEXT:   unreachable: icv:1 ici:1
; CHECK-NEXT: bb102
; CHECK-NEXT:   %i103 = call noalias nonnull align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i4, i64 noundef 64, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 14415035984 to {}*) to {} addrspace(10)*)) #18: icv:0 ici:1
; CHECK-NEXT:   %i104 = bitcast {} addrspace(10)* %i103 to {} addrspace(10)* addrspace(10)*: icv:0 ici:1
; CHECK-NEXT:   %i105 = addrspacecast {} addrspace(10)* addrspace(10)* %i104 to {} addrspace(10)* addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i105, align 8, !tbaa !57, !alias.scope !31, !noalias !56: icv:1 ici:1
; CHECK-NEXT:   %i106 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i105, i64 1: icv:0 ici:1
; CHECK-NEXT:   store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i106, align 8, !tbaa !57, !alias.scope !31, !noalias !56: icv:1 ici:1
; CHECK-NEXT:   %i107 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i105, i64 2: icv:0 ici:1
; CHECK-NEXT:   store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %i107, align 8, !tbaa !57, !alias.scope !31, !noalias !56: icv:1 ici:1
; CHECK-NEXT:   store atomic {} addrspace(10)* %i20, {} addrspace(10)* addrspace(11)* %i105 release, align 8, !tbaa !57, !alias.scope !31, !noalias !56: icv:1 ici:1
; CHECK-NEXT:   %i108 = bitcast {} addrspace(10)* %i103 to i8 addrspace(10)*: icv:0 ici:1
; CHECK-NEXT:   %i109 = addrspacecast i8 addrspace(10)* %i108 to i8 addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   %i110 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 8: icv:0 ici:1
; CHECK-NEXT:   %i111 = bitcast i8 addrspace(11)* %i110 to {} addrspace(10)* addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   store atomic {} addrspace(10)* %i30, {} addrspace(10)* addrspace(11)* %i111 release, align 8, !tbaa !57, !alias.scope !31, !noalias !56: icv:1 ici:1
; CHECK-NEXT:   %i112 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 16: icv:0 ici:1
; CHECK-NEXT:   %i113 = bitcast i8 addrspace(11)* %i112 to {} addrspace(10)* addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   store atomic {} addrspace(10)* %i99, {} addrspace(10)* addrspace(11)* %i113 release, align 8, !tbaa !57, !alias.scope !31, !noalias !56: icv:1 ici:1
; CHECK-NEXT:   %i114 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 24: icv:0 ici:1
; CHECK-NEXT:   %i115 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 48: icv:0 ici:1
; CHECK-NEXT:   %i116 = bitcast i8 addrspace(11)* %i115 to i64 addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   call void @llvm.memset.p11i8.i64(i8 addrspace(11)* noundef align 8 dereferenceable(24) dereferenceable_or_null(40) %i114, i8 noundef 0, i64 noundef 24, i1 noundef false), !enzyme_truetype !59: icv:1 ici:1
; CHECK-NEXT:   store i64 1, i64 addrspace(11)* %i116, align 8, !tbaa !57, !alias.scope !31, !noalias !56: icv:1 ici:1
; CHECK-NEXT:   %i117 = getelementptr inbounds i8, i8 addrspace(11)* %i109, i64 56: icv:0 ici:1
; CHECK-NEXT:   %i118 = bitcast i8 addrspace(11)* %i117 to i64 addrspace(11)*: icv:0 ici:1
; CHECK-NEXT:   store i64 0, i64 addrspace(11)* %i118, align 8, !tbaa !57, !alias.scope !31, !noalias !56: icv:1 ici:1
; CHECK-NEXT:   %i119 = load i64, i64 addrspace(11)* %i94, align 8, !tbaa !60, !alias.scope !11, !noalias !14, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22: icv:1 ici:1
; CHECK-NEXT:   %i120 = icmp eq i64 %i119, 16: icv:1 ici:1
; CHECK-NEXT:   br i1 %i120, label %bb33, label %bb32: icv:1 ici:1
; CHECK-NEXT: bb121
; CHECK-NEXT:   unreachable: icv:1 ici:1
