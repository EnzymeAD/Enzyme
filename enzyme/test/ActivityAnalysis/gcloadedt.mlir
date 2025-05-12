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
  %i68 = call fastcc double @1({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(64) %i103)
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

define internal fastcc void @0({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" "enzymejl_parmtype"="14415035984" "enzymejl_parmtype_ref"="2" %arg, double "enzyme_type"="{[-1]:Float@double}" "enzymejl_parmtype"="4752516336" "enzymejl_parmtype_ref"="0" %arg1) unnamed_addr #15 {
bb:
  %i = alloca { i64, i8 }, align 8, !enzyme_type !62, !enzymejl_byref_MUT_REF !22, !enzyme_inactive !22, !enzymejl_allocart !64, !enzymejl_source_type_Tuple\7BInt64\2C\20UInt8\7D !22
  %i2 = call {}*** @julia.get_pgcstack()
  %i3 = getelementptr inbounds {}**, {}*** %i2, i64 2
  %i4 = bitcast {}*** %i3 to i64***
  %i5 = load i64**, i64*** %i4, align 8, !tbaa !2
  %i6 = getelementptr inbounds i64*, i64** %i5, i64 2
  %i7 = load i64*, i64** %i6, align 8, !tbaa !6
  %i8 = getelementptr inbounds { i64, i8 }, { i64, i8 }* %i, i64 0, i32 0
  %i9 = load i64, i64* %i8, align 8, !tbaa !65, !alias.scope !67, !noalias !68, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i10 = icmp slt i64 %i9, 1
  br i1 %i10, label %bb38, label %bb11

bb11:                                             ; preds = %bb
  %i12 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i13 = addrspacecast i8 addrspace(10)* %i12 to i8 addrspace(11)*
  %i14 = getelementptr inbounds i8, i8 addrspace(11)* %i13, i64 40
  %i15 = bitcast i8 addrspace(11)* %i14 to i64 addrspace(11)*
  %i16 = load i64, i64 addrspace(11)* %i15, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_UInt64 !22
  %i17 = add i64 %i16, 1
  store i64 %i17, i64 addrspace(11)* %i15, align 8, !tbaa !57, !alias.scope !31, !noalias !69
  %i18 = getelementptr inbounds i8, i8 addrspace(11)* %i13, i64 8
  %i19 = bitcast i8 addrspace(11)* %i18 to {} addrspace(10)* addrspace(11)*
  %i20 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i19 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !45, !enzymejl_byref_MUT_REF !22, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BSymbol\7D !22
  %i21 = bitcast {} addrspace(10)* %i20 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i22 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i21 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i23 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i22, i64 0, i32 1
  %i24 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i23, align 8, !tbaa !34, !alias.scope !72, !noalias !73, !nonnull !22, !enzyme_type !74, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BSymbol\7D !22
  %i25 = add nsw i64 %i9, -1
  %i26 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i20, {} addrspace(10)** noundef %i24)
  %i27 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i26, i64 %i25
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 4336030960 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(13)* %i27 release, align 8, !tbaa !75, !alias.scope !31, !noalias !69
  %i28 = getelementptr inbounds i8, i8 addrspace(11)* %i13, i64 16
  %i29 = bitcast i8 addrspace(11)* %i28 to {} addrspace(10)* addrspace(11)*
  %i30 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i29 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_MUT_REF !22
  %i31 = bitcast {} addrspace(10)* %i30 to { i64, {} addrspace(10)** } addrspace(10)*
  %i32 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i31 to { i64, {} addrspace(10)** } addrspace(11)*
  %i33 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i32, i64 0, i32 1
  %i34 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i33, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22
  %i35 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i30, {} addrspace(10)** noundef %i34)
  %i36 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i35, i64 %i25
  %i37 = bitcast {} addrspace(10)* addrspace(13)* %i36 to double addrspace(13)*
  store double %arg1, double addrspace(13)* %i37, align 8, !tbaa !28, !alias.scope !31, !noalias !69
  br label %bb108

bb38:                                             ; preds = %bb
  %i39 = getelementptr inbounds { i64, i8 }, { i64, i8 }* %i, i64 0, i32 1
  %i40 = sub i64 0, %i9
  %i41 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i42 = addrspacecast i8 addrspace(10)* %i41 to i8 addrspace(11)*
  %i43 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 24
  %i44 = bitcast i8 addrspace(11)* %i43 to i64 addrspace(11)*
  %i45 = load i64, i64 addrspace(11)* %i44, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i46 = bitcast {} addrspace(10)* %arg to {} addrspace(10)* addrspace(10)*
  %i47 = addrspacecast {} addrspace(10)* addrspace(10)* %i46 to {} addrspace(10)* addrspace(11)*
  %i48 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i47 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !enzyme_type !37, !enzymejl_byref_MUT_REF !22, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BUInt8\7D !22
  %i49 = bitcast {} addrspace(10)* %i48 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i50 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i49 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i51 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i50, i64 0, i32 1
  %i52 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i51, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BUInt8\7D !22
  %i53 = xor i64 %i9, -1
  %i54 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i48, {} addrspace(10)** noundef %i52)
  %i55 = bitcast {} addrspace(10)* addrspace(13)* %i54 to i8 addrspace(13)*, !enzyme_inactive !22
  %i56 = getelementptr inbounds i8, i8 addrspace(13)* %i55, i64 %i53
  %i57 = load i8, i8 addrspace(13)* %i56, align 1, !tbaa !28, !alias.scope !31, !noalias !32
  %i58 = icmp eq i8 %i57, 127
  %i59 = sext i1 %i58 to i64
  %i60 = add i64 %i45, %i59
  store i64 %i60, i64 addrspace(11)* %i44, align 8, !tbaa !57, !alias.scope !31, !noalias !69
  %i61 = load i8, i8* %i39, align 8, !tbaa !65, !alias.scope !67, !noalias !68, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_UInt8 !22
  store i8 %i61, i8 addrspace(13)* %i56, align 1, !tbaa !28, !alias.scope !31, !noalias !69
  %i62 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 8
  %i63 = bitcast i8 addrspace(11)* %i62 to {} addrspace(10)* addrspace(11)*
  %i64 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i63 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !45, !enzymejl_byref_MUT_REF !22, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BSymbol\7D !22
  %i65 = bitcast {} addrspace(10)* %i64 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i66 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i65 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i67 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i66, i64 0, i32 1
  %i68 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i67, align 8, !tbaa !34, !alias.scope !72, !noalias !73, !nonnull !22, !enzyme_type !74, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BSymbol\7D !22
  %i69 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i64, {} addrspace(10)** noundef %i68)
  %i70 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i69, i64 %i53
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 4336030960 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(13)* %i70 release, align 8, !tbaa !75, !alias.scope !31, !noalias !69
  %i71 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 16
  %i72 = bitcast i8 addrspace(11)* %i71 to {} addrspace(10)* addrspace(11)*
  %i73 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i72 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_MUT_REF !22
  %i74 = bitcast {} addrspace(10)* %i73 to { i64, {} addrspace(10)** } addrspace(10)*
  %i75 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i74 to { i64, {} addrspace(10)** } addrspace(11)*
  %i76 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i75, i64 0, i32 1
  %i77 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i76, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22
  %i78 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i73, {} addrspace(10)** noundef %i77)
  %i79 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i78, i64 %i53
  %i80 = bitcast {} addrspace(10)* addrspace(13)* %i79 to double addrspace(13)*
  store double %arg1, double addrspace(13)* %i80, align 8, !tbaa !28, !alias.scope !31, !noalias !69
  %i81 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 32
  %i82 = bitcast i8 addrspace(11)* %i81 to i64 addrspace(11)*
  %i83 = load i64, i64 addrspace(11)* %i82, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i84 = add i64 %i83, 1
  store i64 %i84, i64 addrspace(11)* %i82, align 8, !tbaa !57, !alias.scope !31, !noalias !69
  %i85 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 40
  %i86 = bitcast i8 addrspace(11)* %i85 to i64 addrspace(11)*
  %i87 = load i64, i64 addrspace(11)* %i86, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_UInt64 !22
  %i88 = add i64 %i87, 1
  store i64 %i88, i64 addrspace(11)* %i86, align 8, !tbaa !57, !alias.scope !31, !noalias !69
  %i89 = getelementptr inbounds i8, i8 addrspace(11)* %i42, i64 48
  %i90 = bitcast i8 addrspace(11)* %i89 to i64 addrspace(11)*
  %i91 = load i64, i64 addrspace(11)* %i90, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i92 = icmp sgt i64 %i91, %i40
  br i1 %i92, label %bb93, label %bb94

bb93:                                             ; preds = %bb38
  store i64 %i40, i64 addrspace(11)* %i90, align 8, !tbaa !57, !alias.scope !31, !noalias !69
  br label %bb94

bb94:                                             ; preds = %bb93, %bb38
  %i95 = add i64 %i84, %i60
  %i96 = mul i64 %i95, 3
  %i97 = bitcast {} addrspace(10)* %i64 to i64 addrspace(10)*, !enzyme_inactive !22
  %i98 = addrspacecast i64 addrspace(10)* %i97 to i64 addrspace(11)*, !enzyme_inactive !22
  %i99 = load i64, i64 addrspace(11)* %i98, align 8, !tbaa !60, !alias.scope !11, !noalias !14, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i100 = shl i64 %i99, 1
  %i101 = icmp slt i64 %i100, %i96
  br i1 %i101, label %bb102, label %bb108

bb102:                                            ; preds = %bb94
  %i103 = icmp slt i64 %i84, 64001
  %i104 = shl i64 %i84, 1
  %i105 = shl i64 %i84, 2
  %i106 = call i64 @llvm.smax.i64(i64 %i105, i64 noundef 4)
  %i107 = select i1 %i103, i64 %i106, i64 %i104
  call fastcc void @2({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) %arg, i64 signext %i107)
  br label %bb108

bb108:                                            ; preds = %bb102, %bb94, %bb11
  ret void
}

define internal fastcc "enzyme_type"="{[-1]:Float@double}" double @1({} addrspace(10)* nocapture noundef nonnull readonly align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" "enzymejl_parmtype"="14415035984" "enzymejl_parmtype_ref"="2" %arg) unnamed_addr #16 {
bb:
  %i = call {}*** @julia.get_pgcstack()
  %i1 = getelementptr inbounds {}**, {}*** %i, i64 -14
  %i2 = bitcast {}*** %i1 to {}*
  %i3 = getelementptr inbounds {}**, {}*** %i, i64 2
  %i4 = bitcast {}*** %i3 to i64***
  %i5 = load i64**, i64*** %i4, align 8, !tbaa !2
  %i6 = getelementptr inbounds i64*, i64** %i5, i64 2
  %i7 = load i64*, i64** %i6, align 8, !tbaa !6
  %i8 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i9 = addrspacecast i8 addrspace(10)* %i8 to i8 addrspace(11)*
  %i10 = getelementptr inbounds i8, i8 addrspace(11)* %i9, i64 32
  %i11 = bitcast i8 addrspace(11)* %i10 to i64 addrspace(11)*
  %i12 = load i64, i64 addrspace(11)* %i11, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i13 = icmp eq i64 %i12, 0
  br i1 %i13, label %bb69, label %bb14

bb14:                                             ; preds = %bb
  %i15 = getelementptr inbounds i8, i8 addrspace(11)* %i9, i64 8
  %i16 = bitcast i8 addrspace(11)* %i15 to {} addrspace(10)* addrspace(11)*
  %i17 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i16 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !45, !enzymejl_byref_MUT_REF !22, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BSymbol\7D !22
  %i18 = getelementptr inbounds i8, i8 addrspace(11)* %i9, i64 56
  %i19 = bitcast i8 addrspace(11)* %i18 to i64 addrspace(11)*
  %i20 = load i64, i64 addrspace(11)* %i19, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i21 = bitcast {} addrspace(10)* %i17 to i64 addrspace(10)*, !enzyme_inactive !22
  %i22 = addrspacecast i64 addrspace(10)* %i21 to i64 addrspace(11)*, !enzyme_inactive !22
  %i23 = load i64, i64 addrspace(11)* %i22, align 8, !tbaa !60, !alias.scope !11, !noalias !14, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i24 = icmp slt i64 %i20, %i23
  br i1 %i24, label %bb25, label %bb62

bb25:                                             ; preds = %bb14
  %i26 = load i64, i64 addrspace(11)* getelementptr inbounds (i64, i64 addrspace(11)* addrspacecast (i64* inttoptr (i64 4336030960 to i64*) to i64 addrspace(11)*), i64 2), align 16, !tbaa !6, !alias.scope !43, !noalias !44
  %i27 = add nsw i64 %i23, -1
  %i28 = lshr i64 %i26, 57
  %i29 = trunc i64 %i28 to i8
  %i30 = or i8 %i29, -128
  %i31 = bitcast {} addrspace(10)* %arg to {} addrspace(10)* addrspace(10)*
  %i32 = addrspacecast {} addrspace(10)* addrspace(10)* %i31 to {} addrspace(10)* addrspace(11)*
  %i33 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i32 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !37, !enzymejl_byref_MUT_REF !22, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BUInt8\7D !22
  %i34 = bitcast {} addrspace(10)* %i33 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i35 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i34 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i36 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i35, i64 0, i32 1
  %i37 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i36, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BUInt8\7D !22
  %i38 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i33, {} addrspace(10)** noundef %i37)
  %i39 = bitcast {} addrspace(10)* addrspace(13)* %i38 to i8 addrspace(13)*, !enzyme_inactive !22
  %i40 = bitcast {} addrspace(10)* %i17 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i41 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i40 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i42 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i41, i64 0, i32 1
  %i43 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i42, align 8, !enzyme_type !74, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BSymbol\7D !22
  %i44 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i17, {} addrspace(10)** noundef %i43)
  br label %bb45

bb45:                                             ; preds = %bb59, %bb25
  %i46 = phi i64 [ %i26, %bb25 ], [ %i49, %bb59 ]
  %i47 = phi i64 [ 0, %bb25 ], [ %i60, %bb59 ]
  %i48 = and i64 %i46, %i27
  %i49 = add i64 %i48, 1
  %i50 = getelementptr inbounds i8, i8 addrspace(13)* %i39, i64 %i48
  %i51 = load i8, i8 addrspace(13)* %i50, align 1, !tbaa !28, !alias.scope !31, !noalias !32
  %i52 = icmp eq i8 %i51, 0
  br i1 %i52, label %bb69, label %bb53

bb53:                                             ; preds = %bb45
  %i54 = icmp eq i8 %i30, %i51
  br i1 %i54, label %bb55, label %bb59

bb55:                                             ; preds = %bb53
  %i56 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i44, i64 %i48
  %i57 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i56 unordered, align 8, !tbaa !75, !alias.scope !31, !noalias !32, !enzyme_type !42
  %i58 = icmp eq {} addrspace(10)* %i57, null
  br i1 %i58, label %bb86, label %bb87

bb59:                                             ; preds = %bb87, %bb53
  %i60 = add i64 %i47, 1
  %i61 = icmp slt i64 %i20, %i60
  br i1 %i61, label %bb69, label %bb45

bb62:                                             ; preds = %bb14
  %i63 = call noalias nonnull align 8 dereferenceable(8) "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,-1]:Pointer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i2, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4739065968 to {}*) to {} addrspace(10)*)) #20
  %i64 = bitcast {} addrspace(10)* %i63 to [1 x {} addrspace(10)*] addrspace(10)*, !enzyme_inactive !22
  %i65 = getelementptr [1 x {} addrspace(10)*], [1 x {} addrspace(10)*] addrspace(10)* %i64, i64 0, i64 0
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 4815715632 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %i65, align 8, !tbaa !77, !alias.scope !31, !noalias !79
  %i66 = addrspacecast {} addrspace(10)* %i63 to {} addrspace(12)*, !enzyme_inactive !22
  call void @ijl_throw({} addrspace(12)* %i66) #23
  unreachable

bb67:                                             ; preds = %bb87
  %i68 = icmp sgt i64 %i49, -1
  br i1 %i68, label %bb74, label %bb69

bb69:                                             ; preds = %bb67, %bb59, %bb45, %bb
  %i70 = call noalias nonnull align 8 dereferenceable(8) "enzyme_type"="{[-1]:Pointer, [-1,-1]:Pointer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i2, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4735480912 to {}*) to {} addrspace(10)*)) #20
  %i71 = bitcast {} addrspace(10)* %i70 to [1 x {} addrspace(10)*] addrspace(10)*
  %i72 = getelementptr [1 x {} addrspace(10)*], [1 x {} addrspace(10)*] addrspace(10)* %i71, i64 0, i64 0
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 4336030960 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %i72, align 8, !tbaa !77, !alias.scope !31, !noalias !79
  %i73 = addrspacecast {} addrspace(10)* %i70 to {} addrspace(12)*
  call void @ijl_throw({} addrspace(12)* %i73) #23
  unreachable

bb74:                                             ; preds = %bb67
  %i75 = getelementptr inbounds i8, i8 addrspace(11)* %i9, i64 16
  %i76 = bitcast i8 addrspace(11)* %i75 to {} addrspace(10)* addrspace(11)*
  %i77 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i76 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_MUT_REF !22
  %i78 = bitcast {} addrspace(10)* %i77 to { i64, {} addrspace(10)** } addrspace(10)*
  %i79 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i78 to { i64, {} addrspace(10)** } addrspace(11)*
  %i80 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i79, i64 0, i32 1
  %i81 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i80, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22
  %i82 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i77, {} addrspace(10)** noundef %i81)
  %i83 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i82, i64 %i48
  %i84 = bitcast {} addrspace(10)* addrspace(13)* %i83 to double addrspace(13)*
  %i85 = load double, double addrspace(13)* %i84, align 8, !tbaa !28, !alias.scope !31, !noalias !32
  ret double %i85

bb86:                                             ; preds = %bb55
  unreachable

bb87:                                             ; preds = %bb55
  %i88 = addrspacecast {} addrspace(10)* %i57 to {} addrspace(11)*
  %i89 = icmp eq {} addrspace(11)* %i88, addrspacecast ({}* inttoptr (i64 4336030960 to {}*) to {} addrspace(11)*)
  br i1 %i89, label %bb67, label %bb59
}

define internal fastcc void @2({} addrspace(10)* noundef nonnull align 8 dereferenceable(64) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Integer, [-1,0,1]:Integer, [-1,0,2]:Integer, [-1,0,3]:Integer, [-1,0,4]:Integer, [-1,0,5]:Integer, [-1,0,6]:Integer, [-1,0,7]:Integer, [-1,0,8]:Pointer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,0]:Pointer, [-1,16]:Pointer, [-1,16,0]:Integer, [-1,16,1]:Integer, [-1,16,2]:Integer, [-1,16,3]:Integer, [-1,16,4]:Integer, [-1,16,5]:Integer, [-1,16,6]:Integer, [-1,16,7]:Integer, [-1,16,8]:Pointer, [-1,16,8,-1]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [-1,40]:Integer, [-1,41]:Integer, [-1,42]:Integer, [-1,43]:Integer, [-1,44]:Integer, [-1,45]:Integer, [-1,46]:Integer, [-1,47]:Integer, [-1,48]:Integer, [-1,49]:Integer, [-1,50]:Integer, [-1,51]:Integer, [-1,52]:Integer, [-1,53]:Integer, [-1,54]:Integer, [-1,55]:Integer, [-1,56]:Integer, [-1,57]:Integer, [-1,58]:Integer, [-1,59]:Integer, [-1,60]:Integer, [-1,61]:Integer, [-1,62]:Integer, [-1,63]:Integer}" "enzymejl_parmtype"="14415035984" "enzymejl_parmtype_ref"="2" %arg, i64 signext "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" "enzymejl_parmtype"="4752517136" "enzymejl_parmtype_ref"="0" %arg1) unnamed_addr #17 {
bb:
  %i = call {}*** @julia.get_pgcstack()
  %i2 = getelementptr inbounds {}**, {}*** %i, i64 2
  %i3 = bitcast {}*** %i2 to i64***
  %i4 = load i64**, i64*** %i3, align 8, !tbaa !2
  %i5 = getelementptr inbounds i64*, i64** %i4, i64 2
  %i6 = load i64*, i64** %i5, align 8, !tbaa !6
  %i7 = bitcast {} addrspace(10)* %arg to {} addrspace(10)* addrspace(10)*
  %i8 = addrspacecast {} addrspace(10)* addrspace(10)* %i7 to {} addrspace(10)* addrspace(11)*
  %i9 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i8 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !37, !enzymejl_byref_MUT_REF !22, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BUInt8\7D !22
  %i10 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i11 = addrspacecast i8 addrspace(10)* %i10 to i8 addrspace(11)*
  %i12 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 8
  %i13 = bitcast i8 addrspace(11)* %i12 to {} addrspace(10)* addrspace(11)*
  %i14 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i13 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !45, !enzymejl_byref_MUT_REF !22, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BSymbol\7D !22
  %i15 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 16
  %i16 = bitcast i8 addrspace(11)* %i15 to {} addrspace(10)* addrspace(11)*
  %i17 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i16 unordered, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !nonnull !22, !dereferenceable !23, !align !24, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_MUT_REF !22
  %i18 = icmp sgt i64 %arg1, 15
  br i1 %i18, label %bb19, label %bb26

bb19:                                             ; preds = %bb
  %i20 = add nsw i64 %arg1, -1
  %i21 = call i64 @llvm.ctlz.i64(i64 %i20, i1 noundef false), !range !82
  %i22 = sub nuw nsw i64 64, %i21
  %i23 = shl nuw i64 1, %i22
  %i24 = icmp eq i64 %i21, 0
  %i25 = select i1 %i24, i64 0, i64 %i23
  br label %bb26

bb26:                                             ; preds = %bb19, %bb
  %i27 = phi i64 [ %i25, %bb19 ], [ 16, %bb ]
  %i28 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 40
  %i29 = bitcast i8 addrspace(11)* %i28 to i64 addrspace(11)*
  %i30 = load i64, i64 addrspace(11)* %i29, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_UInt64 !22
  %i31 = add i64 %i30, 1
  store i64 %i31, i64 addrspace(11)* %i29, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  %i32 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 48
  %i33 = bitcast i8 addrspace(11)* %i32 to i64 addrspace(11)*
  store i64 1, i64 addrspace(11)* %i33, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  %i34 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 32
  %i35 = bitcast i8 addrspace(11)* %i34 to i64 addrspace(11)*
  %i36 = load i64, i64 addrspace(11)* %i35, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i37 = icmp eq i64 %i36, 0
  br i1 %i37, label %bb38, label %bb76

bb38:                                             ; preds = %bb26
  %i39 = icmp eq i64 %i27, 0
  br i1 %i39, label %bb40, label %bb43

bb40:                                             ; preds = %bb38
  %i41 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4752515456 to {} addrspace(10)**) unordered, align 128, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !37, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BUInt8\7D !22, !enzymejl_byref_BITS_REF !22
  %i42 = icmp eq {} addrspace(10)* %i41, null
  br i1 %i42, label %bb188, label %bb45

bb43:                                             ; preds = %bb38
  %i44 = call noalias "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4752515424 to {}*) to {} addrspace(10)*), i64 %i27) #19
  br label %bb45

bb45:                                             ; preds = %bb43, %bb40
  %i46 = phi {} addrspace(10)* [ %i44, %bb43 ], [ %i41, %bb40 ], !enzyme_inactive !22
  store atomic {} addrspace(10)* %i46, {} addrspace(10)* addrspace(11)* %i8 release, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i46) #24
  %i47 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* nonnull %i46)
  %i48 = bitcast {} addrspace(10)* %i46 to i64 addrspace(10)*, !enzyme_inactive !22
  %i49 = addrspacecast i64 addrspace(10)* %i48 to i64 addrspace(11)*, !enzyme_inactive !22
  %i50 = load i64, i64 addrspace(11)* %i49, align 8, !tbaa !60, !alias.scope !11, !noalias !14, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i51 = icmp sgt i64 %i50, -1
  br i1 %i51, label %bb53, label %bb52

bb52:                                             ; preds = %bb45
  unreachable

bb53:                                             ; preds = %bb45
  %i54 = bitcast {} addrspace(10)* %i46 to i8 addrspace(10)*, !enzyme_inactive !22
  %i55 = addrspacecast i8 addrspace(10)* %i54 to i8 addrspace(11)*, !enzyme_inactive !22
  %i56 = getelementptr inbounds i8, i8 addrspace(11)* %i55, i64 8
  %i57 = bitcast i8 addrspace(11)* %i56 to i64 addrspace(11)*
  %i58 = load i64, i64 addrspace(11)* %i57, align 8, !tbaa !86, !alias.scope !11, !noalias !14, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BUInt8\7D !22
  %i59 = inttoptr i64 %i58 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 1 %i59, i8 noundef 0, i64 %i50, i1 noundef false)
  call void @llvm.julia.gc_preserve_end(token %i47)
  br i1 %i39, label %bb60, label %bb66

bb60:                                             ; preds = %bb53
  %i61 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4696681696 to {} addrspace(10)**) unordered, align 32, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !45, !enzyme_inactive !22, !enzymejl_byref_BITS_REF !22, !enzymejl_source_type_Memory\7BSymbol\7D !22
  %i62 = icmp eq {} addrspace(10)* %i61, null
  br i1 %i62, label %bb189, label %bb63

bb63:                                             ; preds = %bb60
  store atomic {} addrspace(10)* %i61, {} addrspace(10)* addrspace(11)* %i13 release, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i61) #24
  %i64 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_BITS_REF !22
  %i65 = icmp eq {} addrspace(10)* %i64, null
  br i1 %i65, label %bb190, label %bb70

bb66:                                             ; preds = %bb53
  %i67 = call noalias "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,0]:Pointer}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4696681664 to {}*) to {} addrspace(10)*), i64 %i27) #19
  store atomic {} addrspace(10)* %i67, {} addrspace(10)* addrspace(11)* %i13 release, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i67) #24
  %i68 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4697777888 to {}*) to {} addrspace(10)*), i64 %i27) #19
  br label %bb70

bb69:                                             ; preds = %bb180, %bb70
  ret void

bb70:                                             ; preds = %bb66, %bb63
  %i71 = phi {} addrspace(10)* [ %i68, %bb66 ], [ %i64, %bb63 ]
  store atomic {} addrspace(10)* %i71, {} addrspace(10)* addrspace(11)* %i16 release, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i71) #24
  %i72 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 24
  %i73 = bitcast i8 addrspace(11)* %i72 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i73, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  %i74 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 56
  %i75 = bitcast i8 addrspace(11)* %i74 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i75, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  br label %bb69

bb76:                                             ; preds = %bb26
  %i77 = icmp eq i64 %i27, 0
  br i1 %i77, label %bb78, label %bb81

bb78:                                             ; preds = %bb76
  %i79 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4752515456 to {} addrspace(10)**) unordered, align 128, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !37, !enzyme_inactive !22, !enzymejl_source_type_Memory\7BUInt8\7D !22, !enzymejl_byref_BITS_REF !22
  %i80 = icmp eq {} addrspace(10)* %i79, null
  br i1 %i80, label %bb191, label %bb83

bb81:                                             ; preds = %bb76
  %i82 = call noalias "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4752515424 to {}*) to {} addrspace(10)*), i64 %i27) #19
  br label %bb83

bb83:                                             ; preds = %bb81, %bb78
  %i84 = phi {} addrspace(10)* [ %i82, %bb81 ], [ %i79, %bb78 ], !enzyme_inactive !22
  %i85 = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* nonnull %i84)
  %i86 = bitcast {} addrspace(10)* %i84 to i64 addrspace(10)*, !enzyme_inactive !22
  %i87 = addrspacecast i64 addrspace(10)* %i86 to i64 addrspace(11)*, !enzyme_inactive !22
  %i88 = load i64, i64 addrspace(11)* %i87, align 8, !tbaa !60, !alias.scope !11, !noalias !14, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i89 = icmp sgt i64 %i88, -1
  br i1 %i89, label %bb91, label %bb90

bb90:                                             ; preds = %bb83
  unreachable

bb91:                                             ; preds = %bb83
  %i92 = bitcast {} addrspace(10)* %i84 to i8 addrspace(10)*, !enzyme_inactive !22
  %i93 = addrspacecast i8 addrspace(10)* %i92 to i8 addrspace(11)*, !enzyme_inactive !22
  %i94 = getelementptr inbounds i8, i8 addrspace(11)* %i93, i64 8
  %i95 = bitcast i8 addrspace(11)* %i94 to i64 addrspace(11)*
  %i96 = load i64, i64 addrspace(11)* %i95, align 8, !tbaa !86, !alias.scope !11, !noalias !14, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BUInt8\7D !22
  %i97 = inttoptr i64 %i96 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 1 %i97, i8 noundef 0, i64 %i88, i1 noundef false)
  call void @llvm.julia.gc_preserve_end(token %i85)
  br i1 %i77, label %bb98, label %bb104

bb98:                                             ; preds = %bb91
  %i99 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4696681696 to {} addrspace(10)**) unordered, align 32, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !45, !enzyme_inactive !22, !enzymejl_byref_BITS_REF !22, !enzymejl_source_type_Memory\7BSymbol\7D !22
  %i100 = icmp eq {} addrspace(10)* %i99, null
  br i1 %i100, label %bb192, label %bb101

bb101:                                            ; preds = %bb98
  %i102 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4697777920 to {} addrspace(10)**) unordered, align 256, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !25, !enzymejl_source_type_Memory\7BFloat64\7D !22, !enzymejl_byref_BITS_REF !22
  %i103 = icmp eq {} addrspace(10)* %i102, null
  br i1 %i103, label %bb193, label %bb107

bb104:                                            ; preds = %bb91
  %i105 = call noalias "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,0]:Pointer}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4696681664 to {}*) to {} addrspace(10)*), i64 %i27) #19
  %i106 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4697777888 to {}*) to {} addrspace(10)*), i64 %i27) #19
  br label %bb107

bb107:                                            ; preds = %bb104, %bb101
  %i108 = phi {} addrspace(10)* [ %i105, %bb104 ], [ %i99, %bb101 ], !enzyme_inactive !22
  %i109 = phi {} addrspace(10)* [ %i106, %bb104 ], [ %i102, %bb101 ]
  %i110 = load i64, i64 addrspace(11)* %i29, align 8, !tbaa !57, !alias.scope !31, !noalias !32, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_UInt64 !22
  %i111 = bitcast {} addrspace(10)* %i9 to i64 addrspace(10)*, !enzyme_inactive !22
  %i112 = addrspacecast i64 addrspace(10)* %i111 to i64 addrspace(11)*, !enzyme_inactive !22
  %i113 = load i64, i64 addrspace(11)* %i112, align 8, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !22, !enzyme_inactive !22, !enzymejl_source_type_Int64 !22
  %i114 = call i64 @llvm.smax.i64(i64 %i113, i64 noundef 0)
  %i115 = icmp slt i64 %i113, 1
  br i1 %i115, label %bb180, label %bb116

bb116:                                            ; preds = %bb107
  %i117 = bitcast {} addrspace(10)* %i9 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i118 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i117 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i119 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i118, i64 0, i32 1
  %i120 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i119, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BUInt8\7D !22
  %i121 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i9, {} addrspace(10)** noundef %i120)
  %i122 = bitcast {} addrspace(10)* addrspace(13)* %i121 to i8 addrspace(13)*, !enzyme_inactive !22
  %i123 = bitcast {} addrspace(10)* %i14 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i124 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i123 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i125 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i124, i64 0, i32 1
  %i126 = bitcast {} addrspace(10)* %i17 to { i64, {} addrspace(10)** } addrspace(10)*
  %i127 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i126 to { i64, {} addrspace(10)** } addrspace(11)*
  %i128 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i127, i64 0, i32 1
  %i129 = add i64 %i27, -1
  %i130 = bitcast {} addrspace(10)* %i84 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i131 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i130 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i132 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i131, i64 0, i32 1
  %i133 = bitcast {} addrspace(10)* %i108 to { i64, {} addrspace(10)** } addrspace(10)*, !enzyme_inactive !22
  %i134 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i133 to { i64, {} addrspace(10)** } addrspace(11)*, !enzyme_inactive !22
  %i135 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i134, i64 0, i32 1
  %i136 = bitcast {} addrspace(10)* %i109 to { i64, {} addrspace(10)** } addrspace(10)*
  %i137 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i136 to { i64, {} addrspace(10)** } addrspace(11)*
  %i138 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i137, i64 0, i32 1
  br label %bb139

bb139:                                            ; preds = %bb175, %bb116
  %i140 = phi i64 [ 1, %bb116 ], [ %i179, %bb175 ]
  %i141 = phi i64 [ 0, %bb116 ], [ %i176, %bb175 ]
  %i142 = phi i64 [ 0, %bb116 ], [ %i177, %bb175 ]
  %i143 = add nsw i64 %i140, -1
  %i144 = getelementptr inbounds i8, i8 addrspace(13)* %i122, i64 %i143
  %i145 = load i8, i8 addrspace(13)* %i144, align 1, !tbaa !28, !alias.scope !31, !noalias !32
  %i146 = icmp sgt i8 %i145, -1
  br i1 %i146, label %bb175, label %bb147

bb147:                                            ; preds = %bb139
  %i148 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i125, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !74, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BSymbol\7D !22
  %i149 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i14, {} addrspace(10)** noundef %i148)
  %i150 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i149, i64 %i143
  %i151 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i150 unordered, align 8, !tbaa !75, !alias.scope !31, !noalias !32, !enzyme_type !42
  %i152 = icmp eq {} addrspace(10)* %i151, null
  br i1 %i152, label %bb194, label %bb195

bb153:                                            ; preds = %bb195, %bb153
  %i154 = phi i64 [ %i156, %bb153 ], [ %i206, %bb195 ]
  %i155 = and i64 %i154, %i129
  %i156 = add i64 %i155, 1
  %i157 = getelementptr inbounds i8, i8 addrspace(13)* %i209, i64 %i155
  %i158 = load i8, i8 addrspace(13)* %i157, align 1, !tbaa !28, !alias.scope !31, !noalias !32
  %i159 = icmp eq i8 %i158, 0
  br i1 %i159, label %bb160, label %bb153

bb160:                                            ; preds = %bb195, %bb153
  %i161 = phi i64 [ %i206, %bb195 ], [ %i156, %bb153 ]
  %i162 = phi i64 [ %i205, %bb195 ], [ %i155, %bb153 ]
  %i163 = phi i8 addrspace(13)* [ %i210, %bb195 ], [ %i157, %bb153 ]
  %i164 = sub i64 %i161, %i206
  %i165 = and i64 %i164, %i129
  %i166 = call i64 @llvm.smax.i64(i64 %i141, i64 %i165)
  store i8 %i145, i8 addrspace(13)* %i163, align 1, !tbaa !28, !alias.scope !31, !noalias !83
  %i167 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i135, align 8, !tbaa !34, !alias.scope !72, !noalias !73, !nonnull !22, !enzyme_type !74, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BSymbol\7D !22
  %i168 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i108, {} addrspace(10)** noundef %i167)
  %i169 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i168, i64 %i162
  store atomic {} addrspace(10)* %i151, {} addrspace(10)* addrspace(13)* %i169 release, align 8, !tbaa !75, !alias.scope !31, !noalias !83
  %i170 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i138, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22
  %i171 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i109, {} addrspace(10)** noundef %i170)
  %i172 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i171, i64 %i162
  %i173 = bitcast {} addrspace(10)* addrspace(13)* %i172 to double addrspace(13)*
  store double %i200, double addrspace(13)* %i173, align 8, !tbaa !28, !alias.scope !31, !noalias !83
  %i174 = add i64 %i142, 1
  br label %bb175

bb175:                                            ; preds = %bb160, %bb139
  %i176 = phi i64 [ %i166, %bb160 ], [ %i141, %bb139 ]
  %i177 = phi i64 [ %i174, %bb160 ], [ %i142, %bb139 ]
  %i178 = icmp eq i64 %i140, %i114
  %i179 = add nuw i64 %i140, 1
  br i1 %i178, label %bb180, label %bb139

bb180:                                            ; preds = %bb175, %bb107
  %i181 = phi i64 [ 0, %bb107 ], [ %i176, %bb175 ]
  %i182 = phi i64 [ 0, %bb107 ], [ %i177, %bb175 ]
  %i183 = add i64 %i110, 1
  store i64 %i183, i64 addrspace(11)* %i29, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  store atomic {} addrspace(10)* %i84, {} addrspace(10)* addrspace(11)* %i8 release, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* %i84) #24
  store atomic {} addrspace(10)* %i108, {} addrspace(10)* addrspace(11)* %i13 release, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* nonnull %i108) #24
  store atomic {} addrspace(10)* %i109, {} addrspace(10)* addrspace(11)* %i16 release, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %arg, {} addrspace(10)* %i109) #24
  store i64 %i182, i64 addrspace(11)* %i35, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  %i184 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 24
  %i185 = bitcast i8 addrspace(11)* %i184 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i185, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  %i186 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 56
  %i187 = bitcast i8 addrspace(11)* %i186 to i64 addrspace(11)*
  store i64 %i181, i64 addrspace(11)* %i187, align 8, !tbaa !57, !alias.scope !31, !noalias !83
  br label %bb69

bb188:                                            ; preds = %bb40
  unreachable

bb189:                                            ; preds = %bb60
  unreachable

bb190:                                            ; preds = %bb63
  unreachable

bb191:                                            ; preds = %bb78
  unreachable

bb192:                                            ; preds = %bb98
  unreachable

bb193:                                            ; preds = %bb101
  unreachable

bb194:                                            ; preds = %bb147
  unreachable

bb195:                                            ; preds = %bb147
  %i196 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i128, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !nonnull !22, !enzyme_type !19, !enzymejl_byref_BITS_VALUE !22, !enzymejl_source_type_Ptr\7BFloat64\7D !22, !enzyme_nocache !22
  %i197 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i17, {} addrspace(10)** noundef %i196)
  %i198 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i197, i64 %i143
  %i199 = bitcast {} addrspace(10)* addrspace(13)* %i198 to double addrspace(13)*
  %i200 = load double, double addrspace(13)* %i199, align 8, !tbaa !28, !alias.scope !31, !noalias !32
  %i201 = bitcast {} addrspace(10)* %i151 to i64 addrspace(10)*
  %i202 = addrspacecast i64 addrspace(10)* %i201 to i64 addrspace(11)*
  %i203 = getelementptr inbounds i64, i64 addrspace(11)* %i202, i64 2
  %i204 = load i64, i64 addrspace(11)* %i203, align 8, !tbaa !6, !alias.scope !43, !noalias !44
  %i205 = and i64 %i204, %i129
  %i206 = add i64 %i205, 1
  %i207 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i132, align 8, !tbaa !6, !alias.scope !43, !noalias !44, !enzyme_type !42, !enzymejl_byref_BITS_VALUE !22, !enzyme_nocache !22, !enzyme_inactive !22, !enzymejl_source_type_Ptr\7BUInt8\7D !22
  %i208 = call "enzyme_inactive" "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i84, {} addrspace(10)** noundef %i207)
  %i209 = bitcast {} addrspace(10)* addrspace(13)* %i208 to i8 addrspace(13)*, !enzyme_inactive !22
  %i210 = getelementptr inbounds i8, i8 addrspace(13)* %i209, i64 %i205
  %i211 = load i8, i8 addrspace(13)* %i210, align 1, !tbaa !28, !alias.scope !31, !noalias !32
  %i212 = icmp eq i8 %i211, 0
  br i1 %i212, label %bb160, label %bb153
}

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