; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

source_filename = "start"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"
target triple = "arm64-apple-darwin24.0.0"

@jl_undefref_exception = external local_unnamed_addr constant {}*
@ccall_jl_n_threads_per_pool_16033 = local_unnamed_addr global void ()* null

; Function Attrs: noinline noreturn
define internal fastcc void @a0({} addrspace(10)* nofree noundef nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="5018769360" "enzymejl_parmtype_ref"="2" %arg, [1 x i64] addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(8) "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}" "enzymejl_parmtype"="5016827600" "enzymejl_parmtype_ref"="1" %arg1) unnamed_addr #0 {
bb:
  %i = call {}*** @julia.get_pgcstack()
  %i2 = getelementptr inbounds {}**, {}*** %i, i64 -14
  %i3 = bitcast {}*** %i2 to {}*
  %i4 = getelementptr inbounds {}**, {}*** %i, i64 2
  %i5 = bitcast {}*** %i4 to i64***
  %i6 = load i64**, i64*** %i5, align 8, !tbaa !2
  %i7 = getelementptr inbounds i64*, i64** %i6, i64 2
  %i8 = load i64*, i64** %i7, align 8, !tbaa !6
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %i8) #16
  fence syncscope("singlethread") seq_cst
  %i9 = call noalias nonnull align 8 dereferenceable(8) "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i3, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 5016827600 to {}*) to {} addrspace(10)*)) #17
  %i10 = getelementptr inbounds [1 x i64], [1 x i64] addrspace(11)* %arg1, i64 0, i64 0
  %i11 = bitcast {} addrspace(10)* %i9 to i64 addrspace(10)*, !enzyme_inactive !8
  %i12 = load i64, i64 addrspace(11)* %i10, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  store i64 %i12, i64 addrspace(10)* %i11, align 8, !tbaa !19, !alias.scope !23, !noalias !24
  %i13 = call nonnull "enzyme_type"="{[-1]:Pointer}" {} addrspace(10)* ({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* noundef nonnull @ijl_invoke, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 5052222256 to {}*) to {} addrspace(10)*), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 5072082592 to {}*) to {} addrspace(10)*), {} addrspace(10)* nofree nonnull %arg, {} addrspace(10)* nofree nonnull %i9) #18
  %i14 = addrspacecast {} addrspace(10)* %i13 to {} addrspace(12)*
  call void @ijl_throw({} addrspace(12)* %i14) #19
  unreachable
}

; Function Attrs: nofree memory(none)
declare {}*** @julia.get_pgcstack() local_unnamed_addr #1

; Function Attrs: nofree memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @julia.safepoint(i64*) local_unnamed_addr #2

; Function Attrs: nofree
declare nonnull {} addrspace(10)* @ijl_invoke({} addrspace(10)*, {} addrspace(10)** nocapture readonly, i32, {} addrspace(10)*) #3

; Function Attrs: nofree
declare nonnull {} addrspace(10)* @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite)
declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}*, i64, {} addrspace(10)*) local_unnamed_addr #4

; Function Attrs: noreturn
declare void @ijl_throw({} addrspace(12)*) local_unnamed_addr #5

; Function Attrs: nofree noinline speculatable memory(none)
define dso_local "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" i64 @julia_nthreads_16030() local_unnamed_addr #6 {
bb:
  %i = call {}*** @julia.get_pgcstack()
  %i1 = getelementptr inbounds {}**, {}*** %i, i64 2
  %i2 = bitcast {}*** %i1 to i64***
  %i3 = load i64**, i64*** %i2, align 8, !tbaa !2
  %i4 = getelementptr inbounds i64*, i64** %i3, i64 2
  %i5 = load i64*, i64** %i4, align 8, !tbaa !6
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %i5)
  fence syncscope("singlethread") seq_cst
  %i6 = load i64, i64* inttoptr (i64 4345551192 to i64*), align 8, !tbaa !27, !alias.scope !23, !noalias !28
  %i7 = inttoptr i64 %i6 to i32*
  %i8 = getelementptr inbounds i32, i32* %i7, i64 1
  %i9 = load i32, i32* %i8, align 1, !tbaa !27, !alias.scope !23, !noalias !28
  %i10 = sext i32 %i9 to i64
  ret i64 %i10
}

; Function Attrs: nofree norecurse nosync nounwind speculatable willreturn memory(none)
declare noundef nonnull {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* nocapture noundef nonnull readnone, {} addrspace(10)** noundef nonnull readnone) local_unnamed_addr #7

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #8

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #9

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite)
define noalias nonnull align 16 dereferenceable(16) {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* %arg, i64 %arg1) local_unnamed_addr #10 {
bb:
  ret {} addrspace(10)* null
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #9

define internal fastcc void @a1([2 x [1 x double]]* noalias nocapture nofree noundef nonnull writeonly sret([2 x [1 x double]]) align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %arg, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)* nocapture noundef nonnull readonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Pointer, [-1,16,0]:Pointer, [-1,16,0,-1]:Float@double, [-1,16,8]:Pointer, [-1,16,8,0]:Integer, [-1,16,8,1]:Integer, [-1,16,8,2]:Integer, [-1,16,8,3]:Integer, [-1,16,8,4]:Integer, [-1,16,8,5]:Integer, [-1,16,8,6]:Integer, [-1,16,8,7]:Integer, [-1,16,8,8]:Pointer, [-1,16,8,8,-1]:Float@double, [-1,16,16]:Integer, [-1,16,17]:Integer, [-1,16,18]:Integer, [-1,16,19]:Integer, [-1,16,20]:Integer, [-1,16,21]:Integer, [-1,16,22]:Integer, [-1,16,23]:Integer}" "enzymejl_parmtype"="4522976336" "enzymejl_parmtype_ref"="1" %arg1) unnamed_addr #11 {
bb:
  %i = alloca { {} addrspace(10)* }, align 8
  %i2 = alloca { {} addrspace(10)* }, align 8
  %i3 = call {}*** @julia.get_pgcstack()
  %i4 = getelementptr inbounds {}**, {}*** %i3, i64 2
  %i5 = bitcast {}*** %i4 to i64***
  %i6 = load i64**, i64*** %i5, align 8, !tbaa !2
  %i7 = getelementptr inbounds i64*, i64** %i6, i64 2
  %i8 = load i64*, i64** %i7, align 8, !tbaa !6
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %i8) #16
  fence syncscope("singlethread") seq_cst
  %i9 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)* %arg1, i64 0, i32 1
  %i10 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i9 unordered, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !dereferenceable !29, !align !30, !enzyme_type !31, !enzymejl_source_type_Vector\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8, !enzymejl_byref_MUT_REF !8
  %i11 = getelementptr inbounds { {} addrspace(10)* }, { {} addrspace(10)* }* %i, i64 0, i32 0
  store {} addrspace(10)* %i10, {} addrspace(10)** %i11, align 8, !noalias !36
  %i12 = addrspacecast { {} addrspace(10)* }* %i to { {} addrspace(10)* } addrspace(11)*
  %i13 = call fastcc nonnull {} addrspace(10)* @a4({ {} addrspace(10)* } addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(8) %i12)
  %i14 = bitcast {} addrspace(10)* %i13 to i8 addrspace(10)*
  %i15 = addrspacecast i8 addrspace(10)* %i14 to i8 addrspace(11)*
  %i16 = getelementptr inbounds i8, i8 addrspace(11)* %i15, i64 16
  %i17 = bitcast i8 addrspace(11)* %i16 to i64 addrspace(11)*
  %i18 = load i64, i64 addrspace(11)* %i17, align 8, !tbaa !37, !alias.scope !40, !noalias !41
  %i19 = icmp eq i64 %i18, 0
  br i1 %i19, label %bb89, label %bb20

bb20:                                             ; preds = %bb
  %i21 = bitcast {} addrspace(10)* %i13 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i22 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i21 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i23 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i22, i64 0, i32 1
  %i24 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i23, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !dereferenceable_or_null !44, !align !30, !enzyme_type !45
  %i25 = bitcast {} addrspace(10)* %i13 to {} addrspace(10)** addrspace(10)*
  %i26 = addrspacecast {} addrspace(10)** addrspace(10)* %i25 to {} addrspace(10)** addrspace(11)*
  %i27 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i26, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !enzyme_nocache !8
  %i28 = call {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i24, {} addrspace(10)** noundef %i27)
  %i29 = bitcast {} addrspace(10)* addrspace(13)* %i28 to double addrspace(13)*
  %i30 = load double, double addrspace(13)* %i29, align 8, !tbaa !47, !alias.scope !23, !noalias !28
  %i31 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)* %arg1, i64 0, i32 0, i64 0, i64 0, i64 0, i64 0
  %i32 = load double, double addrspace(11)* %i31, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i33 = fadd double %i30, %i32
  %i34 = icmp eq i64 %i18, 1
  br i1 %i34, label %bb45, label %bb35

bb35:                                             ; preds = %bb35, %bb20
  %i36 = phi i64 [ %i37, %bb35 ], [ 1, %bb20 ]
  %i37 = phi i64 [ %i39, %bb35 ], [ 2, %bb20 ]
  %i38 = phi double [ %i43, %bb35 ], [ %i33, %bb20 ]
  %i39 = add i64 %i37, 1
  %i40 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i28, i64 %i36
  %i41 = bitcast {} addrspace(10)* addrspace(13)* %i40 to double addrspace(13)*
  %i42 = load double, double addrspace(13)* %i41, align 8, !tbaa !47, !alias.scope !23, !noalias !28
  %i43 = fadd double %i38, %i42
  %i44 = icmp eq i64 %i37, %i18
  br i1 %i44, label %bb45, label %bb35

bb45:                                             ; preds = %bb89, %bb35, %bb20
  %i46 = phi double [ %i91, %bb89 ], [ %i33, %bb20 ], [ %i43, %bb35 ]
  %i47 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)* %arg1, i64 0, i32 0, i64 0, i64 0, i64 1
  %i48 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i9 unordered, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !dereferenceable !29, !align !30, !enzyme_type !31, !enzymejl_source_type_Vector\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8, !enzymejl_byref_MUT_REF !8
  %i49 = getelementptr inbounds { {} addrspace(10)* }, { {} addrspace(10)* }* %i2, i64 0, i32 0
  store {} addrspace(10)* %i48, {} addrspace(10)** %i49, align 8, !noalias !36
  %i50 = addrspacecast { {} addrspace(10)* }* %i2 to { {} addrspace(10)* } addrspace(11)*
  %i51 = call fastcc nonnull {} addrspace(10)* @a3({ {} addrspace(10)* } addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(8) %i50)
  %i52 = bitcast {} addrspace(10)* %i51 to i8 addrspace(10)*
  %i53 = addrspacecast i8 addrspace(10)* %i52 to i8 addrspace(11)*
  %i54 = getelementptr inbounds i8, i8 addrspace(11)* %i53, i64 16
  %i55 = bitcast i8 addrspace(11)* %i54 to i64 addrspace(11)*
  %i56 = load i64, i64 addrspace(11)* %i55, align 8, !tbaa !37, !alias.scope !40, !noalias !41
  %i57 = icmp eq i64 %i56, 0
  br i1 %i57, label %bb58, label %bb60

bb58:                                             ; preds = %bb45
  %i59 = icmp eq [1 x double] addrspace(11)* %i47, null
  br i1 %i59, label %bb85, label %bb92

bb60:                                             ; preds = %bb45
  %i61 = bitcast {} addrspace(10)* %i51 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i62 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i61 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i63 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i62, i64 0, i32 1
  %i64 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i63, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !dereferenceable_or_null !44, !align !30, !enzyme_type !45
  %i65 = bitcast {} addrspace(10)* %i51 to {} addrspace(10)** addrspace(10)*
  %i66 = addrspacecast {} addrspace(10)** addrspace(10)* %i65 to {} addrspace(10)** addrspace(11)*
  %i67 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i66, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !enzyme_nocache !8
  %i68 = call {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i64, {} addrspace(10)** noundef %i67)
  %i69 = bitcast {} addrspace(10)* addrspace(13)* %i68 to double addrspace(13)*
  %i70 = load double, double addrspace(13)* %i69, align 8, !tbaa !47, !alias.scope !23, !noalias !28
  %i71 = getelementptr inbounds [1 x double], [1 x double] addrspace(11)* %i47, i64 0, i64 0
  %i72 = load double, double addrspace(11)* %i71, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i73 = fadd double %i70, %i72
  %i74 = icmp eq i64 %i56, 1
  br i1 %i74, label %bb85, label %bb75

bb75:                                             ; preds = %bb75, %bb60
  %i76 = phi i64 [ %i77, %bb75 ], [ 1, %bb60 ]
  %i77 = phi i64 [ %i79, %bb75 ], [ 2, %bb60 ]
  %i78 = phi double [ %i83, %bb75 ], [ %i73, %bb60 ]
  %i79 = add i64 %i77, 1
  %i80 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i68, i64 %i76
  %i81 = bitcast {} addrspace(10)* addrspace(13)* %i80 to double addrspace(13)*
  %i82 = load double, double addrspace(13)* %i81, align 8, !tbaa !47, !alias.scope !23, !noalias !28
  %i83 = fadd double %i78, %i82
  %i84 = icmp eq i64 %i77, %i56
  br i1 %i84, label %bb85, label %bb75

bb85:                                             ; preds = %bb92, %bb75, %bb60, %bb58
  %i86 = phi double [ %i94, %bb92 ], [ undef, %bb58 ], [ %i73, %bb60 ], [ %i83, %bb75 ]
  %i87 = getelementptr inbounds [2 x [1 x double]], [2 x [1 x double]]* %arg, i64 0, i64 0, i64 0
  store double %i46, double* %i87, align 8, !noalias !36
  %i88 = getelementptr inbounds [2 x [1 x double]], [2 x [1 x double]]* %arg, i64 0, i64 1, i64 0
  store double %i86, double* %i88, align 8, !noalias !36
  ret void

bb89:                                             ; preds = %bb
  %i90 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)* %arg1, i64 0, i32 0, i64 0, i64 0, i64 0, i64 0
  %i91 = load double, double addrspace(11)* %i90, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  br label %bb45

bb92:                                             ; preds = %bb58
  %i93 = getelementptr inbounds [1 x double], [1 x double] addrspace(11)* %i47, i64 0, i64 0
  %i94 = load double, double addrspace(11)* %i93, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  br label %bb85
}

define internal fastcc void @a2([2 x [1 x double]]* noalias nocapture nofree noundef nonnull writeonly sret([2 x [1 x double]]) align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %arg, double "enzyme_type"="{[-1]:Float@double}" "enzymejl_parmtype"="5072643312" "enzymejl_parmtype_ref"="0" %arg1, [1 x [1 x [2 x [1 x double]]]] addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}" "enzymejl_parmtype"="4507524432" "enzymejl_parmtype_ref"="1" %arg2) unnamed_addr #12 {
bb:
  %i = alloca { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, align 8, !enzyme_type !50, !enzymejl_byref_MUT_REF !8, !enzymejl_allocart !52, !enzymejl_allocart_name !53, !enzymejl_source_type_TSVI4\7BV0\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D\2C\20AT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
  %i3 = alloca { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, align 8
  %i4 = alloca [2 x [1 x double]], align 8, !enzyme_type !54, !enzymejl_byref_MUT_REF !8, !enzymejl_allocart !55, !enzymejl_allocart_name !56, !enzymejl_source_type_\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D !8
  %i5 = call {}*** @julia.get_pgcstack()
  %i6 = getelementptr inbounds {}**, {}*** %i5, i64 2
  %i7 = bitcast {}*** %i6 to i64***
  %i8 = load i64**, i64*** %i7, align 8, !tbaa !2
  %i9 = getelementptr inbounds i64*, i64** %i8, i64 2
  %i10 = load i64*, i64** %i9, align 8, !tbaa !6
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %i10) #16
  fence syncscope("singlethread") seq_cst
  %i11 = call i64 @julia_nthreads_16030() #18
  %i12 = icmp slt i64 %i11, 2
  br i1 %i12, label %bb61, label %bb13

bb13:                                             ; preds = %bb
  %i14 = getelementptr inbounds {}**, {}*** %i5, i64 -14
  %i15 = bitcast {}*** %i14 to {}*
  %i16 = getelementptr inbounds [1 x [1 x [2 x [1 x double]]]], [1 x [1 x [2 x [1 x double]]]] addrspace(11)* %arg2, i64 0, i64 0, i64 0, i64 0, i64 0
  %i17 = load double, double addrspace(11)* %i16, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i18 = getelementptr inbounds [1 x [1 x [2 x [1 x double]]]], [1 x [1 x [2 x [1 x double]]]] addrspace(11)* %arg2, i64 0, i64 0, i64 0, i64 1, i64 0
  %i19 = load double, double addrspace(11)* %i18, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i20 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4522732752 to {}*) to {} addrspace(10)*), i64 noundef 1) #20
  %i21 = bitcast {} addrspace(10)* %i20 to { i64, {} addrspace(10)** } addrspace(10)*
  %i22 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i21 to { i64, {} addrspace(10)** } addrspace(11)*
  %i23 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i22, i64 0, i32 1
  %i24 = bitcast {} addrspace(10)** addrspace(11)* %i23 to i8* addrspace(11)*
  %i25 = load i8*, i8* addrspace(11)* %i24, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
  %i26 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i15, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4522732432 to {}*) to {} addrspace(10)*)) #17
  %i27 = bitcast {} addrspace(10)* %i26 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i28 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i27 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i29 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i28, i64 0, i32 0
  store i8* %i25, i8* addrspace(11)* %i29, align 8, !tbaa !42, !alias.scope !40, !noalias !57
  %i30 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i28, i64 0, i32 1
  store {} addrspace(10)* %i20, {} addrspace(10)* addrspace(11)* %i30, align 8, !tbaa !42, !alias.scope !40, !noalias !57
  %i31 = bitcast {} addrspace(10)* %i26 to i8 addrspace(10)*
  %i32 = addrspacecast i8 addrspace(10)* %i31 to i8 addrspace(11)*
  %i33 = getelementptr inbounds i8, i8 addrspace(11)* %i32, i64 16
  %i34 = bitcast i8 addrspace(11)* %i33 to i64 addrspace(11)*
  store i64 1, i64 addrspace(11)* %i34, align 8, !tbaa !37, !alias.scope !40, !noalias !57
  %i35 = bitcast i8* %i25 to {} addrspace(10)**
  %i36 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i20, {} addrspace(10)** noundef %i35)
  %i37 = bitcast {} addrspace(10)* addrspace(13)* %i36 to double addrspace(13)*
  store double %i17, double addrspace(13)* %i37, align 8, !tbaa !58, !alias.scope !59, !noalias !60
  %i38 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i36, i64 1
  %i39 = bitcast {} addrspace(10)* addrspace(13)* %i38 to double addrspace(13)*
  store double %i19, double addrspace(13)* %i39, align 8, !tbaa !58, !alias.scope !59, !noalias !60
  %i40 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }* %i3, i64 0, i32 0, i64 0, i64 0, i64 0, i64 0
  store double %i17, double* %i40, align 8, !noalias !36
  %i41 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }* %i3, i64 0, i32 0, i64 0, i64 0, i64 1, i64 0
  store double %i19, double* %i41, align 8, !noalias !36
  %i42 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }* %i3, i64 0, i32 1
  store {} addrspace(10)* %i26, {} addrspace(10)** %i42, align 8, !noalias !36
  %i43 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }* %i, i64 0, i32 1
  %i44 = load atomic {} addrspace(10)*, {} addrspace(10)** %i43 unordered, align 8, !tbaa !61, !alias.scope !63, !noalias !64, !nonnull !8, !dereferenceable !29, !align !30, !enzyme_type !31, !enzymejl_source_type_Vector\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8, !enzymejl_byref_MUT_REF !8
  br label %bb45

bb45:                                             ; preds = %bb13
  %i46 = bitcast {} addrspace(10)* %i44 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i47 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i46 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i48 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i47, i64 0, i32 1
  %i49 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i48, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !dereferenceable_or_null !44, !align !30, !enzyme_type !65, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
  %i50 = bitcast {} addrspace(10)* %i44 to {} addrspace(10)** addrspace(10)*
  %i51 = addrspacecast {} addrspace(10)** addrspace(10)* %i50 to {} addrspace(10)** addrspace(11)*
  %i52 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i51, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
  %i53 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i49, {} addrspace(10)** noundef %i52)
  %i54 = bitcast {} addrspace(10)* addrspace(13)* %i53 to double addrspace(13)*
  %i55 = load double, double addrspace(13)* %i54, align 8, !tbaa !58, !alias.scope !66, !noalias !60, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i56 = fadd double %i55, %arg1
  store double %i56, double addrspace(13)* %i54, align 8, !tbaa !58, !alias.scope !59, !noalias !60
  %i57 = addrspacecast { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }* %i to { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)*
  call fastcc void @a1([2 x [1 x double]]* noalias nocapture nofree noundef nonnull writeonly sret([2 x [1 x double]]) align 8 dereferenceable(16) %i4, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)* nocapture noundef nonnull readonly align 8 dereferenceable(24) %i57)
  %i58 = bitcast [2 x [1 x double]]* %arg to i8*
  %i59 = bitcast [2 x [1 x double]]* %i4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture nofree noundef nonnull writeonly align 8 dereferenceable(16) %i58, i8* noundef nonnull align 8 dereferenceable(16) %i59, i64 noundef 16, i1 noundef false), !noalias !36
  br label %bb60

bb60:                                             ; preds = %bb61, %bb45
  ret void

bb61:                                             ; preds = %bb
  br label %bb60
}

define "enzyme_type"="{[-1]:Float@double}" "enzymejl_parmtype"="5072643312" "enzymejl_parmtype_ref"="1" double @f({} addrspace(10)* noundef nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="5018769360" "enzymejl_parmtype_ref"="2" %arg, [1 x [1 x [2 x [1 x double]]]] "enzyme_type"="{[-1]:Float@double}" "enzymejl_parmtype"="4507524432" "enzymejl_parmtype_ref"="0" %arg1) local_unnamed_addr #13 {
bb:
  %i = alloca [2 x [1 x double]], align 8, !enzyme_type !54, !enzymejl_byref_MUT_REF !8, !enzymejl_allocart !55, !enzymejl_allocart_name !56, !enzymejl_source_type_\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D !8
  %i2 = alloca [1 x i64], align 8
  %i3 = alloca [1 x [1 x [2 x [1 x double]]]], align 8, !enzyme_inactive !8, !enzyme_type !54
  %i4 = extractvalue [1 x [1 x [2 x [1 x double]]]] %arg1, 0, 0, 0, 0, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i5 = getelementptr inbounds [1 x [1 x [2 x [1 x double]]]], [1 x [1 x [2 x [1 x double]]]]* %i3, i64 0, i64 0, i64 0, i64 0, i64 0
  store double %i4, double* %i5, align 8, !noalias !36
  %i6 = extractvalue [1 x [1 x [2 x [1 x double]]]] %arg1, 0, 0, 1, 0, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i7 = getelementptr inbounds [1 x [1 x [2 x [1 x double]]]], [1 x [1 x [2 x [1 x double]]]]* %i3, i64 0, i64 0, i64 0, i64 1, i64 0
  store double %i6, double* %i7, align 8, !noalias !36
  %i8 = bitcast [2 x [1 x double]]* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 noundef 16, i8* noundef nonnull align 8 dereferenceable(16) %i8) #21
  %i9 = bitcast [1 x i64]* %i2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull dereferenceable(8) %i9) #21
  %i10 = call {}*** @julia.get_pgcstack()
  %i11 = bitcast {} addrspace(10)* %arg to i8 addrspace(10)*
  %i12 = addrspacecast i8 addrspace(10)* %i11 to i8 addrspace(11)*
  %i13 = getelementptr inbounds i8, i8 addrspace(11)* %i12, i64 16
  %i14 = bitcast i8 addrspace(11)* %i13 to i64 addrspace(11)*
  %i15 = load i64, i64 addrspace(11)* %i14, align 8, !tbaa !37, !alias.scope !40, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i16 = icmp eq i64 %i15, 0
  br i1 %i16, label %bb17, label %bb20

bb17:                                             ; preds = %bb
  %i18 = getelementptr inbounds [1 x i64], [1 x i64]* %i2, i64 0, i64 0
  store i64 1, i64* %i18, align 8, !tbaa !61, !alias.scope !63, !noalias !67
  %i19 = addrspacecast [1 x i64]* %i2 to [1 x i64] addrspace(11)*
  call fastcc void @a0({} addrspace(10)* nofree noundef nonnull align 8 dereferenceable(24) %arg, [1 x i64] addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(8) %i19) #19
  unreachable

bb20:                                             ; preds = %bb
  %i21 = addrspacecast [1 x [1 x [2 x [1 x double]]]]* %i3 to [1 x [1 x [2 x [1 x double]]]] addrspace(11)*
  %i22 = bitcast {} addrspace(10)* %arg to { i8*, {} addrspace(10)* } addrspace(10)*
  %i23 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i22 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i24 = bitcast {} addrspace(10)* %arg to {} addrspace(10)** addrspace(10)*
  %i25 = addrspacecast {} addrspace(10)** addrspace(10)* %i24 to {} addrspace(10)** addrspace(11)*
  %i26 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i25, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BFloat64\7D !8
  %i27 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i23, i64 0, i32 1
  %i28 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i27, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !dereferenceable_or_null !44, !align !30, !enzyme_type !65, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BFloat64\7D !8
  %i29 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i28, {} addrspace(10)** noundef %i26)
  %i30 = bitcast {} addrspace(10)* addrspace(13)* %i29 to double addrspace(13)*
  %i31 = load double, double addrspace(13)* %i30, align 8, !tbaa !47, !alias.scope !23, !noalias !28, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  call fastcc void @a2([2 x [1 x double]]* noalias nocapture nofree noundef nonnull writeonly sret([2 x [1 x double]]) align 8 dereferenceable(16) %i, double %i31, [1 x [1 x [2 x [1 x double]]]] addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(16) %i21)
  %i32 = getelementptr inbounds [2 x [1 x double]], [2 x [1 x double]]* %i, i64 0, i64 0, i64 0
  %i33 = load double, double* %i32, align 8, !tbaa !61, !alias.scope !63, !noalias !64, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  call void @llvm.lifetime.end.p0i8(i64 noundef 16, i8* noundef nonnull %i8)
  call void @llvm.lifetime.end.p0i8(i64 noundef 8, i8* noundef nonnull %i9)
  ret double %i33
}

define internal fastcc noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @a3({ {} addrspace(10)* } addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(8) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Pointer, [-1,0,0,-1]:Float@double, [-1,0,8]:Pointer, [-1,0,8,0]:Integer, [-1,0,8,1]:Integer, [-1,0,8,2]:Integer, [-1,0,8,3]:Integer, [-1,0,8,4]:Integer, [-1,0,8,5]:Integer, [-1,0,8,6]:Integer, [-1,0,8,7]:Integer, [-1,0,8,8]:Pointer, [-1,0,8,8,-1]:Float@double, [-1,0,16]:Integer, [-1,0,17]:Integer, [-1,0,18]:Integer, [-1,0,19]:Integer, [-1,0,20]:Integer, [-1,0,21]:Integer, [-1,0,22]:Integer, [-1,0,23]:Integer}" "enzymejl_parmtype"="4522635664" "enzymejl_parmtype_ref"="1" %arg) unnamed_addr #14 {
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
  call void @julia.safepoint(i64* %i7) #16
  fence syncscope("singlethread") seq_cst
  %i8 = getelementptr inbounds { {} addrspace(10)* }, { {} addrspace(10)* } addrspace(11)* %arg, i64 0, i32 0
  %i9 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i8 unordered, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !31, !enzymejl_source_type_Vector\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8, !enzymejl_byref_MUT_REF !8
  %i10 = bitcast {} addrspace(10)* %i9 to i8 addrspace(10)*
  %i11 = addrspacecast i8 addrspace(10)* %i10 to i8 addrspace(11)*
  %i12 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 16
  %i13 = bitcast i8 addrspace(11)* %i12 to i64 addrspace(11)*
  %i14 = load i64, i64 addrspace(11)* %i13, align 8, !tbaa !37, !alias.scope !40, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i15 = icmp eq i64 %i14, 0
  br i1 %i15, label %bb16, label %bb36

bb16:                                             ; preds = %bb
  %i17 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4516519536 to {} addrspace(10)**) unordered, align 16, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !65, !enzymejl_source_type_Memory\7BLL\7BFloat64\7D\7D !8, !enzymejl_byref_BITS_REF !8
  %i18 = icmp eq {} addrspace(10)* %i17, null
  br i1 %i18, label %bb79, label %bb19

bb19:                                             ; preds = %bb16
  %i20 = bitcast {} addrspace(10)* %i17 to { i64, {} addrspace(10)** } addrspace(10)*
  %i21 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i20 to { i64, {} addrspace(10)** } addrspace(11)*
  %i22 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i21, i64 0, i32 1
  %i23 = bitcast {} addrspace(10)** addrspace(11)* %i22 to i8* addrspace(11)*
  %i24 = load i8*, i8* addrspace(11)* %i23, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BLL\7BFloat64\7D\7D !8
  %i25 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i2, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4516519056 to {}*) to {} addrspace(10)*)) #17
  %i26 = bitcast {} addrspace(10)* %i25 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i27 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i26 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i28 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i27, i64 0, i32 0
  store i8* %i24, i8* addrspace(11)* %i28, align 8, !tbaa !42, !alias.scope !40, !noalias !68
  %i29 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i27, i64 0, i32 1
  store {} addrspace(10)* %i17, {} addrspace(10)* addrspace(11)* %i29, align 8, !tbaa !42, !alias.scope !40, !noalias !68
  %i30 = bitcast {} addrspace(10)* %i25 to i8 addrspace(10)*
  %i31 = addrspacecast i8 addrspace(10)* %i30 to i8 addrspace(11)*
  %i32 = getelementptr inbounds i8, i8 addrspace(11)* %i31, i64 16
  %i33 = bitcast i8 addrspace(11)* %i32 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i33, align 8, !tbaa !37, !alias.scope !40, !noalias !68
  br label %bb34

bb34:                                             ; preds = %bb69, %bb36, %bb19
  %i35 = phi {} addrspace(10)* [ %i25, %bb19 ], [ %i54, %bb36 ], [ %i54, %bb69 ]
  ret {} addrspace(10)* %i35

bb36:                                             ; preds = %bb
  %i37 = bitcast {} addrspace(10)* %i9 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i38 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i37 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i39 = bitcast {} addrspace(10)* %i9 to {} addrspace(10)** addrspace(10)*
  %i40 = addrspacecast {} addrspace(10)** addrspace(10)* %i39 to {} addrspace(10)** addrspace(11)*
  %i41 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i40, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
  %i42 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i38, i64 0, i32 1
  %i43 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i42, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !dereferenceable_or_null !44, !align !30, !enzyme_type !65, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
  %i44 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i43, {} addrspace(10)** noundef %i41)
  %i45 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i44, i64 1
  %i46 = bitcast {} addrspace(10)* addrspace(13)* %i45 to double addrspace(13)*
  %i47 = load double, double addrspace(13)* %i46, align 8, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i48 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4516519504 to {}*) to {} addrspace(10)*), i64 %i14) #20
  %i49 = bitcast {} addrspace(10)* %i48 to { i64, {} addrspace(10)** } addrspace(10)*
  %i50 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i49 to { i64, {} addrspace(10)** } addrspace(11)*
  %i51 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i50, i64 0, i32 1
  %i52 = bitcast {} addrspace(10)** addrspace(11)* %i51 to i8* addrspace(11)*
  %i53 = load i8*, i8* addrspace(11)* %i52, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BLL\7BFloat64\7D\7D !8
  %i54 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i2, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4516519056 to {}*) to {} addrspace(10)*)) #17
  %i55 = bitcast {} addrspace(10)* %i54 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i56 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i55 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i57 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i56, i64 0, i32 0
  store i8* %i53, i8* addrspace(11)* %i57, align 8, !tbaa !42, !alias.scope !40, !noalias !68
  %i58 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i56, i64 0, i32 1
  store {} addrspace(10)* %i48, {} addrspace(10)* addrspace(11)* %i58, align 8, !tbaa !42, !alias.scope !40, !noalias !68
  %i59 = bitcast {} addrspace(10)* %i54 to i8 addrspace(10)*
  %i60 = addrspacecast i8 addrspace(10)* %i59 to i8 addrspace(11)*
  %i61 = getelementptr inbounds i8, i8 addrspace(11)* %i60, i64 16
  %i62 = bitcast i8 addrspace(11)* %i61 to i64 addrspace(11)*
  store i64 %i14, i64 addrspace(11)* %i62, align 8, !tbaa !37, !alias.scope !40, !noalias !68
  %i63 = bitcast i8* %i53 to {} addrspace(10)**
  %i64 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i48, {} addrspace(10)** noundef %i63)
  %i65 = bitcast {} addrspace(10)* addrspace(13)* %i64 to double addrspace(13)*
  store double %i47, double addrspace(13)* %i65, align 8, !tbaa !47, !alias.scope !23, !noalias !71
  %i66 = icmp eq i64 %i14, 1
  br i1 %i66, label %bb34, label %bb67

bb67:                                             ; preds = %bb36
  %i68 = bitcast {} addrspace(10)* addrspace(13)* %i44 to [1 x [2 x [1 x double]]] addrspace(13)*
  br label %bb69

bb69:                                             ; preds = %bb69, %bb67
  %i70 = phi i64 [ 1, %bb67 ], [ %i71, %bb69 ]
  %i71 = phi i64 [ 2, %bb67 ], [ %i72, %bb69 ]
  %i72 = add i64 %i71, 1
  %i73 = getelementptr inbounds [1 x [2 x [1 x double]]], [1 x [2 x [1 x double]]] addrspace(13)* %i68, i64 %i70, i64 0, i64 1, i64 0
  %i74 = load double, double addrspace(13)* %i73, align 8
  %i75 = add i64 %i71, -1
  %i76 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i64, i64 %i75
  %i77 = bitcast {} addrspace(10)* addrspace(13)* %i76 to double addrspace(13)*
  store double %i74, double addrspace(13)* %i77, align 8, !tbaa !47, !alias.scope !23, !noalias !71
  %i78 = icmp eq i64 %i71, %i14
  br i1 %i78, label %bb34, label %bb69

bb79:                                             ; preds = %bb16
  %i80 = load {}*, {}** @jl_undefref_exception, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8
  %i81 = addrspacecast {}* %i80 to {} addrspace(12)*
  call void @ijl_throw({} addrspace(12)* %i81) #19
  unreachable
}

define internal fastcc noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @a4({ {} addrspace(10)* } addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(8) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Pointer, [-1,0,0,-1]:Float@double, [-1,0,8]:Pointer, [-1,0,8,0]:Integer, [-1,0,8,1]:Integer, [-1,0,8,2]:Integer, [-1,0,8,3]:Integer, [-1,0,8,4]:Integer, [-1,0,8,5]:Integer, [-1,0,8,6]:Integer, [-1,0,8,7]:Integer, [-1,0,8,8]:Pointer, [-1,0,8,8,-1]:Float@double, [-1,0,16]:Integer, [-1,0,17]:Integer, [-1,0,18]:Integer, [-1,0,19]:Integer, [-1,0,20]:Integer, [-1,0,21]:Integer, [-1,0,22]:Integer, [-1,0,23]:Integer}" "enzymejl_parmtype"="4538117264" "enzymejl_parmtype_ref"="1" %arg) unnamed_addr #15 {
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
  call void @julia.safepoint(i64* %i7) #16
  fence syncscope("singlethread") seq_cst
  %i8 = getelementptr inbounds { {} addrspace(10)* }, { {} addrspace(10)* } addrspace(11)* %arg, i64 0, i32 0
  %i9 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i8 unordered, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !31, !enzymejl_source_type_Vector\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8, !enzymejl_byref_MUT_REF !8
  %i10 = bitcast {} addrspace(10)* %i9 to i8 addrspace(10)*
  %i11 = addrspacecast i8 addrspace(10)* %i10 to i8 addrspace(11)*
  %i12 = getelementptr inbounds i8, i8 addrspace(11)* %i11, i64 16
  %i13 = bitcast i8 addrspace(11)* %i12 to i64 addrspace(11)*
  %i14 = load i64, i64 addrspace(11)* %i13, align 8, !tbaa !37, !alias.scope !40, !noalias !41, !enzyme_inactive !8, !enzyme_type !17, !enzymejl_source_type_Int64 !8, !enzymejl_byref_BITS_VALUE !8
  %i15 = icmp eq i64 %i14, 0
  br i1 %i15, label %bb16, label %bb36

bb16:                                             ; preds = %bb
  %i17 = load atomic {} addrspace(10)*, {} addrspace(10)** inttoptr (i64 4522229168 to {} addrspace(10)**) unordered, align 16, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !65, !enzymejl_byref_BITS_REF !8, !enzymejl_source_type_Memory\7BLP\7BFloat64\7D\7D !8
  %i18 = icmp eq {} addrspace(10)* %i17, null
  br i1 %i18, label %bb78, label %bb19

bb19:                                             ; preds = %bb16
  %i20 = bitcast {} addrspace(10)* %i17 to { i64, {} addrspace(10)** } addrspace(10)*
  %i21 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i20 to { i64, {} addrspace(10)** } addrspace(11)*
  %i22 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i21, i64 0, i32 1
  %i23 = bitcast {} addrspace(10)** addrspace(11)* %i22 to i8* addrspace(11)*
  %i24 = load i8*, i8* addrspace(11)* %i23, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Ptr\7BLP\7BFloat64\7D\7D !8
  %i25 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i2, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4522228816 to {}*) to {} addrspace(10)*)) #17
  %i26 = bitcast {} addrspace(10)* %i25 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i27 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i26 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i28 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i27, i64 0, i32 0
  store i8* %i24, i8* addrspace(11)* %i28, align 8, !tbaa !42, !alias.scope !40, !noalias !72
  %i29 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i27, i64 0, i32 1
  store {} addrspace(10)* %i17, {} addrspace(10)* addrspace(11)* %i29, align 8, !tbaa !42, !alias.scope !40, !noalias !72
  %i30 = bitcast {} addrspace(10)* %i25 to i8 addrspace(10)*
  %i31 = addrspacecast i8 addrspace(10)* %i30 to i8 addrspace(11)*
  %i32 = getelementptr inbounds i8, i8 addrspace(11)* %i31, i64 16
  %i33 = bitcast i8 addrspace(11)* %i32 to i64 addrspace(11)*
  store i64 0, i64 addrspace(11)* %i33, align 8, !tbaa !37, !alias.scope !40, !noalias !72
  br label %bb34

bb34:                                             ; preds = %bb68, %bb36, %bb19
  %i35 = phi {} addrspace(10)* [ %i25, %bb19 ], [ %i53, %bb36 ], [ %i53, %bb68 ]
  ret {} addrspace(10)* %i35

bb36:                                             ; preds = %bb
  %i37 = bitcast {} addrspace(10)* %i9 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i38 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i37 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i39 = bitcast {} addrspace(10)* %i9 to {} addrspace(10)** addrspace(10)*
  %i40 = addrspacecast {} addrspace(10)** addrspace(10)* %i39 to {} addrspace(10)** addrspace(11)*
  %i41 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i40, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
  %i42 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i38, i64 0, i32 1
  %i43 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i42, align 8, !tbaa !42, !alias.scope !40, !noalias !41, !dereferenceable_or_null !44, !align !30, !enzyme_type !65, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
  %i44 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i43, {} addrspace(10)** noundef %i41)
  %i45 = bitcast {} addrspace(10)* addrspace(13)* %i44 to double addrspace(13)*
  %i46 = load double, double addrspace(13)* %i45, align 8, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
  %i47 = call noalias "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@double}" {} addrspace(10)* @jl_alloc_genericmemory({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4522229136 to {}*) to {} addrspace(10)*), i64 %i14) #20
  %i48 = bitcast {} addrspace(10)* %i47 to { i64, {} addrspace(10)** } addrspace(10)*
  %i49 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i48 to { i64, {} addrspace(10)** } addrspace(11)*
  %i50 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i49, i64 0, i32 1
  %i51 = bitcast {} addrspace(10)** addrspace(11)* %i50 to i8* addrspace(11)*
  %i52 = load i8*, i8* addrspace(11)* %i51, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BLP\7BFloat64\7D\7D !8
  %i53 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i2, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4522228816 to {}*) to {} addrspace(10)*)) #17
  %i54 = bitcast {} addrspace(10)* %i53 to { i8*, {} addrspace(10)* } addrspace(10)*
  %i55 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i54 to { i8*, {} addrspace(10)* } addrspace(11)*
  %i56 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i55, i64 0, i32 0
  store i8* %i52, i8* addrspace(11)* %i56, align 8, !tbaa !42, !alias.scope !40, !noalias !72
  %i57 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i55, i64 0, i32 1
  store {} addrspace(10)* %i47, {} addrspace(10)* addrspace(11)* %i57, align 8, !tbaa !42, !alias.scope !40, !noalias !72
  %i58 = bitcast {} addrspace(10)* %i53 to i8 addrspace(10)*
  %i59 = addrspacecast i8 addrspace(10)* %i58 to i8 addrspace(11)*
  %i60 = getelementptr inbounds i8, i8 addrspace(11)* %i59, i64 16
  %i61 = bitcast i8 addrspace(11)* %i60 to i64 addrspace(11)*
  store i64 %i14, i64 addrspace(11)* %i61, align 8, !tbaa !37, !alias.scope !40, !noalias !72
  %i62 = bitcast i8* %i52 to {} addrspace(10)**
  %i63 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i47, {} addrspace(10)** noundef %i62)
  %i64 = bitcast {} addrspace(10)* addrspace(13)* %i63 to double addrspace(13)*
  store double %i46, double addrspace(13)* %i64, align 8, !tbaa !47, !alias.scope !23, !noalias !75
  %i65 = icmp eq i64 %i14, 1
  br i1 %i65, label %bb34, label %bb66

bb66:                                             ; preds = %bb36
  %i67 = bitcast {} addrspace(10)* addrspace(13)* %i44 to [1 x [2 x [1 x double]]] addrspace(13)*
  br label %bb68

bb68:                                             ; preds = %bb68, %bb66
  %i69 = phi i64 [ 1, %bb66 ], [ %i70, %bb68 ]
  %i70 = phi i64 [ 2, %bb66 ], [ %i71, %bb68 ]
  %i71 = add i64 %i70, 1
  %i72 = getelementptr inbounds [1 x [2 x [1 x double]]], [1 x [2 x [1 x double]]] addrspace(13)* %i67, i64 %i69, i64 0, i64 0, i64 0
  %i73 = load double, double addrspace(13)* %i72, align 8
  %i74 = add i64 %i70, -1
  %i75 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i63, i64 %i74
  %i76 = bitcast {} addrspace(10)* addrspace(13)* %i75 to double addrspace(13)*
  store double %i73, double addrspace(13)* %i76, align 8, !tbaa !47, !alias.scope !23, !noalias !75
  %i77 = icmp eq i64 %i70, %i14
  br i1 %i77, label %bb34, label %bb68

bb78:                                             ; preds = %bb16
  %i79 = load {}*, {}** @jl_undefref_exception, align 8, !tbaa !6, !alias.scope !9, !noalias !12, !nonnull !8
  %i80 = addrspacecast {}* %i79 to {} addrspace(12)*
  call void @ijl_throw({} addrspace(12)* %i80) #19
  unreachable
}

declare void @__enzyme_autodiff(...) local_unnamed_addr

define double @dsquare({} addrspace(10)* %arg, {} addrspace(10)* %arg1, [1 x [1 x [2 x [1 x double]]]] %arg2) local_unnamed_addr {
bb:
  call void (...) @__enzyme_autodiff(double ({} addrspace(10)*, [1 x [1 x [2 x [1 x double]]]])* @f, metadata !"enzyme_dup", {} addrspace(10)* %arg, {} addrspace(10)* %arg1, metadata !"enzyme_out", [1 x [1 x [2 x [1 x double]]]] %arg2)
  ret double undef
}

attributes #0 = { noinline noreturn "enzyme_ta_norecur" "enzymejl_mi"="5052725408" "enzymejl_rt"="5135844496" "enzymejl_world"="26752" }
attributes #1 = { nofree memory(none) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzyme_shouldrecompute" "enzymejl_world"="26752" }
attributes #2 = { nofree memory(argmem: readwrite, inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26752" }
attributes #3 = { nofree "enzymejl_world"="26752" }
attributes #4 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" "enzymejl_world"="26752" }
attributes #5 = { noreturn "enzyme_no_escaping_allocation" "enzymejl_world"="26752" }
attributes #6 = { nofree noinline speculatable memory(none) "enzyme_inactive" "enzyme_math"="jl_nthreads" "enzyme_no_escaping_allocation" "enzyme_nofree" "enzyme_shouldrecompute" "enzyme_ta_norecur" "enzymejl_mi"="4517727760" "enzymejl_rt"="5072644112" "enzymejl_world"="26752" }
attributes #7 = { nofree norecurse nosync nounwind speculatable willreturn memory(none) "enzyme_nocache" "enzyme_shouldrecompute" "enzymejl_world"="26752" }
attributes #8 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #10 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" "enzymejl_world"="26752" }
attributes #11 = { "enzyme_ta_norecur" "enzymejl_mi"="4538635920" "enzymejl_rt"="4507777552" "enzymejl_world"="26752" }
attributes #12 = { "enzyme_ta_norecur" "enzymejl_mi"="4522543888" "enzymejl_rt"="4507777552" "enzymejl_world"="26752" }
attributes #13 = { "enzyme_ta_norecur" "enzymejl_mi"="4522542480" "enzymejl_rt"="5072643312" "enzymejl_world"="26752" }
attributes #14 = { "enzyme_parmremove"="0" "enzyme_ta_norecur" "enzymejl_mi"="4522235920" "enzymejl_rt"="4516519056" "enzymejl_world"="26752" }
attributes #15 = { "enzyme_parmremove"="0" "enzyme_ta_norecur" "enzymejl_mi"="4538017616" "enzymejl_rt"="4522228816" "enzymejl_world"="26752" }
attributes #16 = { memory(readwrite) }
attributes #17 = { nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #18 = { nofree }
attributes #19 = { noreturn }
attributes #20 = { nounwind memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #21 = { willreturn }

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
!27 = !{!22, !22, i64 0}
!28 = !{!13, !14, !16, !10}
!29 = !{i64 24}
!30 = !{i64 8}
!31 = !{!"Unknown", i32 -1, !32}
!32 = !{!"Pointer", i32 0, !33, i32 8, !35, i32 16, !18, i32 17, !18, i32 18, !18, i32 19, !18, i32 20, !18, i32 21, !18, i32 22, !18, i32 23, !18}
!33 = !{!"Pointer", i32 -1, !34}
!34 = !{!"Float@double"}
!35 = !{!"Pointer", i32 0, !18, i32 1, !18, i32 2, !18, i32 3, !18, i32 4, !18, i32 5, !18, i32 6, !18, i32 7, !18, i32 8, !33}
!36 = !{!25}
!37 = !{!38, !38, i64 0}
!38 = !{!"jtbaa_arraysize", !39, i64 0}
!39 = !{!"jtbaa_array", !4, i64 0}
!40 = !{!16}
!41 = !{!13, !14, !15, !10}
!42 = !{!43, !43, i64 0}
!43 = !{!"jtbaa_arrayptr", !39, i64 0}
!44 = !{i64 16}
!45 = !{!"Unknown", i32 -1, !46}
!46 = !{!"Pointer"}
!47 = !{!48, !48, i64 0}
!48 = !{!"jtbaa_arraybuf", !22, i64 0}
!49 = !{!"Unknown", i32 -1, !34}
!50 = !{!"Unknown", i32 -1, !51}
!51 = !{!"Pointer", i32 0, !34, i32 8, !34, i32 16, !32}
!52 = !{!"4522976336"}
!53 = !{!"TSVI4{V0{AT2{2, @NamedTuple{SymLP::LP{Float64}, SymLL::LL{Float64}}}}, AT2{2, @NamedTuple{SymLP::LP{Float64}, SymLL::LL{Float64}}}}"}
!54 = !{!"Unknown", i32 -1, !33}
!55 = !{!"4507777552"}
!56 = !{!"@NamedTuple{SymLP::LP{Float64}, SymLL::LL{Float64}}"}
!57 = !{!25, !13, !14, !15, !10}
!58 = !{!4, !4, i64 0}
!59 = !{!14, !15}
!60 = !{!25, !13, !16, !10}
!61 = !{!62, !62, i64 0}
!62 = !{!"jtbaa_stack", !4, i64 0}
!63 = !{!14}
!64 = !{!13, !15, !16, !10}
!65 = !{!"Unknown", i32 -1, !35}
!66 = !{!15, !14}
!67 = !{!25, !13, !15, !16, !10}
!68 = !{!69, !13, !14, !15, !10}
!69 = distinct !{!69, !70, !"na_addr13"}
!70 = distinct !{!70, !"addr13"}
!71 = !{!69, !13, !14, !16, !10}
!72 = !{!73, !13, !14, !15, !10}
!73 = distinct !{!73, !74, !"na_addr13"}
!74 = distinct !{!74, !"addr13"}
!75 = !{!73, !13, !14, !16, !10}

; CHECK: define internal fastcc { double } @diffea2([2 x [1 x double]]* noalias nocapture nofree writeonly align 8 dereferenceable(16) "enzyme_sret"="{{[0-9]+}}" "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %arg, [2 x [1 x double]]* nocapture nofree align 8 "enzyme_sret"="{{[0-9]+}}" "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %"arg'", double "enzyme_type"="{[-1]:Float@double}" "enzymejl_parmtype"="5072643312" "enzymejl_parmtype_ref"="0" %arg1, [1 x [1 x [2 x [1 x double]]]] addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}" "enzymejl_parmtype"="4507524432" "enzymejl_parmtype_ref"="1" %arg2) 
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"arg1'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"arg1'de", align 8
; CHECK-NEXT:   %"i56'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"i56'de", align 8
; CHECK-NEXT:   %"i20'ip_phi_cache" = alloca {} addrspace(10)*, align 8
; CHECK-NEXT:   %i = alloca { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, align 8, !enzyme_type !50, !enzymejl_byref_MUT_REF !8, !enzymejl_allocart !52, !enzymejl_allocart_name !53, !enzymejl_source_type_TSVI4\7BV0\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D\2C\20AT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
; CHECK-NEXT:   %"i4'ipa" = alloca [2 x [1 x double]], align 8
; CHECK-NEXT:   store [2 x [1 x double]] zeroinitializer, [2 x [1 x double]]* %"i4'ipa", align 8
; CHECK-NEXT:   %i5 = call {}*** @julia.get_pgcstack() #25
; CHECK-NEXT:   %i6 = getelementptr inbounds {}**, {}*** %i5, i64 2
; CHECK-NEXT:   %i7 = bitcast {}*** %i6 to i64***
; CHECK-NEXT:   %i8 = load i64**, i64*** %i7, align 8, !tbaa !2, !alias.scope !95, !noalias !98
; CHECK-NEXT:   %i9 = getelementptr inbounds i64*, i64** %i8, i64 2
; CHECK-NEXT:   %i10 = load i64*, i64** %i9, align 8, !tbaa !6, !alias.scope !100, !noalias !103
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   call void @julia.safepoint(i64* %i10) #27
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   %i11 = call i64 @julia_nthreads_16030() #28
; CHECK-NEXT:   %i12 = icmp slt i64 %i11, 2
; CHECK-NEXT:   br i1 %i12, label %bb60, label %bb13

; CHECK: bb13:                                             ; preds = %bb
; CHECK-NEXT:   %i14 = getelementptr inbounds {}**, {}*** %i5, i64 -14
; CHECK-NEXT:   %i15 = bitcast {}*** %i14 to {}*
; CHECK-NEXT:   %i16 = getelementptr inbounds [1 x [1 x [2 x [1 x double]]]], [1 x [1 x [2 x [1 x double]]]] addrspace(11)* %arg2, i64 0, i64 0, i64 0, i64 0, i64 0
; CHECK-NEXT:   %i17 = load double, double addrspace(11)* %i16, align 8, !tbaa !6, !alias.scope !105, !noalias !108, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
; CHECK-NEXT:   %i18 = getelementptr inbounds [1 x [1 x [2 x [1 x double]]]], [1 x [1 x [2 x [1 x double]]]] addrspace(11)* %arg2, i64 0, i64 0, i64 0, i64 1, i64 0
; CHECK-NEXT:   %i19 = load double, double addrspace(11)* %i18, align 8, !tbaa !6, !alias.scope !105, !noalias !108, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
; CHECK-NEXT:   %i20_augmented = call { {} addrspace(10)*, {} addrspace(10)* } @augmented_jl_alloc_genericmemory({} addrspace(10)* addrspacecast ({}* inttoptr (i64 4522732752 to {}*) to {} addrspace(10)*), i64 1)
; CHECK-NEXT:   %i20 = extractvalue { {} addrspace(10)*, {} addrspace(10)* } %i20_augmented, 0
; CHECK-NEXT:   %"i20'ac" = extractvalue { {} addrspace(10)*, {} addrspace(10)* } %i20_augmented, 1
; CHECK-NEXT:   %"i21'ipc" = bitcast {} addrspace(10)* %"i20'ac" to { i64, {} addrspace(10)** } addrspace(10)*
; CHECK-NEXT:   %i21 = bitcast {} addrspace(10)* %i20 to { i64, {} addrspace(10)** } addrspace(10)*
; CHECK-NEXT:   %"i22'ipc" = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %"i21'ipc" to { i64, {} addrspace(10)** } addrspace(11)*
; CHECK-NEXT:   %i22 = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %i21 to { i64, {} addrspace(10)** } addrspace(11)*
; CHECK-NEXT:   %"i23'ipg" = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %"i22'ipc", i64 0, i32 1
; CHECK-NEXT:   %i23 = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %i22, i64 0, i32 1
; CHECK-NEXT:   %"i24'ipc" = bitcast {} addrspace(10)** addrspace(11)* %"i23'ipg" to i8* addrspace(11)*
; CHECK-NEXT:   %i24 = bitcast {} addrspace(10)** addrspace(11)* %i23 to i8* addrspace(11)*
; CHECK-NEXT:   %"i25'ipl" = load i8*, i8* addrspace(11)* %"i24'ipc", align 8, !tbaa !6, !alias.scope !110, !noalias !113, !nonnull !8
; CHECK-NEXT:   %i25 = load i8*, i8* addrspace(11)* %i24, align 8, !tbaa !6, !alias.scope !115, !noalias !116, !nonnull !8, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
; CHECK-NEXT:   %i26 = call noalias nonnull align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" {} addrspace(10)* @julia.gc_alloc_obj({}* nonnull %i15, i64 noundef 24, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 4522732432 to {}*) to {} addrspace(10)*)) #29
; CHECK-NEXT:   %i27 = bitcast {} addrspace(10)* %i26 to { i8*, {} addrspace(10)* } addrspace(10)*
; CHECK-NEXT:   %i28 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i27 to { i8*, {} addrspace(10)* } addrspace(11)*
; CHECK-NEXT:   %i29 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i28, i64 0, i32 0
; CHECK-NEXT:   store i8* %i25, i8* addrspace(11)* %i29, align 8, !tbaa !42, !alias.scope !40, !noalias !57
; CHECK-NEXT:   store {} addrspace(10)* %"i20'ac", {} addrspace(10)** %"i20'ip_phi_cache", align 8, !invariant.group !117
; CHECK-NEXT:   %i30 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i28, i64 0, i32 1
; CHECK-NEXT:   store {} addrspace(10)* %i20, {} addrspace(10)* addrspace(11)* %i30, align 8, !tbaa !42, !alias.scope !40, !noalias !57
; CHECK-NEXT:   %i31 = bitcast {} addrspace(10)* %i26 to i8 addrspace(10)*
; CHECK-NEXT:   %i32 = addrspacecast i8 addrspace(10)* %i31 to i8 addrspace(11)*
; CHECK-NEXT:   %i33 = getelementptr inbounds i8, i8 addrspace(11)* %i32, i64 16
; CHECK-NEXT:   %i34 = bitcast i8 addrspace(11)* %i33 to i64 addrspace(11)*
; CHECK-NEXT:   store i64 1, i64 addrspace(11)* %i34, align 8, !tbaa !37, !alias.scope !40, !noalias !57
; CHECK-NEXT:   %"i35'ipc" = bitcast i8* %"i25'ipl" to {} addrspace(10)**
; CHECK-NEXT:   %i35 = bitcast i8* %i25 to {} addrspace(10)**
; CHECK-NEXT:   %0 = call {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* %"i20'ac", {} addrspace(10)** %"i35'ipc")
; CHECK-NEXT:   %i36 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i20, {} addrspace(10)** noundef %i35) #25
; CHECK-NEXT:   %"i37'ipc" = bitcast {} addrspace(10)* addrspace(13)* %0 to double addrspace(13)*
; CHECK-NEXT:   %i37 = bitcast {} addrspace(10)* addrspace(13)* %i36 to double addrspace(13)*
; CHECK-NEXT:   store double %i17, double addrspace(13)* %i37, align 8, !tbaa !58, !alias.scope !118, !noalias !121
; CHECK-NEXT:   %"i38'ipg" = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %0, i64 1
; CHECK-NEXT:   %i38 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %i36, i64 1
; CHECK-NEXT:   %"i39'ipc" = bitcast {} addrspace(10)* addrspace(13)* %"i38'ipg" to double addrspace(13)*
; CHECK-NEXT:   %i39 = bitcast {} addrspace(10)* addrspace(13)* %i38 to double addrspace(13)*
; CHECK-NEXT:   store double %i19, double addrspace(13)* %i39, align 8, !tbaa !58, !alias.scope !118, !noalias !121
; CHECK-NEXT:   %i43 = getelementptr inbounds { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }, { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }* %i, i64 0, i32 1
; CHECK-NEXT:   %i44 = load atomic {} addrspace(10)*, {} addrspace(10)** %i43 unordered, align 8, !tbaa !61, !alias.scope !123, !noalias !126, !nonnull !8, !dereferenceable !29, !align !30, !enzyme_type !31, !enzymejl_source_type_Vector\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8, !enzymejl_byref_MUT_REF !8
; CHECK-NEXT:   %i46 = bitcast {} addrspace(10)* %i44 to { i8*, {} addrspace(10)* } addrspace(10)*
; CHECK-NEXT:   %i47 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %i46 to { i8*, {} addrspace(10)* } addrspace(11)*
; CHECK-NEXT:   %i48 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %i47, i64 0, i32 1
; CHECK-NEXT:   %i49 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i48, align 8, !tbaa !42, !alias.scope !128, !noalias !131, !dereferenceable_or_null !44, !align !30, !enzyme_type !65, !enzymejl_byref_MUT_REF !8, !enzymejl_source_type_Memory\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
; CHECK-NEXT:   %i50 = bitcast {} addrspace(10)* %i44 to {} addrspace(10)** addrspace(10)*
; CHECK-NEXT:   %i51 = addrspacecast {} addrspace(10)** addrspace(10)* %i50 to {} addrspace(10)** addrspace(11)*
; CHECK-NEXT:   %i52 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %i51, align 8, !tbaa !42, !alias.scope !128, !noalias !131, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !8, !enzyme_nocache !8, !enzymejl_source_type_Ptr\7BAT2\7B2\2C\20\40NamedTuple\7BSymLP\3A\3ALP\7BFloat64\7D\2C\20SymLL\3A\3ALL\7BFloat64\7D\7D\7D\7D !8
; CHECK-NEXT:   %i53 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %i49, {} addrspace(10)** noundef %i52) #25
; CHECK-NEXT:   %i54 = bitcast {} addrspace(10)* addrspace(13)* %i53 to double addrspace(13)*
; CHECK-NEXT:   %i55 = load double, double addrspace(13)* %i54, align 8, !tbaa !58, !alias.scope !133, !noalias !136, !enzyme_type !49, !enzymejl_byref_BITS_VALUE !8, !enzymejl_source_type_Float64 !8
; CHECK-NEXT:   %i56 = fadd double %i55, %arg1
; CHECK-NEXT:   store double %i56, double addrspace(13)* %i54, align 8, !tbaa !58, !alias.scope !59, !noalias !60
; CHECK-NEXT:   %i57 = addrspacecast { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }* %i to { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)*
; CHECK-NEXT:   %"i58'ipc" = bitcast [2 x [1 x double]]* %"arg'" to i8*
; CHECK-NEXT:   %"i59'ipc" = bitcast [2 x [1 x double]]* %"i4'ipa" to i8*
; CHECK-NEXT:   br label %bb60

; CHECK: bb60:                                             ; preds = %bb13, %bb
; CHECK-NEXT:   br label %invertbb60

; CHECK: invertbb:                                         ; preds = %invertbb60, %__enzyme_memcpyadd_doubleda8sa8.exit
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   fence syncscope("singlethread") seq_cst
; CHECK-NEXT:   %1 = load double, double* %"arg1'de", align 8
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2

; CHECK: invertbb13:                                       ; preds = %invertbb60
; CHECK-NEXT:   %"i58'ipc_unwrap" = bitcast [2 x [1 x double]]* %"arg'" to i8*
; CHECK-NEXT:   %"i59'ipc_unwrap" = bitcast [2 x [1 x double]]* %"i4'ipa" to i8*
; CHECK-NEXT:   %3 = bitcast i8* %"i58'ipc_unwrap" to double*
; CHECK-NEXT:   %4 = bitcast i8* %"i59'ipc_unwrap" to double*
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invertbb13
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invertbb13 ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %3, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load double, double* %dst.i.i, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i, align 8
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %4, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i, align 8
; CHECK-NEXT:   %5 = fadd fast double %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store double %5, double* %src.i.i, align 8
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %6 = icmp eq i64 2, %idx.next.i
; CHECK-NEXT:   br i1 %6, label %__enzyme_memcpyadd_doubleda8sa8.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_doubleda8sa8.exit:             ; preds = %for.body.i
; CHECK-NEXT:   %i57_unwrap = addrspacecast { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* }* %i to { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)*
; CHECK-NEXT:   call fastcc void @diffea1([2 x [1 x double]]* nocapture nofree writeonly align 8 "enzyme_sret"="{{[0-9]+}}" undef, [2 x [1 x double]]* nocapture nofree align 8 "enzyme_sret"="{{[0-9]+}}" %"i4'ipa", { [1 x [1 x [2 x [1 x double]]]], {} addrspace(10)* } addrspace(11)* nocapture readonly align 8 %i57_unwrap)
; CHECK-NEXT:   %7 = load double, double* %"i56'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"i56'de", align 8
; CHECK-NEXT:   %8 = load double, double* %"arg1'de", align 8
; CHECK-NEXT:   %9 = fadd fast double %8, %7
; CHECK-NEXT:   store double %9, double* %"arg1'de", align 8
; CHECK-NEXT:   %10 = load {} addrspace(10)*, {} addrspace(10)** %"i20'ip_phi_cache", align 8, !invariant.group !117
; CHECK-NEXT:   %"i21'ipc_unwrap" = bitcast {} addrspace(10)* %10 to { i64, {} addrspace(10)** } addrspace(10)*
; CHECK-NEXT:   %"i22'ipc_unwrap" = addrspacecast { i64, {} addrspace(10)** } addrspace(10)* %"i21'ipc_unwrap" to { i64, {} addrspace(10)** } addrspace(11)*
; CHECK-NEXT:   %"i23'ipg_unwrap" = getelementptr inbounds { i64, {} addrspace(10)** }, { i64, {} addrspace(10)** } addrspace(11)* %"i22'ipc_unwrap", i64 0, i32 1
; CHECK-NEXT:   %"i24'ipc_unwrap" = bitcast {} addrspace(10)** addrspace(11)* %"i23'ipg_unwrap" to i8* addrspace(11)*
; CHECK-NEXT:   %"i25'il_phi_unwrap" = load i8*, i8* addrspace(11)* %"i24'ipc_unwrap", align 8, !tbaa !6, !alias.scope !110, !noalias !113, !nonnull !8
; CHECK-NEXT:   %"i35'ipc_unwrap" = bitcast i8* %"i25'il_phi_unwrap" to {} addrspace(10)**
; CHECK-NEXT:   %11 = call {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* %10, {} addrspace(10)** %"i35'ipc_unwrap")
; CHECK-NEXT:   %"i38'ipg_unwrap" = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %11, i64 1
; CHECK-NEXT:   %"i39'ipc_unwrap" = bitcast {} addrspace(10)* addrspace(13)* %"i38'ipg_unwrap" to double addrspace(13)*
; CHECK-NEXT:   store double 0.000000e+00, double addrspace(13)* %"i39'ipc_unwrap", align 8, !tbaa !58, !alias.scope !138, !noalias !139
; CHECK-NEXT:   %"i37'ipc_unwrap" = bitcast {} addrspace(10)* addrspace(13)* %11 to double addrspace(13)*
; CHECK-NEXT:   store double 0.000000e+00, double addrspace(13)* %"i37'ipc_unwrap", align 8, !tbaa !58, !alias.scope !138, !noalias !139
; CHECK-NEXT:   call void @diffejl_alloc_genericmemory({} addrspace(10)* {{(undef|poison)}}, i64 1)
; CHECK-NEXT:   br label %invertbb

; CHECK: invertbb60:                                       ; preds = %bb60
; CHECK-NEXT:   br i1 %i12, label %invertbb, label %invertbb13
; CHECK-NEXT: }
