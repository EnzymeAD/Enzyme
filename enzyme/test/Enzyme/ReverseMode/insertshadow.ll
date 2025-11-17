; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S -enzyme-julia-addr-load | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S -enzyme-julia-addr-load | FileCheck %s

source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

; Function Attrs: nofree readnone
declare {}*** @julia.get_pgcstack() local_unnamed_addr #0

; Function Attrs: argmemonly nocallback nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p13i8.i64(i8* noalias nocapture writeonly, i8 addrspace(13)* noalias nocapture readonly, i64, i1 immarg) #8

; Function Attrs: noinline
define internal fastcc double @a0({} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer}" "enzymejl_parmtype"="139771446999952" "enzymejl_parmtype_ref"="2" %arg, [2 x [3 x {} addrspace(10)*]] addrspace(11)* nocapture noundef nonnull readonly align 8 dereferenceable(48) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Pointer, [-1,0,0,-1]:Integer, [-1,0,8]:Integer, [-1,0,9]:Integer, [-1,0,10]:Integer, [-1,0,11]:Integer, [-1,0,12]:Integer, [-1,0,13]:Integer, [-1,0,14]:Integer, [-1,0,15]:Integer, [-1,0,16]:Integer, [-1,0,17]:Integer, [-1,0,18]:Integer, [-1,0,19]:Integer, [-1,0,20]:Integer, [-1,0,21]:Integer, [-1,0,22]:Integer, [-1,0,23]:Integer, [-1,0,24]:Integer, [-1,0,25]:Integer, [-1,0,26]:Integer, [-1,0,27]:Integer, [-1,0,28]:Integer, [-1,0,29]:Integer, [-1,0,30]:Integer, [-1,0,31]:Integer, [-1,0,32]:Integer, [-1,0,33]:Integer, [-1,0,34]:Integer, [-1,0,35]:Integer, [-1,0,36]:Integer, [-1,0,37]:Integer, [-1,0,38]:Integer, [-1,0,39]:Integer, [-1,8]:Pointer, [-1,8,0]:Pointer, [-1,8,0,-1]:Integer, [-1,8,8]:Integer, [-1,8,9]:Integer, [-1,8,10]:Integer, [-1,8,11]:Integer, [-1,8,12]:Integer, [-1,8,13]:Integer, [-1,8,14]:Integer, [-1,8,15]:Integer, [-1,8,16]:Integer, [-1,8,17]:Integer, [-1,8,18]:Integer, [-1,8,19]:Integer, [-1,8,20]:Integer, [-1,8,21]:Integer, [-1,8,22]:Integer, [-1,8,23]:Integer, [-1,8,24]:Integer, [-1,8,25]:Integer, [-1,8,26]:Integer, [-1,8,27]:Integer, [-1,8,28]:Integer, [-1,8,29]:Integer, [-1,8,30]:Integer, [-1,8,31]:Integer, [-1,8,32]:Integer, [-1,8,33]:Integer, [-1,8,34]:Integer, [-1,8,35]:Integer, [-1,8,36]:Integer, [-1,8,37]:Integer, [-1,8,38]:Integer, [-1,8,39]:Integer, [-1,16]:Pointer, [-1,16,0]:Pointer, [-1,16,0,-1]:Float@double, [-1,16,8]:Integer, [-1,16,9]:Integer, [-1,16,10]:Integer, [-1,16,11]:Integer, [-1,16,12]:Integer, [-1,16,13]:Integer, [-1,16,14]:Integer, [-1,16,15]:Integer, [-1,16,16]:Integer, [-1,16,17]:Integer, [-1,16,18]:Integer, [-1,16,19]:Integer, [-1,16,20]:Integer, [-1,16,21]:Integer, [-1,16,22]:Integer, [-1,16,23]:Integer, [-1,16,24]:Integer, [-1,16,25]:Integer, [-1,16,26]:Integer, [-1,16,27]:Integer, [-1,16,28]:Integer, [-1,16,29]:Integer, [-1,16,30]:Integer, [-1,16,31]:Integer, [-1,16,32]:Integer, [-1,16,33]:Integer, [-1,16,34]:Integer, [-1,16,35]:Integer, [-1,16,36]:Integer, [-1,16,37]:Integer, [-1,16,38]:Integer, [-1,16,39]:Integer, [-1,24]:Pointer, [-1,24,0]:Pointer, [-1,24,0,-1]:Integer, [-1,24,8]:Integer, [-1,24,9]:Integer, [-1,24,10]:Integer, [-1,24,11]:Integer, [-1,24,12]:Integer, [-1,24,13]:Integer, [-1,24,14]:Integer, [-1,24,15]:Integer, [-1,24,16]:Integer, [-1,24,17]:Integer, [-1,24,18]:Integer, [-1,24,19]:Integer, [-1,24,20]:Integer, [-1,24,21]:Integer, [-1,24,22]:Integer, [-1,24,23]:Integer, [-1,24,24]:Integer, [-1,24,25]:Integer, [-1,24,26]:Integer, [-1,24,27]:Integer, [-1,24,28]:Integer, [-1,24,29]:Integer, [-1,24,30]:Integer, [-1,24,31]:Integer, [-1,24,32]:Integer, [-1,24,33]:Integer, [-1,24,34]:Integer, [-1,24,35]:Integer, [-1,24,36]:Integer, [-1,24,37]:Integer, [-1,24,38]:Integer, [-1,24,39]:Integer, [-1,32]:Pointer, [-1,32,0]:Pointer, [-1,32,0,-1]:Integer, [-1,32,8]:Integer, [-1,32,9]:Integer, [-1,32,10]:Integer, [-1,32,11]:Integer, [-1,32,12]:Integer, [-1,32,13]:Integer, [-1,32,14]:Integer, [-1,32,15]:Integer, [-1,32,16]:Integer, [-1,32,17]:Integer, [-1,32,18]:Integer, [-1,32,19]:Integer, [-1,32,20]:Integer, [-1,32,21]:Integer, [-1,32,22]:Integer, [-1,32,23]:Integer, [-1,32,24]:Integer, [-1,32,25]:Integer, [-1,32,26]:Integer, [-1,32,27]:Integer, [-1,32,28]:Integer, [-1,32,29]:Integer, [-1,32,30]:Integer, [-1,32,31]:Integer, [-1,32,32]:Integer, [-1,32,33]:Integer, [-1,32,34]:Integer, [-1,32,35]:Integer, [-1,32,36]:Integer, [-1,32,37]:Integer, [-1,32,38]:Integer, [-1,32,39]:Integer, [-1,40]:Pointer, [-1,40,0]:Pointer, [-1,40,0,-1]:Float@double, [-1,40,8]:Integer, [-1,40,9]:Integer, [-1,40,10]:Integer, [-1,40,11]:Integer, [-1,40,12]:Integer, [-1,40,13]:Integer, [-1,40,14]:Integer, [-1,40,15]:Integer, [-1,40,16]:Integer, [-1,40,17]:Integer, [-1,40,18]:Integer, [-1,40,19]:Integer, [-1,40,20]:Integer, [-1,40,21]:Integer, [-1,40,22]:Integer, [-1,40,23]:Integer, [-1,40,24]:Integer, [-1,40,25]:Integer, [-1,40,26]:Integer, [-1,40,27]:Integer, [-1,40,28]:Integer, [-1,40,29]:Integer, [-1,40,30]:Integer, [-1,40,31]:Integer, [-1,40,32]:Integer, [-1,40,33]:Integer, [-1,40,34]:Integer, [-1,40,35]:Integer, [-1,40,36]:Integer, [-1,40,37]:Integer, [-1,40,38]:Integer, [-1,40,39]:Integer}" "enzymejl_parmtype"="139771446990736" "enzymejl_parmtype_ref"="1" %arg1) unnamed_addr #9 {
bb:
  %i = alloca [1 x [2 x double]], align 8
  %i2 = alloca { [1 x [2 x double]], double }, align 8, !enzymejl_allocart !2, !enzyme_type !3, !enzymejl_allocart_name !6, !enzymejl_source_type_Tuple\7BSVector\7B2\2C\20Float64\7D\2C\20Float64\7D !7, !enzymejl_byref_MUT_REF !7
  %i3 = call {}*** @julia.get_pgcstack()
  %i4 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]] addrspace(11)* %arg1, i64 0, i64 0, i64 2
  %i5 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i4, align 8, !enzyme_type !8, !enzymejl_byref_MUT_REF !7, !enzymejl_source_type_Vector\7BFloat64\7D !7
  %i6 = addrspacecast {} addrspace(10)* %arg to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)*
  %i7 = getelementptr inbounds { i8 addrspace(13)*, i64, i16, i16, i32 }, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)* %i6, i64 0, i32 0
  %i8 = load i8 addrspace(13)*, i8 addrspace(13)* addrspace(11)* %i7, align 16, !tbaa !11, !alias.scope !16, !noalias !21, !enzyme_type !3, !enzymejl_source_type_Ptr\7BSVector\7B2\2C\20Float64\7D\7D !7, !enzymejl_byref_BITS_VALUE !7
  %i9 = bitcast [1 x [2 x double]]* %i to i8*
  %i10 = addrspacecast [1 x [2 x double]]* %i to [1 x [2 x double]] addrspace(11)*
  %i11 = getelementptr inbounds { [1 x [2 x double]], double }, { [1 x [2 x double]], double }* %i2, i64 0, i32 1
  call void @llvm.memcpy.p0i8.p13i8.i64(i8* noundef nonnull align 8 dereferenceable(16) %i9, i8 addrspace(13)* noundef align 1 dereferenceable(16) %i8, i64 noundef 16, i1 noundef false), !tbaa !26, !alias.scope !27, !noalias !28
  %i12 = addrspacecast {} addrspace(10)* %i5 to double addrspace(13)* addrspace(11)*
  %i13 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %i12, align 8, !tbaa !11, !alias.scope !16, !noalias !21, !nonnull !7, !enzyme_type !3, !enzymejl_byref_BITS_VALUE !7, !enzymejl_source_type_Ptr\7BFloat64\7D !7
  %i14 = load double, double addrspace(13)* %i13, align 8, !tbaa !29, !alias.scope !32, !noalias !33, !enzyme_type !34, !enzymejl_byref_BITS_VALUE !7, !enzymejl_source_type_Float64 !7
  %i15 = load double, double* %i11, align 8, !tbaa !35, !alias.scope !37, !noalias !38, !enzyme_type !34, !enzymejl_byref_BITS_VALUE !7, !enzymejl_source_type_Float64 !7
  %i16 = fmul double %i14, %i15
  %i17 = load i8 addrspace(13)*, i8 addrspace(13)* addrspace(11)* %i7, align 16, !tbaa !11, !alias.scope !16, !noalias !21, !nonnull !7, !enzyme_type !3, !enzymejl_source_type_Ptr\7BSVector\7B2\2C\20Float64\7D\7D !7, !enzymejl_byref_BITS_VALUE !7
  %i18 = bitcast i8 addrspace(13)* %i17 to double addrspace(13)*
  store double %i16, double addrspace(13)* %i18, align 8, !tbaa !26, !alias.scope !39, !noalias !28
  %i19 = getelementptr inbounds i8, i8 addrspace(13)* %i17, i64 8
  %i20 = bitcast i8 addrspace(13)* %i19 to double addrspace(13)*
  store double 0.000000e+00, double addrspace(13)* %i20, align 8, !tbaa !26, !alias.scope !39, !noalias !28
  %i21 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]] addrspace(11)* %arg1, i64 0, i64 1
  %i22 = icmp eq [3 x {} addrspace(10)*] addrspace(11)* %i21, null
  %i23 = getelementptr inbounds [3 x {} addrspace(10)*], [3 x {} addrspace(10)*] addrspace(11)* %i21, i64 0, i64 0
  %i24 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i23, align 8, !tbaa !40, !alias.scope !42, !noalias !43, !enzyme_type !44, !enzymejl_byref_MUT_REF !7, !enzyme_inactive !7, !enzymejl_source_type_Vector\7BInt64\7D !7
  %i25 = insertvalue [3 x {} addrspace(10)*] undef, {} addrspace(10)* %i24, 0
  %i26 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]] addrspace(11)* %arg1, i64 0, i64 1, i64 1
  %i27 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i26, align 8, !tbaa !40, !alias.scope !42, !noalias !43, !enzyme_type !44, !enzymejl_byref_MUT_REF !7, !enzyme_inactive !7, !enzymejl_source_type_Vector\7BInt64\7D !7
  %i28 = insertvalue [3 x {} addrspace(10)*] %i25, {} addrspace(10)* %i27, 1
  %i29 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]] addrspace(11)* %arg1, i64 0, i64 1, i64 2
  %i30 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %i29, align 8, !tbaa !40, !alias.scope !42, !noalias !43, !enzyme_type !8, !enzymejl_byref_MUT_REF !7, !enzymejl_source_type_Vector\7BFloat64\7D !7
  %i31 = insertvalue [3 x {} addrspace(10)*] %i28, {} addrspace(10)* %i30, 2
  %i32 = select i1 %i22, [3 x {} addrspace(10)*] zeroinitializer, [3 x {} addrspace(10)*] %i31
  call void @llvm.memcpy.p0i8.p13i8.i64(i8* noundef nonnull align 8 dereferenceable(16) %i9, i8 addrspace(13)* noundef nonnull align 1 dereferenceable(16) %i17, i64 noundef 16, i1 noundef false), !tbaa !26, !alias.scope !27, !noalias !28
  call fastcc void @a1({ [1 x [2 x double]], double }* noalias nocapture nofree noundef nonnull writeonly sret({ [1 x [2 x double]], double }) align 8 dereferenceable(24) %i2, [1 x [2 x double]] addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(16) %i10)
  %i33 = extractvalue [3 x {} addrspace(10)*] %i32, 2, !enzyme_type !47
  %i34 = addrspacecast {} addrspace(10)* %i33 to double addrspace(13)* addrspace(11)*
  %i35 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %i34, align 8, !tbaa !11, !alias.scope !16, !noalias !21, !nonnull !7
  %i36 = load double, double addrspace(13)* %i35, align 8, !tbaa !29, !alias.scope !32, !noalias !33
  %i37 = load double, double* %i11, align 8, !tbaa !35, !alias.scope !37, !noalias !38, !enzyme_type !34, !enzymejl_byref_BITS_VALUE !7, !enzymejl_source_type_Float64 !7
  %i38 = fmul double %i36, %i37
  ret double %i38
}

define internal fastcc void @a1({ [1 x [2 x double]], double }* noalias nocapture nofree noundef nonnull writeonly sret({ [1 x [2 x double]], double }) align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %arg, [1 x [2 x double]] addrspace(11)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}" "enzymejl_parmtype"="139771433171344" "enzymejl_parmtype_ref"="1" %arg1) unnamed_addr #10 {
bb:
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn
define noalias nonnull {} addrspace(10)* @ijl_array_copy({} addrspace(10)* %arg) local_unnamed_addr #11 {
bb:
  ret {} addrspace(10)* %arg
}

declare void @__enzyme_autodiff(...)

define void @wrap([2 x [3 x {} addrspace(10)*]] %arg, {} addrspace(10)* %arg1, {} addrspace(10)* %arg2) {
bb:
  call void (...) @__enzyme_autodiff(double ([2 x [3 x {} addrspace(10)*]], {} addrspace(10)*)* @julia_test_forces_grad_159_inner.1, metadata !"enzyme_const", [2 x [3 x {} addrspace(10)*]] %arg, metadata !"enzyme_dup", {} addrspace(10)* %arg1, {} addrspace(10)* %arg2)
  ret void
}

define "enzyme_type"="{[-1]:Float@double}" "enzymejl_parmtype"="139771311454768" "enzymejl_parmtype_ref"="1" double @julia_test_forces_grad_159_inner.1([2 x [3 x {} addrspace(10)*]] "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer, [0,0,-1]:Integer, [8,0,-1]:Integer, [16,0,-1]:Float@double, [24,0,-1]:Integer, [32,0,-1]:Integer, [40,0,-1]:Float@double}" "enzymejl_parmtype"="139771446990736" "enzymejl_parmtype_ref"="0" %arg, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer}" "enzymejl_parmtype"="139771446999952" "enzymejl_parmtype_ref"="2" %arg1) local_unnamed_addr #12 {
bb:
  %i = alloca [2 x [3 x {} addrspace(10)*]], align 8
  %i2 = extractvalue [2 x [3 x {} addrspace(10)*]] %arg, 0, 0, !enzyme_type !44, !enzymejl_byref_MUT_REF !7, !enzyme_inactive !7, !enzymejl_source_type_Vector\7BInt64\7D !7
  %i3 = extractvalue [2 x [3 x {} addrspace(10)*]] %arg, 0, 1, !enzyme_type !44, !enzymejl_byref_MUT_REF !7, !enzyme_inactive !7, !enzymejl_source_type_Vector\7BInt64\7D !7
  %i4 = extractvalue [2 x [3 x {} addrspace(10)*]] %arg, 0, 2, !enzyme_type !8, !enzymejl_byref_MUT_REF !7, !enzymejl_source_type_Vector\7BFloat64\7D !7
  %i5 = extractvalue [2 x [3 x {} addrspace(10)*]] %arg, 1, 0, !enzyme_type !44, !enzymejl_byref_MUT_REF !7, !enzyme_inactive !7, !enzymejl_source_type_Vector\7BInt64\7D !7
  %i6 = extractvalue [2 x [3 x {} addrspace(10)*]] %arg, 1, 1, !enzyme_type !44, !enzymejl_byref_MUT_REF !7, !enzyme_inactive !7, !enzymejl_source_type_Vector\7BInt64\7D !7
  %i7 = extractvalue [2 x [3 x {} addrspace(10)*]] %arg, 1, 2, !enzyme_type !8, !enzymejl_byref_MUT_REF !7, !enzymejl_source_type_Vector\7BFloat64\7D !7
  %i8 = call {}*** @julia.get_pgcstack()
  %i9 = icmp ne {} addrspace(10)* %i2, null
  %i10 = icmp ne {} addrspace(10)* %i3, null
  %i11 = icmp ne {} addrspace(10)* %i4, null
  %i12 = call noalias nonnull "enzyme_ReadOnlyOrThrow" {} addrspace(10)* @ijl_array_copy({} addrspace(10)* nonnull %i4) #15
  %i13 = icmp ne {} addrspace(10)* %i5, null
  %i14 = icmp ne {} addrspace(10)* %i6, null
  %i15 = icmp ne {} addrspace(10)* %i7, null
  %i16 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]]* %i, i64 0, i64 0, i64 0
  store {} addrspace(10)* %i2, {} addrspace(10)** %i16, align 8, !noalias !49
  %i17 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]]* %i, i64 0, i64 0, i64 1
  store {} addrspace(10)* %i3, {} addrspace(10)** %i17, align 8, !noalias !49
  %i18 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]]* %i, i64 0, i64 0, i64 2
  store {} addrspace(10)* %i12, {} addrspace(10)** %i18, align 8, !noalias !49
  %i19 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]]* %i, i64 0, i64 1, i64 0
  store {} addrspace(10)* %i5, {} addrspace(10)** %i19, align 8, !noalias !49
  %i20 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]]* %i, i64 0, i64 1, i64 1
  store {} addrspace(10)* %i6, {} addrspace(10)** %i20, align 8, !noalias !49
  %i21 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]]* %i, i64 0, i64 1, i64 2
  store {} addrspace(10)* %i7, {} addrspace(10)** %i21, align 8, !noalias !49
  %i22 = addrspacecast [2 x [3 x {} addrspace(10)*]]* %i to [2 x [3 x {} addrspace(10)*]] addrspace(11)*
  %i23 = call fastcc double @a0({} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %arg1, [2 x [3 x {} addrspace(10)*]] addrspace(11)* nocapture noundef nonnull readonly align 8 dereferenceable(48) %i22)
  %i24 = addrspacecast {} addrspace(10)* %arg1 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)*
  %i25 = getelementptr inbounds { i8 addrspace(13)*, i64, i16, i16, i32 }, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)* %i24, i64 0, i32 1
  %i26 = load i64, i64 addrspace(11)* %i25, align 8, !tbaa !50, !range !52, !alias.scope !53, !noalias !21, !enzyme_type !54, !enzymejl_byref_BITS_VALUE !7, !enzyme_inactive !7, !enzymejl_source_type_UInt64 !7
  %i27 = icmp eq i64 %i26, 0
  br i1 %i27, label %bb28, label %bb29

bb28:                                             ; preds = %bb
  unreachable

bb29:                                             ; preds = %bb
  ret double %i23
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #13

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #13

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nofree readnone "enzyme_inactive" "enzyme_no_escaping_allocation" "enzyme_shouldrecompute" "enzymejl_world"="31534" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { inaccessiblemem_or_argmemonly nofree "enzyme_ReadOnlyOrThrow" "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="31534" }
attributes #3 = { inaccessiblememonly mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) "enzyme_ReadOnlyOrThrow" "enzyme_no_escaping_allocation" "enzymejl_world"="31534" }
attributes #4 = { inaccessiblememonly nofree norecurse nounwind "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="31534" }
attributes #5 = { nofree "enzymejl_world"="31534" }
attributes #6 = { noreturn "enzyme_no_escaping_allocation" "enzymejl_world"="31534" }
attributes #7 = { noreturn "enzymejl_world"="31534" }
attributes #8 = { argmemonly nocallback nofree nounwind willreturn }
attributes #9 = { noinline "enzyme_ta_norecur" "enzymejl_mi"="139771423616832" "enzymejl_rt"="139771311459520" "enzymejl_world"="31534" }
attributes #10 = { "enzyme_LocalReadOnlyOrThrow" "enzyme_ta_norecur" "enzymejl_mi"="139771423302816" "enzymejl_rt"="139771421471888" "enzymejl_world"="31534" }
attributes #11 = { mustprogress nofree nounwind willreturn "enzyme_LocalReadOnlyOrThrow" "enzyme_no_escaping_allocation" "enzymejl_needs_restoration"="139771609860491" }
attributes #12 = { "enzyme_ta_norecur" "enzymejl_mi"="139771425527200" "enzymejl_rt"="139771311454768" "enzymejl_world"="31534" }
attributes #13 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #14 = { inaccessiblememonly nocallback nofree nosync nounwind willreturn }
attributes #15 = { nounwind "enzyme_no_escaping_allocation" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{!"139771421471888"}
!3 = !{!"Unknown", i32 -1, !4}
!4 = !{!"Pointer", i32 -1, !5}
!5 = !{!"Float@double"}
!6 = !{!"Tuple{SVector{2, Float64}, Float64}"}
!7 = !{}
!8 = !{!"Unknown", i32 -1, !9}
!9 = !{!"Pointer", i32 0, !4, i32 8, !10, i32 9, !10, i32 10, !10, i32 11, !10, i32 12, !10, i32 13, !10, i32 14, !10, i32 15, !10, i32 16, !10, i32 17, !10, i32 18, !10, i32 19, !10, i32 20, !10, i32 21, !10, i32 22, !10, i32 23, !10, i32 24, !10, i32 25, !10, i32 26, !10, i32 27, !10, i32 28, !10, i32 29, !10, i32 30, !10, i32 31, !10, i32 32, !10, i32 33, !10, i32 34, !10, i32 35, !10, i32 36, !10, i32 37, !10, i32 38, !10, i32 39, !10}
!10 = !{!"Integer"}
!11 = !{!12, !12, i64 0}
!12 = !{!"jtbaa_arrayptr", !13, i64 0}
!13 = !{!"jtbaa_array", !14, i64 0}
!14 = !{!"jtbaa", !15, i64 0}
!15 = !{!"jtbaa"}
!16 = !{!17, !19}
!17 = distinct !{!17, !18, !"na_addr13"}
!18 = distinct !{!18, !"addr13"}
!19 = !{!"jnoalias_typemd", !20}
!20 = !{!"jnoalias"}
!21 = !{!22, !23, !24, !25}
!22 = !{!"jnoalias_gcframe", !20}
!23 = !{!"jnoalias_stack", !20}
!24 = !{!"jnoalias_data", !20}
!25 = !{!"jnoalias_const", !20}
!26 = !{!14, !14, i64 0}
!27 = !{!24, !23}
!28 = !{!17, !22, !19, !25}
!29 = !{!30, !30, i64 0}
!30 = !{!"jtbaa_arraybuf", !31, i64 0}
!31 = !{!"jtbaa_data", !14, i64 0}
!32 = !{!24}
!33 = !{!22, !23, !19, !25}
!34 = !{!"Unknown", i32 -1, !5}
!35 = !{!36, !36, i64 0}
!36 = !{!"jtbaa_stack", !14, i64 0}
!37 = !{!23}
!38 = !{!22, !24, !19, !25}
!39 = !{!23, !24}
!40 = !{!41, !41, i64 0, i64 0}
!41 = !{!"jtbaa_const", !14, i64 0}
!42 = !{!25}
!43 = !{!22, !23, !24, !19}
!44 = !{!"Unknown", i32 -1, !45}
!45 = !{!"Pointer", i32 0, !46, i32 8, !10, i32 9, !10, i32 10, !10, i32 11, !10, i32 12, !10, i32 13, !10, i32 14, !10, i32 15, !10, i32 16, !10, i32 17, !10, i32 18, !10, i32 19, !10, i32 20, !10, i32 21, !10, i32 22, !10, i32 23, !10, i32 24, !10, i32 25, !10, i32 26, !10, i32 27, !10, i32 28, !10, i32 29, !10, i32 30, !10, i32 31, !10, i32 32, !10, i32 33, !10, i32 34, !10, i32 35, !10, i32 36, !10, i32 37, !10, i32 38, !10, i32 39, !10}
!46 = !{!"Pointer", i32 -1, !10}
!47 = !{!"Unknown", i32 -1, !48}
!48 = !{!"Pointer"}
!49 = !{!17}
!50 = !{!51, !51, i64 0}
!51 = !{!"jtbaa_arraylen", !13, i64 0}
!52 = !{i64 0, i64 9223372036854775807}
!53 = !{!19}
!54 = !{!"Unknown", i32 -1, !10}

; CHECK: define internal fastcc void @diffea0
; CHECK-NEXT: bb:
; CHECK-NEXT:  %i2 = alloca { [1 x [2 x double]], double }, i64 1, align 8, !enzymejl_allocart !2, !enzyme_type !3, !enzymejl_allocart_name !6
; CHECK-NEXT:  %i3 = call {}*** @julia.get_pgcstack() #13
; CHECK-NEXT:  %"i5'il_phi" = extractvalue { {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)* } %tapeArg, 1
; CHECK-NEXT:  %"i6'ipc" = addrspacecast {} addrspace(10)* %"arg'" to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)*
; CHECK-NEXT:  %"i7'ipg" = getelementptr inbounds { i8 addrspace(13)*, i64, i16, i16, i32 }, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)* %"i6'ipc", i64 0, i32 0
; CHECK-NEXT:  %i11 = getelementptr inbounds { [1 x [2 x double]], double }, { [1 x [2 x double]], double }* %i2, i64 0, i32 1
; CHECK-NEXT:  %"i12'ipc" = addrspacecast {} addrspace(10)* %"i5'il_phi" to double addrspace(13)* addrspace(11)*
; CHECK-NEXT:  %"i13'ipl" = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %"i12'ipc", align 8, !tbaa !11, !alias.scope !101, !noalias !104, !nonnull !7
; CHECK-NEXT:  %i15 = load double, double* %i11, align 8, !tbaa !35, !alias.scope !106, !noalias !109, !enzyme_type !34, !enzymejl_byref_BITS_VALUE !7, !enzymejl_source_type_Float64 !7  %"i17'ipl" = load i8 addrspace(13)*, i8 addrspace(13)* addrspace(11)* %"i7'ipg", align 16, !tbaa !11, !alias.scope !111, !noalias !114, !nonnull !7
; CHECK-NEXT:  %"i18'ipc" = bitcast i8 addrspace(13)* %"i17'ipl" to double addrspace(13)*
; CHECK-NEXT:  %"i19'ipg" = getelementptr inbounds i8, i8 addrspace(13)* %"i17'ipl", i64 8
; CHECK-NEXT:  %"i20'ipc" = bitcast i8 addrspace(13)* %"i19'ipg" to double addrspace(13)*
; CHECK-NEXT:  %i21 = getelementptr inbounds [2 x [3 x {} addrspace(10)*]], [2 x [3 x {} addrspace(10)*]] addrspace(11)* %arg1, i64 0, i64 1
; CHECK-NEXT:  %i22 = icmp eq [3 x {} addrspace(10)*] addrspace(11)* %i21, null
; CHECK-NEXT:  %i24 = extractvalue { {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)* } %tapeArg, 2
; CHECK-NEXT:  %"i25'ipiv" = insertvalue [3 x {} addrspace(10)*] undef, {} addrspace(10)* %i24, 0
; CHECK-NEXT:  %i27 = extractvalue { {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)* } %tapeArg, 3
; CHECK-NEXT:  %"i28'ipiv" = insertvalue [3 x {} addrspace(10)*] %"i25'ipiv", {} addrspace(10)* %i27, 1
; CHECK-NEXT:  %"i30'il_phi" = extractvalue { {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)* } %tapeArg, 0
; CHECK-NEXT:  %"i31'ipiv" = insertvalue [3 x {} addrspace(10)*] %"i28'ipiv", {} addrspace(10)* %"i30'il_phi", 2
; CHECK-NEXT:  %"i32'ipse" = select i1 %i22, [3 x {} addrspace(10)*] zeroinitializer, [3 x {} addrspace(10)*] %"i31'ipiv"
; CHECK-NEXT:  %"i33'ipev" = extractvalue [3 x {} addrspace(10)*] %"i32'ipse", 2
; CHECK-NEXT:  %"i34'ipc" = addrspacecast {} addrspace(10)* %"i33'ipev" to double addrspace(13)* addrspace(11)*
; CHECK-NEXT:  %"i35'ipl" = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %"i34'ipc", align 8, !tbaa !11, !alias.scope !116, !noalias !119, !nonnull !7
; CHECK-NEXT:  %i37 = load double, double* %i11, align 8, !tbaa !35, !alias.scope !106, !noalias !109, !enzyme_type !34, !enzymejl_byref_BITS_VALUE !7, !enzymejl_source_type_Float64 !7  %0 = fmul fast double %differeturn, %i37
; CHECK-NEXT:  %1 = load double, double addrspace(13)* %"i35'ipl", align 8, !tbaa !29, !alias.scope !121, !noalias !124
; CHECK-NEXT:  %2 = fadd fast double %1, %0
; CHECK-NEXT:  store double %2, double addrspace(13)* %"i35'ipl", align 8, !tbaa !29, !alias.scope !121, !noalias !124
; CHECK-NEXT:  store double 0.000000e+00, double addrspace(13)* %"i20'ipc", align 8, !tbaa !26, !alias.scope !126, !noalias !129
; CHECK-NEXT:  %3 = load double, double addrspace(13)* %"i18'ipc", align 8, !tbaa !26, !alias.scope !126, !noalias !129
; CHECK-NEXT:  store double 0.000000e+00, double addrspace(13)* %"i18'ipc", align 8, !tbaa !26, !alias.scope !126, !noalias !129
; CHECK-NEXT:  %4 = fmul fast double %3, %i15
; CHECK-NEXT:  %5 = load double, double addrspace(13)* %"i13'ipl", align 8, !tbaa !29, !alias.scope !131, !noalias !134
; CHECK-NEXT:  %6 = fadd fast double %5, %4
; CHECK-NEXT:  store double %6, double addrspace(13)* %"i13'ipl", align 8, !tbaa !29, !alias.scope !131, !noalias !134
; CHECK-NEXT:  ret void
; CHECK-NEXT:}

