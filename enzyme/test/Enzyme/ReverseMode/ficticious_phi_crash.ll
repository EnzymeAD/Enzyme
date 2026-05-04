; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

@_j_str_invalid_GenericMemory_siz____3 = private unnamed_addr constant [108 x i8] c"invalid GenericMemory size: the number of elements is either negative or too large for system address width\00", align 1
@jl_small_typeof = external local_unnamed_addr constant i8
@jl_nothing = external local_unnamed_addr constant ptr
@"ejl_inserted$_Core_throw_inexacterror_3991$false$4868049760" = external global {}
@"ejl_inserted$jl_global_3962$false$4810921328" = external global {}
@"ejl_inserted$jl_sym_trunc_3992$false$4417812112" = external global {}
@"ejl_inserted$jl_global_3989$false$4940597312" = external global {}
@"ejl_inserted$_Core_throw_inexacterror_3961$false$4810976704" = external global {}
@"ejl_inserted$jl_sym_convert_3963$false$4417663248" = external global {}
@"ejl_inserted$jl_global_3975$false$4940591872" = external global {}
@"ejl_inserted$jl_global_3976$false$4940591824" = external global {}
@"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" = external global {}
@"ejl_inserted$_Core_Tuple_3978$false$4846611664" = external global {}
@"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" = external global {}
@"ejl_inserted$jl_global_3968$false$4900287488" = external global {}
@"ejl_inserted$_Core_ArgumentError_3969$false$4895492080" = external global {}
@"ejl_inserted$jl_global_3957$false$4866807728" = external global {}
@"ejl_inserted$_Core_Array_3959$false$4896072880" = external global {}
@"ejl_inserted$_Core_GenericMemoryRef_3960$false$4867651680" = external global {}
@"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824" = external global {}
@"ejl_inserted$jl_global_3970$false$4627012928" = external global {}
@"ejl_inserted$_Core_Array_3972$false$5309578256" = external global {}
@"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" = external global {}
@"ejl_inserted$_Core_BoundsError_4000$false$4587528272" = external global {}
@"ejl_inserted$_Core_BoundsError_3996$false$4895491232" = external global {}
@"ejl_inserted$_Core_Tuple_4001$false$4810594496" = external global {}
@"ejl_inserted$_Main_Base_string_4030$false$4856608912" = external global {}
@"ejl_inserted$jl_global_4031$false$4810542960" = external global {}
@"ejl_inserted$_Core_BoundsError_3995$false$4811961792" = external global {}
@"ejl_inserted$_Core_Tuple_3997$false$4810667440" = external global {}
@"ejl_inserted$_Main_Base_ReshapedArray_4014$false$5428434384" = external global {}
@"ejl_inserted$_Core_BoundsError_4010$false$4861657120" = external global {}
@"ejl_inserted$_Main_Base_UnitRange_4011$false$5462866832" = external global {}
@"ejl_inserted$_Core_Array_4021$false$5375415824" = external global {}
@"ejl_inserted$jl_global_4041$false$4964413504" = external global {}
@"ejl_inserted$jl_global_4042$false$4964413440" = external global {}
@"ejl_inserted$jl_global_4043$false$4900295312" = external global {}

; Function Attrs: alwaysinline
define "enzyme_type"="{[-1]:Float@float}" float @julia_core_3955(ptr noundef nonnull align 8 dereferenceable(40) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@float, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@float, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer}" "enzymejl_parmtype"="5309578256" "enzymejl_parmtype_ref"="2" "enzymejl_parmtype_str"="Array{Float32, 3}" %"pred::Array", ptr noundef nonnull align 8 dereferenceable(24) "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Integer, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4896073008" "enzymejl_parmtype_ref"="2" "enzymejl_parmtype_str"="Vector{UInt32}" %"y::Array") local_unnamed_addr #0 {
top:
  %jlcallframe9138 = alloca [3 x ptr], align 8
  %gcframe9139 = alloca [4 x ptr], align 16
  call void @llvm.memset.p0.i64(ptr align 16 %gcframe9139, i8 0, i64 32, i1 true)
  %newly_emitted_pgc_stack = call ptr inttoptr (i64 4338679580 to ptr)(i64 4338679616) #20
  store i64 8, ptr %gcframe9139, align 8
  %task.gcstack = load ptr, ptr %newly_emitted_pgc_stack, align 8
  %frame.prev = getelementptr inbounds ptr, ptr %gcframe9139, i64 1
  store ptr %task.gcstack, ptr %frame.prev, align 8
  store ptr %gcframe9139, ptr %newly_emitted_pgc_stack, align 8
  %"new::UnitRange" = alloca [2 x i32], align 4
  %"new::Tuple" = alloca [3 x i64], align 8
  %"new::ReshapedArray" = alloca { [2 x i32], [3 x i64] }, align 8
  %"new::Tuple113" = alloca [1 x i64], align 8
  %"new::Tuple428" = alloca [1 x i64], align 8
  %"new::Tuple628" = alloca [3 x i64], align 8
  %"new::Tuple667" = alloca [3 x i64], align 8
  %"new::Tuple703" = alloca [3 x i64], align 8
  %"new::Tuple738" = alloca [1 x i64], align 8
  %"new::Tuple741" = alloca [1 x i64], align 8
  %"new::Tuple888" = alloca [1 x i64], align 8
  %"new::Tuple976" = alloca [3 x i64], align 8
  %"new::Tuple983" = alloca [3 x i64], align 8
  %"new::Tuple1019" = alloca [3 x i64], align 8
  %"new::Tuple1054" = alloca [1 x i64], align 8
  %"new::Tuple1190" = alloca [1 x i64], align 8
  %ptls_field = getelementptr inbounds i8, ptr %newly_emitted_pgc_stack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %i = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %i, align 8
  fence syncscope("singlethread") seq_cst
  %i1 = load volatile i64, ptr %safepoint, align 8
  fence syncscope("singlethread") seq_cst
  %"pred::Array.size_ptr" = getelementptr inbounds i8, ptr %"pred::Array", i64 16
  %"pred::Array.size.sroa.0.0.copyload" = load i64, ptr %"pred::Array.size_ptr", align 8
  %"pred::Array.size.sroa.4.0.pred::Array.size_ptr.sroa_idx" = getelementptr inbounds i8, ptr %"pred::Array", i64 24
  %i2 = add i64 %"pred::Array.size.sroa.0.0.copyload", -2147483648
  %i3 = icmp ult i64 %i2, -4294967296
  br i1 %i3, label %L8, label %L12

L8:                                               ; preds = %top
  %i4 = load ptr, ptr getelementptr inbounds (i8, ptr @jl_small_typeof, i64 240), align 8
  %box_Int64 = call noalias nonnull align 8 dereferenceable(8) "enzyme_ReadOnlyOrThrow" "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}" ptr @ijl_box_int64(i64 signext %"pred::Array.size.sroa.0.0.copyload") #21
  %gc_slot_addr_0 = getelementptr inbounds ptr, ptr %gcframe9139, i64 2
  store ptr %box_Int64, ptr %gc_slot_addr_0, align 8
  store ptr @"ejl_inserted$jl_sym_trunc_3992$false$4417812112", ptr %jlcallframe9138, align 8
  %i5 = getelementptr inbounds ptr, ptr %jlcallframe9138, i64 1
  store ptr %i4, ptr %i5, align 8
  %i6 = getelementptr inbounds ptr, ptr %jlcallframe9138, i64 2
  store ptr %box_Int64, ptr %i6, align 8
  %i7 = call nonnull "enzyme_type"="{[-1]:Pointer}" ptr @ijl_invoke(ptr @"ejl_inserted$jl_global_3962$false$4810921328", ptr nonnull %jlcallframe9138, i32 3, ptr @"ejl_inserted$_Core_throw_inexacterror_3991$false$4868049760") #8
  call void @llvm.trap()
  unreachable

L12:                                              ; preds = %top
  %i8 = trunc i64 %"pred::Array.size.sroa.0.0.copyload" to i32
  %. = call i32 @llvm.smax.i32(i32 %i8, i32 0)
  store i32 1, ptr %"new::UnitRange", align 4
  %i9 = getelementptr inbounds i8, ptr %"new::UnitRange", i64 4
  store i32 %., ptr %i9, align 4
  %i10 = zext nneg i32 %. to i64
  %i11 = icmp sgt i32 %i8, 0
  %value_phi2 = select i1 %i11, i64 %i10, i64 0
  store i64 %value_phi2, ptr %"new::Tuple", align 8
  %i12 = getelementptr inbounds i8, ptr %"new::Tuple", i64 8
  store <2 x i64> <i64 1, i64 1>, ptr %i12, align 8
  %.not = icmp eq i64 %value_phi2, %i10
  br i1 %.not, label %L46, label %L48

L46:                                              ; preds = %L12
  %i13 = load i64, ptr %"new::UnitRange", align 4
  store i64 %i13, ptr %"new::ReshapedArray", align 8
  %i14 = getelementptr inbounds i8, ptr %"new::ReshapedArray", i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %i14, ptr noundef nonnull align 8 dereferenceable(24) %"new::Tuple", i64 24, i1 false)
  %"y::Array.size_ptr" = getelementptr inbounds i8, ptr %"y::Array", i64 16
  %"y::Array.size.0.copyload" = load i64, ptr %"y::Array.size_ptr", align 8
  %memorynew_empty = icmp eq i64 %"y::Array.size.0.copyload", 0
  br i1 %memorynew_empty, label %L46.retval_crit_edge, label %nonemptymem

L46.retval_crit_edge:                             ; preds = %L46
  br label %retval

L48:                                              ; preds = %L12
  call fastcc void @julia__throw_dmrs_4039(i64 signext %i10, ptr nocapture readonly %"new::Tuple") #22
  unreachable

L79:                                              ; preds = %load50, %L79.preheader
  %value_phi14 = phi i64 [ %i319, %load50 ], [ 1, %L79.preheader ]
  %i15 = add nsw i64 %value_phi14, -1
  %exitcond.not = icmp eq i64 %value_phi14, %i313
  br i1 %exitcond.not, label %L92, label %L95

L92:                                              ; preds = %L79
  store i64 %i313, ptr %"new::Tuple1190", align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_4037(ptr %"y::Array", ptr nocapture readonly %"new::Tuple1190") #22
  unreachable

L95:                                              ; preds = %L79
  %i16 = add nuw nsw i64 %i15, %memory_len
  %memoryref_ovflw.not = icmp ult i64 %i16, %i314
  %memoryref_byteoffset = shl i64 %i15, 2
  %memoryref_data_byteoffset = getelementptr i8, ptr %memoryref_data, i64 %memoryref_byteoffset
  %i17 = ptrtoint ptr %memoryref_data_byteoffset to i64
  %i18 = sub i64 %i17, %i315
  %memoryref_isinbounds = icmp ult i64 %i18, %memoryref_bytelen
  %"memoryref_isinbounds&notovflw" = and i1 %memoryref_ovflw.not, %memoryref_isinbounds
  br i1 %"memoryref_isinbounds&notovflw", label %idxend, label %oob

L104:                                             ; preds = %load
  %i19 = load ptr, ptr getelementptr inbounds (i8, ptr @jl_small_typeof, i64 240), align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  %box_UInt32 = call nonnull align 8 dereferenceable(4) "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}" ptr @ijl_box_uint32(i32 zeroext %i317) #23
  store ptr %box_UInt32, ptr %gc_slot_addr_08726, align 8
  store ptr @"ejl_inserted$jl_sym_convert_3963$false$4417663248", ptr %jlcallframe9138, align 8
  %i20 = getelementptr inbounds ptr, ptr %jlcallframe9138, i64 1
  store ptr %i19, ptr %i20, align 8
  %i21 = getelementptr inbounds ptr, ptr %jlcallframe9138, i64 2
  store ptr %box_UInt32, ptr %i21, align 8
  %i22 = call nonnull "enzyme_type"="{[-1]:Pointer}" ptr @ijl_invoke(ptr @"ejl_inserted$jl_global_3962$false$4810921328", ptr nonnull %jlcallframe9138, i32 3, ptr @"ejl_inserted$_Core_throw_inexacterror_3961$false$4810976704") #8
  call void @llvm.trap()
  unreachable

L126:                                             ; preds = %load
  %memory_len31 = load i64, ptr %memoryref_mem47, align 8
  %i23 = add i64 %memory_len31, %i15
  %i24 = shl nuw nsw i64 %memory_len31, 1
  %memoryref_ovflw32.not = icmp ult i64 %i23, %i24
  %memoryref_bytelen38 = shl nuw nsw i64 %memory_len31, 2
  %memoryref_isinbounds39 = icmp ult i64 %memoryref_byteoffset, %memoryref_bytelen38
  %"memoryref_isinbounds&notovflw40" = and i1 %memoryref_ovflw32.not, %memoryref_isinbounds39
  br i1 %"memoryref_isinbounds&notovflw40", label %idxend45, label %oob41

L144:                                             ; preds = %load50, %retval
  store ptr %"new::Array", ptr %gc_slot_addr_08726, align 8
  %i25 = call fastcc nonnull ptr @julia_reshape_4017(ptr %"new::Array")
  %.size_ptr = getelementptr inbounds i8, ptr %i25, i64 16
  %.size.sroa.0.0.copyload = load i64, ptr %.size_ptr, align 8
  %.size.sroa.2.0..size_ptr.sroa_idx = getelementptr inbounds i8, ptr %i25, i64 24
  %.size.sroa.2.0.copyload = load i64, ptr %.size.sroa.2.0..size_ptr.sroa_idx, align 8
  %.size.sroa.3.0..size_ptr.sroa_idx = getelementptr inbounds i8, ptr %i25, i64 32
  %.size.sroa.3.0.copyload = load i64, ptr %.size.sroa.3.0..size_ptr.sroa_idx, align 8
  %i26 = icmp eq i64 %.size.sroa.0.0.copyload, %i10
  %i27 = icmp eq i32 %i8, 1
  %i28 = or i1 %i27, %i26
  br i1 %i28, label %L230, label %L164

L164:                                             ; preds = %L144
  %value_phi1168.v.not = icmp eq i64 %.size.sroa.0.0.copyload, 1
  br i1 %value_phi1168.v.not, label %L230, label %L172

L172:                                             ; preds = %L164
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", i64 16) ]
  %ptls_load8831 = load ptr, ptr %ptls_field, align 8
  %"new::LazyString1175" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8831, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" to i64)) #15
  %"new::LazyString1175.tag_addr" = getelementptr inbounds ptr, ptr %"new::LazyString1175", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", ptr %"new::LazyString1175.tag_addr" unordered, align 8
  store <2 x ptr> zeroinitializer, ptr %"new::LazyString1175", align 8
  store ptr %"new::LazyString1175", ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", i64 16) ]
  %ptls_load8834 = load ptr, ptr %ptls_field, align 8
  %"box::Tuple1179" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8834, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664" to i64)) #15
  %"box::Tuple1179.tag_addr" = getelementptr inbounds ptr, ptr %"box::Tuple1179", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", ptr %"box::Tuple1179.tag_addr" unordered, align 8
  store atomic ptr @"ejl_inserted$jl_global_3975$false$4940591872", ptr %"box::Tuple1179" unordered, align 8
  %i29 = getelementptr inbounds i8, ptr %"box::Tuple1179", i64 8
  store i64 %i10, ptr %i29, align 8
  %i30 = getelementptr inbounds i8, ptr %"box::Tuple1179", i64 16
  store atomic ptr @"ejl_inserted$jl_global_3976$false$4940591824", ptr %i30 unordered, align 8
  %i31 = getelementptr inbounds i8, ptr %"box::Tuple1179", i64 24
  store i64 %.size.sroa.0.0.copyload, ptr %i31, align 8
  store atomic ptr %"box::Tuple1179", ptr %"new::LazyString1175" release, align 8
  %"new::LazyString1175.tag" = load atomic volatile i64, ptr %"new::LazyString1175.tag_addr" unordered, align 8
  %parent_bits = and i64 %"new::LazyString1175.tag", 3
  %parent_old_marked = icmp eq i64 %parent_bits, 3
  br i1 %parent_old_marked, label %may_trigger_wb, label %bb

may_trigger_wb:                                   ; preds = %L172
  %"box::Tuple1179.tag" = load atomic volatile i64, ptr %"box::Tuple1179.tag_addr" unordered, align 8
  %child_bit = and i64 %"box::Tuple1179.tag", 1
  %child_not_marked = icmp eq i64 %child_bit, 0
  br i1 %child_not_marked, label %trigger_wb, label %bb

trigger_wb:                                       ; preds = %may_trigger_wb
  call void @ijl_gc_queue_root(ptr nonnull %"new::LazyString1175")
  br label %bb

bb:                                               ; preds = %trigger_wb, %may_trigger_wb, %L172
  %i32 = getelementptr inbounds i8, ptr %"new::LazyString1175", i64 8
  %jl_nothing1180 = load ptr, ptr @jl_nothing, align 8
  store atomic ptr %jl_nothing1180, ptr %i32 release, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", i64 16) ]
  %"box::DimensionMismatch1184" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8834, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" to i64)) #15
  %"box::DimensionMismatch1184.tag_addr" = getelementptr inbounds ptr, ptr %"box::DimensionMismatch1184", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", ptr %"box::DimensionMismatch1184.tag_addr" unordered, align 8
  store atomic ptr %"new::LazyString1175", ptr %"box::DimensionMismatch1184" unordered, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::DimensionMismatch1184")
  unreachable

L230:                                             ; preds = %L164, %L144
  %value_phi170.size232.sroa.0.0.copyload = phi i64 [ %i10, %L164 ], [ %.size.sroa.0.0.copyload, %L144 ]
  %i33 = icmp ugt i64 %value_phi170.size232.sroa.0.0.copyload, 9223372036854775806
  %i34 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %value_phi170.size232.sroa.0.0.copyload, i64 %.size.sroa.2.0.copyload)
  %i35 = extractvalue { i64, i1 } %i34, 0
  %i36 = extractvalue { i64, i1 } %i34, 1
  %i37 = icmp ne i64 %.size.sroa.2.0.copyload, 0
  %i38 = icmp ugt i64 %.size.sroa.2.0.copyload, 9223372036854775806
  %i39 = or i1 %i38, %i33
  %i40 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %i35, i64 %.size.sroa.3.0.copyload)
  %i41 = extractvalue { i64, i1 } %i40, 0
  %i42 = extractvalue { i64, i1 } %i40, 1
  %i43 = icmp ne i64 %.size.sroa.3.0.copyload, 0
  %.not3895 = and i1 %i37, %i43
  %i44 = or i1 %i36, %i42
  %i45 = icmp ugt i64 %.size.sroa.3.0.copyload, 9223372036854775806
  %i46 = or i1 %i45, %i39
  %i47 = and i1 %.not3895, %i44
  %i48 = or i1 %i46, %i47
  br i1 %i48, label %L255, label %L259

L255:                                             ; preds = %L230
  %i49 = call fastcc [1 x ptr] @julia_ArgumentError_4002()
  %i50 = extractvalue [1 x ptr] %i49, 0
  store ptr %i50, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_ArgumentError_3969$false$4895492080", i64 16) ]
  %ptls_load8841 = load ptr, ptr %ptls_field, align 8
  %"box::ArgumentError" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8841, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Core_ArgumentError_3969$false$4895492080" to i64)) #15
  %"box::ArgumentError.tag_addr" = getelementptr inbounds ptr, ptr %"box::ArgumentError", i64 -1
  store atomic ptr @"ejl_inserted$_Core_ArgumentError_3969$false$4895492080", ptr %"box::ArgumentError.tag_addr" unordered, align 8
  store ptr %i50, ptr %"box::ArgumentError", align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::ArgumentError")
  unreachable

L259:                                             ; preds = %L230
  %memorynew_empty76 = icmp eq i64 %i41, 0
  br i1 %memorynew_empty76, label %L259.retval84_crit_edge, label %nonemptymem78

L259.retval84_crit_edge:                          ; preds = %L259
  br label %retval84

L381:                                             ; preds = %retval84
  %i51 = icmp slt i32 %i8, 1
  br i1 %i51, label %L864, label %L400.preheader

L400.preheader:                                   ; preds = %L381
  %i52 = getelementptr inbounds { ptr, ptr }, ptr %i25, i64 0, i32 1
  %i53 = mul i64 %.size.sroa.3.0.copyload, %.size.sroa.2.0.copyload
  %i54 = mul i64 %i53, %value_phi170.size232.sroa.0.0.copyload
  %i55 = add nuw nsw i64 %i10, 1
  %i56 = add i64 %i54, 1
  br label %L400

L400:                                             ; preds = %load165, %L400.preheader
  %value_phi111 = phi i64 [ %i328, %load165 ], [ 1, %L400.preheader ]
  %i57 = add nsw i64 %value_phi111, -1
  %exitcond3459.not = icmp eq i64 %value_phi111, %i55
  br i1 %exitcond3459.not, label %L416, label %L461

L416:                                             ; preds = %L400
  store i64 %i55, ptr %"new::Tuple113", align 8
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_4012(ptr nocapture readonly %"new::ReshapedArray", ptr nocapture readonly %"new::Tuple113") #22
  unreachable

L461:                                             ; preds = %L400
  %i58 = trunc i64 %value_phi111 to i32
  %memoryref_data117 = load ptr, ptr %i25, align 8
  %memoryref_mem137 = load ptr, ptr %i52, align 8
  %memory_len120 = load i64, ptr %memoryref_mem137, align 8
  %i59 = add nuw nsw i64 %memory_len120, %i57
  %i60 = shl nuw nsw i64 %memory_len120, 1
  %memoryref_ovflw121.not = icmp ult i64 %i59, %i60
  %memoryref_byteoffset122 = shl i64 %i57, 2
  %memoryref_data_byteoffset123 = getelementptr i8, ptr %memoryref_data117, i64 %memoryref_byteoffset122
  %memory_data_ptr125 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem137, i64 0, i32 1
  %memory_data126 = load ptr, ptr %memory_data_ptr125, align 8
  %i61 = ptrtoint ptr %memoryref_data_byteoffset123 to i64
  %i62 = ptrtoint ptr %memory_data126 to i64
  %i63 = sub i64 %i61, %i62
  %memoryref_bytelen127 = shl nuw nsw i64 %memory_len120, 2
  %memoryref_isinbounds128 = icmp ult i64 %i63, %memoryref_bytelen127
  %"memoryref_isinbounds&notovflw129" = and i1 %memoryref_ovflw121.not, %memoryref_isinbounds128
  br i1 %"memoryref_isinbounds&notovflw129", label %idxend135, label %oob130

L484:                                             ; preds = %load140
  store i64 %i56, ptr %"new::Tuple888", align 8
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  store ptr %"new::Array91", ptr %gc_slot_addr_18736, align 8
  call fastcc void @julia_throw_boundserror_3993(ptr %"new::Array91", ptr nocapture readonly %"new::Tuple888") #22
  unreachable

L487:                                             ; preds = %load140
  %memory_len146 = load i64, ptr %memoryref_mem162, align 8
  %i64 = add nuw nsw i64 %memory_len146, %i57
  %i65 = shl nuw nsw i64 %memory_len146, 1
  %memoryref_ovflw147.not = icmp ult i64 %i64, %i65
  %memoryref_bytelen153 = shl nuw nsw i64 %memory_len146, 2
  %memoryref_isinbounds154 = icmp ult i64 %memoryref_byteoffset122, %memoryref_bytelen153
  %"memoryref_isinbounds&notovflw155" = and i1 %memoryref_ovflw147.not, %memoryref_isinbounds154
  br i1 %"memoryref_isinbounds&notovflw155", label %idxend160, label %oob156

L505:                                             ; preds = %retval84
  %i66 = icmp eq i64 %.size93.sroa.0.0.copyload, 1
  %narrow = or i1 %i27, %.not1666
  %brmerge = or i1 %i66, %narrow
  %.size93.sroa.0.0.copyload.mux = select i1 %i27, i64 %.size93.sroa.0.0.copyload, i64 %i10
  br i1 %brmerge, label %L607, label %L547

L547:                                             ; preds = %L505
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", i64 16) ]
  %ptls_load8851 = load ptr, ptr %ptls_field, align 8
  %"new::LazyString1110" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8851, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" to i64)) #15
  %"new::LazyString1110.tag_addr" = getelementptr inbounds ptr, ptr %"new::LazyString1110", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", ptr %"new::LazyString1110.tag_addr" unordered, align 8
  store <2 x ptr> zeroinitializer, ptr %"new::LazyString1110", align 8
  store ptr %"new::LazyString1110", ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", i64 16) ]
  %ptls_load8854 = load ptr, ptr %ptls_field, align 8
  %"box::Tuple1114" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8854, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664" to i64)) #15
  %"box::Tuple1114.tag_addr" = getelementptr inbounds ptr, ptr %"box::Tuple1114", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", ptr %"box::Tuple1114.tag_addr" unordered, align 8
  store atomic ptr @"ejl_inserted$jl_global_3975$false$4940591872", ptr %"box::Tuple1114" unordered, align 8
  %i67 = getelementptr inbounds i8, ptr %"box::Tuple1114", i64 8
  store i64 %i10, ptr %i67, align 8
  %i68 = getelementptr inbounds i8, ptr %"box::Tuple1114", i64 16
  store atomic ptr @"ejl_inserted$jl_global_3976$false$4940591824", ptr %i68 unordered, align 8
  %i69 = getelementptr inbounds i8, ptr %"box::Tuple1114", i64 24
  store i64 %.size93.sroa.0.0.copyload, ptr %i69, align 8
  store atomic ptr %"box::Tuple1114", ptr %"new::LazyString1110" release, align 8
  %"new::LazyString1110.tag" = load atomic volatile i64, ptr %"new::LazyString1110.tag_addr" unordered, align 8
  %parent_bits9082 = and i64 %"new::LazyString1110.tag", 3
  %parent_old_marked9083 = icmp eq i64 %parent_bits9082, 3
  br i1 %parent_old_marked9083, label %may_trigger_wb9084, label %bb70

may_trigger_wb9084:                               ; preds = %L547
  %"box::Tuple1114.tag" = load atomic volatile i64, ptr %"box::Tuple1114.tag_addr" unordered, align 8
  %child_bit9086 = and i64 %"box::Tuple1114.tag", 1
  %child_not_marked9087 = icmp eq i64 %child_bit9086, 0
  br i1 %child_not_marked9087, label %trigger_wb9088, label %bb70

trigger_wb9088:                                   ; preds = %may_trigger_wb9084
  call void @ijl_gc_queue_root(ptr nonnull %"new::LazyString1110")
  br label %bb70

bb70:                                             ; preds = %trigger_wb9088, %may_trigger_wb9084, %L547
  %i71 = getelementptr inbounds i8, ptr %"new::LazyString1110", i64 8
  %jl_nothing1115 = load ptr, ptr @jl_nothing, align 8
  store atomic ptr %jl_nothing1115, ptr %i71 release, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", i64 16) ]
  %"box::DimensionMismatch1119" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8854, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" to i64)) #15
  %"box::DimensionMismatch1119.tag_addr" = getelementptr inbounds ptr, ptr %"box::DimensionMismatch1119", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", ptr %"box::DimensionMismatch1119.tag_addr" unordered, align 8
  store atomic ptr %"new::LazyString1110", ptr %"box::DimensionMismatch1119" unordered, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::DimensionMismatch1119")
  unreachable

L607:                                             ; preds = %L505
  %i72 = icmp eq i64 %.size93.sroa.0.0.copyload.mux, 0
  %i73 = icmp eq i64 %.size93.sroa.2.0.copyload, 0
  %narrow1684.not = select i1 %i72, i1 true, i1 %i73
  %i74 = icmp eq i64 %.size93.sroa.3.0.copyload, 0
  %or.cond1766 = select i1 %narrow1684.not, i1 true, i1 %i74
  br i1 %or.cond1766, label %L864, label %L620.preheader

L620.preheader:                                   ; preds = %L607
  %i75 = getelementptr inbounds i8, ptr %"new::Tuple976", i64 8
  %i76 = getelementptr inbounds i8, ptr %"new::Tuple983", i64 8
  %i77 = getelementptr inbounds i8, ptr %"new::Tuple983", i64 16
  %i78 = getelementptr inbounds { ptr, ptr }, ptr %i25, i64 0, i32 1
  %i79 = getelementptr inbounds i8, ptr %"new::Tuple1019", i64 8
  %i80 = getelementptr inbounds i8, ptr %"new::Tuple1019", i64 16
  %.not1687.peel.not = icmp slt i32 %i8, 1
  %i81 = icmp ne i64 %.size93.sroa.0.0.copyload, 0
  %i82 = mul i64 %.size93.sroa.2.0.copyload, %.size93.sroa.0.0.copyload
  %i83 = mul i64 %i82, %.size93.sroa.3.0.copyload
  %i84 = icmp ne i64 %value_phi170.size232.sroa.0.0.copyload, 0
  %.not1695.peel = icmp eq i64 %.size93.sroa.0.0.copyload.mux, 1
  br label %L620

L620:                                             ; preds = %L855, %L620.preheader
  %value_phi972 = phi i64 [ %i85, %L855 ], [ 0, %L620.preheader ]
  %i85 = add nuw i64 %value_phi972, 1
  %i86 = select i1 %.not1668, i64 1, i64 %i85
  %i87 = add i64 %i86, -1
  %i88 = icmp ult i64 %i87, %.size93.sroa.3.0.copyload
  %i89 = icmp ult i64 %value_phi972, %.size.sroa.3.0.copyload
  %i90 = mul i64 %value_phi972, %.size.sroa.2.0.copyload
  %i91 = mul i64 %i87, %i82
  br i1 %.not1687.peel.not, label %L644, label %L622

L622:                                             ; preds = %L850, %L620
  %value_phi973 = phi i64 [ %i92, %L850 ], [ 0, %L620 ]
  %i92 = add nuw i64 %value_phi973, 1
  %i93 = select i1 %.not1667, i64 1, i64 %i92
  %i94 = add i64 %i93, -1
  %i95 = icmp ult i64 %value_phi973, %.size.sroa.2.0.copyload
  %i96 = and i1 %i89, %i95
  %memoryref_data1028 = load ptr, ptr %"new::Array91", align 8
  %memoryref_mem1048 = load ptr, ptr %i322, align 8
  %reass.add = add i64 %value_phi973, %i90
  %reass.mul = mul i64 %reass.add, %value_phi170.size232.sroa.0.0.copyload
  %memory_data_ptr1036 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem1048, i64 0, i32 1
  %i97 = icmp ult i64 %i94, %.size93.sroa.2.0.copyload
  %i98 = and i1 %i88, %i97
  %i99 = and i1 %i81, %i98
  br i1 %i99, label %L740.peel, label %L737

L740.peel:                                        ; preds = %L622
  %i100 = mul i64 %i94, %.size93.sroa.0.0.copyload
  %i101 = add i64 %i100, %i91
  %.not1690.peel = icmp ult i64 %i101, %i83
  br i1 %.not1690.peel, label %L773.peel, label %L770.loopexit4481

L773.peel:                                        ; preds = %L740.peel
  %memoryref_data994.peel = load ptr, ptr %i25, align 8
  %memoryref_mem1014.peel = load ptr, ptr %i78, align 8
  %memory_len997.peel = load i64, ptr %memoryref_mem1014.peel, align 8
  %i102 = add i64 %memory_len997.peel, %i101
  %i103 = shl nuw nsw i64 %memory_len997.peel, 1
  %memoryref_ovflw998.not.peel = icmp ult i64 %i102, %i103
  %memoryref_byteoffset999.peel = shl i64 %i101, 2
  %memoryref_data_byteoffset1000.peel = getelementptr i8, ptr %memoryref_data994.peel, i64 %memoryref_byteoffset999.peel
  %memory_data_ptr1002.peel = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem1014.peel, i64 0, i32 1
  %memory_data1003.peel = load ptr, ptr %memory_data_ptr1002.peel, align 8
  %i104 = ptrtoint ptr %memoryref_data_byteoffset1000.peel to i64
  %i105 = ptrtoint ptr %memory_data1003.peel to i64
  %i106 = sub i64 %i104, %i105
  %memoryref_bytelen1004.peel = shl nuw nsw i64 %memory_len997.peel, 2
  %memoryref_isinbounds1005.peel = icmp ult i64 %i106, %memoryref_bytelen1004.peel
  %"memoryref_isinbounds&notovflw1006.peel" = and i1 %memoryref_ovflw998.not.peel, %memoryref_isinbounds1005.peel
  br i1 %"memoryref_isinbounds&notovflw1006.peel", label %idxend1012.peel, label %oob1007.loopexit4482

idxend1012.peel:                                  ; preds = %L773.peel
  %i107 = icmp eq i64 %memory_len997.peel, 0
  br i1 %i107, label %oob1016, label %load1017.peel

load1017.peel:                                    ; preds = %idxend1012.peel
  %memoryref_data1018.peel = getelementptr inbounds i8, ptr %memoryref_data994.peel, i64 %memoryref_byteoffset999.peel
  %i108 = load i32, ptr %memoryref_data1018.peel, align 4
  %i109 = icmp eq i32 %i108, 1
  %i110 = uitofp i1 %i109 to float
  %i111 = and i1 %i84, %i96
  br i1 %i111, label %L824.peel, label %L821

L824.peel:                                        ; preds = %load1017.peel
  %memory_len1031.peel = load i64, ptr %memoryref_mem1048, align 8
  %i112 = add i64 %memory_len1031.peel, %reass.mul
  %i113 = shl nuw nsw i64 %memory_len1031.peel, 1
  %memoryref_ovflw1032.not.peel = icmp ult i64 %i112, %i113
  %memoryref_byteoffset1033.peel = shl i64 %reass.mul, 2
  %memoryref_data_byteoffset1034.peel = getelementptr i8, ptr %memoryref_data1028, i64 %memoryref_byteoffset1033.peel
  %memory_data1037.peel = load ptr, ptr %memory_data_ptr1036, align 8
  %i114 = ptrtoint ptr %memoryref_data_byteoffset1034.peel to i64
  %i115 = ptrtoint ptr %memory_data1037.peel to i64
  %i116 = sub i64 %i114, %i115
  %memoryref_bytelen1038.peel = shl nuw nsw i64 %memory_len1031.peel, 2
  %memoryref_isinbounds1039.peel = icmp ult i64 %i116, %memoryref_bytelen1038.peel
  %"memoryref_isinbounds&notovflw1040.peel" = and i1 %memoryref_ovflw1032.not.peel, %memoryref_isinbounds1039.peel
  br i1 %"memoryref_isinbounds&notovflw1040.peel", label %idxend1046.peel, label %oob1041

idxend1046.peel:                                  ; preds = %L824.peel
  %i117 = icmp eq i64 %memory_len1031.peel, 0
  br i1 %i117, label %oob1050, label %load1051.peel

load1051.peel:                                    ; preds = %idxend1046.peel
  %memoryref_data1052.peel = getelementptr inbounds i8, ptr %memoryref_data1028, i64 %memoryref_byteoffset1033.peel
  store float %i110, ptr %memoryref_data1052.peel, align 4
  br i1 %.not1695.peel, label %L850, label %L624

L624:                                             ; preds = %load1051, %load1051.peel
  %i118 = phi i64 [ %i119, %load1051 ], [ 1, %load1051.peel ]
  %i119 = add nuw nsw i64 %i118, 1
  %i120 = select i1 %i27, i64 1, i64 %i119
  %i121 = add nsw i64 %i120, -1
  %.not1687 = icmp ult i64 %i121, %i10
  br i1 %.not1687, label %L673, label %L644

L644:                                             ; preds = %L624, %L620
  %.lcssa3140 = phi i64 [ %i120, %L624 ], [ 1, %L620 ]
  store i64 %.lcssa3140, ptr %"new::Tuple976", align 8
  store <2 x i64> <i64 1, i64 1>, ptr %i75, align 8
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_4015(ptr nocapture readonly %"new::ReshapedArray", ptr nocapture readonly %"new::Tuple976") #22
  unreachable

L673:                                             ; preds = %L624
  %.not1689 = icmp ugt i64 %i120, %i10
  br i1 %.not1689, label %L691, label %L684

L684:                                             ; preds = %L673
  %i122 = trunc i64 %i120 to i32
  %i123 = select i1 %i66, i64 1, i64 %i119
  %i124 = add nsw i64 %i123, -1
  %i125 = icmp ult i64 %i124, %.size93.sroa.0.0.copyload
  br i1 %i125, label %L740, label %L737

L691:                                             ; preds = %L673
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_4008(ptr nocapture readonly %"new::UnitRange", i64 signext %i119) #22
  unreachable

L737:                                             ; preds = %L684, %L622
  %.lcssa3239 = phi i64 [ %i119, %L684 ], [ 1, %L622 ]
  store i64 %.lcssa3239, ptr %"new::Tuple983", align 1
  store i64 %i93, ptr %i76, align 1
  store i64 %i86, ptr %i77, align 1
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_4006(ptr %i25, ptr nocapture readonly %"new::Tuple983") #22
  unreachable

L740:                                             ; preds = %L684
  %i126 = add i64 %i123, %i101
  %i127 = add i64 %i126, -1
  %.not1690 = icmp ult i64 %i127, %i83
  br i1 %.not1690, label %L773, label %L770

L770.loopexit4481:                                ; preds = %L740.peel
  %i128 = add i64 %i101, 1
  br label %L770

L770:                                             ; preds = %L770.loopexit4481, %L740
  %.lcssa3247 = phi i64 [ %i128, %L770.loopexit4481 ], [ %i126, %L740 ]
  store i64 %.lcssa3247, ptr %"new::Tuple1054", align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_4004(ptr %i25, ptr nocapture readonly %"new::Tuple1054") #22
  unreachable

L773:                                             ; preds = %L740
  %i129 = add i64 %i127, %memory_len997.peel
  %memoryref_ovflw998.not = icmp ult i64 %i129, %i103
  %memoryref_byteoffset999 = shl i64 %i127, 2
  %memoryref_data_byteoffset1000 = getelementptr i8, ptr %memoryref_data994.peel, i64 %memoryref_byteoffset999
  %i130 = ptrtoint ptr %memoryref_data_byteoffset1000 to i64
  %i131 = sub i64 %i130, %i105
  %memoryref_isinbounds1005 = icmp ult i64 %i131, %memoryref_bytelen1004.peel
  %"memoryref_isinbounds&notovflw1006" = and i1 %memoryref_ovflw998.not, %memoryref_isinbounds1005
  br i1 %"memoryref_isinbounds&notovflw1006", label %load1017, label %oob1007

L821:                                             ; preds = %load1017, %load1017.peel
  %.lcssa3158 = phi i64 [ %i119, %load1017 ], [ 1, %load1017.peel ]
  store i64 %.lcssa3158, ptr %"new::Tuple1019", align 1
  store i64 %i92, ptr %i79, align 1
  store i64 %i85, ptr %i80, align 1
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  store ptr %"new::Array91", ptr %gc_slot_addr_18736, align 8
  call fastcc void @julia_throw_boundserror_3998(ptr %"new::Array91", ptr nocapture readonly %"new::Tuple1019") #22
  unreachable

L824:                                             ; preds = %load1017
  %memoryref_offset1030 = add i64 %i118, %reass.mul
  %i132 = add i64 %memoryref_offset1030, %memory_len1031.peel
  %memoryref_ovflw1032.not = icmp ult i64 %i132, %i113
  %memoryref_byteoffset1033 = shl i64 %memoryref_offset1030, 2
  %memoryref_data_byteoffset1034 = getelementptr i8, ptr %memoryref_data1028, i64 %memoryref_byteoffset1033
  %i133 = ptrtoint ptr %memoryref_data_byteoffset1034 to i64
  %i134 = sub i64 %i133, %i115
  %memoryref_isinbounds1039 = icmp ult i64 %i134, %memoryref_bytelen1038.peel
  %"memoryref_isinbounds&notovflw1040" = and i1 %memoryref_ovflw1032.not, %memoryref_isinbounds1039
  br i1 %"memoryref_isinbounds&notovflw1040", label %load1051, label %oob1041

L850:                                             ; preds = %load1051, %load1051.peel
  %.lcssa3246 = phi i64 [ 1, %load1051.peel ], [ %i123, %load1051 ]
  %.lcssa3150 = phi i64 [ 1, %load1051.peel ], [ %i120, %load1051 ]
  %.not1696 = icmp eq i64 %i92, %.size93.sroa.2.0.copyload
  br i1 %.not1696, label %L855, label %L622

L855:                                             ; preds = %L850
  %.not1697 = icmp eq i64 %i85, %.size93.sroa.3.0.copyload
  br i1 %.not1697, label %L864.loopexit4485, label %L620

L864.loopexit:                                    ; preds = %load165
  store i64 %i10, ptr %"new::Tuple113", align 8
  br label %L864

L864.loopexit4485:                                ; preds = %L855
  store i64 %.lcssa3246, ptr %"new::Tuple983", align 1
  store i64 %i93, ptr %i76, align 1
  store i64 %i86, ptr %i77, align 1
  store i64 %.size93.sroa.0.0.copyload.mux, ptr %"new::Tuple1019", align 1
  store i64 %.size93.sroa.2.0.copyload, ptr %i79, align 1
  store i64 %.size93.sroa.3.0.copyload, ptr %i80, align 1
  store i64 %.lcssa3150, ptr %"new::Tuple976", align 1
  store <2 x i64> <i64 1, i64 1>, ptr %i75, align 1
  br label %L864

L864:                                             ; preds = %L864.loopexit4485, %L864.loopexit, %L607, %L381
  %"pred::Array.size175.sroa.0.0.copyload" = load i64, ptr %"pred::Array.size_ptr", align 8
  %"pred::Array.size175.sroa.2.0.copyload" = load i64, ptr %"pred::Array.size.sroa.4.0.pred::Array.size_ptr.sroa_idx", align 8
  %"pred::Array.size175.sroa.3.0.pred::Array.size_ptr.sroa_idx" = getelementptr inbounds i8, ptr %"pred::Array", i64 32
  %"pred::Array.size175.sroa.3.0.copyload" = load i64, ptr %"pred::Array.size175.sroa.3.0.pred::Array.size_ptr.sroa_idx", align 8
  %i135 = icmp eq i64 %"pred::Array.size175.sroa.0.0.copyload", %value_phi170.size232.sroa.0.0.copyload
  %i136 = icmp eq i64 %value_phi170.size232.sroa.0.0.copyload, 1
  %narrow1698 = or i1 %i136, %i135
  br i1 %narrow1698, label %L906, label %L893

L893:                                             ; preds = %L864
  %value_phi867.v.not = icmp eq i64 %"pred::Array.size175.sroa.0.0.copyload", 1
  br i1 %value_phi867.v.not, label %L906, label %L901

L901:                                             ; preds = %L893
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", i64 16) ]
  %ptls_load8870 = load ptr, ptr %ptls_field, align 8
  %"new::LazyString874" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8870, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" to i64)) #15
  %"new::LazyString874.tag_addr" = getelementptr inbounds ptr, ptr %"new::LazyString874", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", ptr %"new::LazyString874.tag_addr" unordered, align 8
  store <2 x ptr> zeroinitializer, ptr %"new::LazyString874", align 8
  store ptr %"new::LazyString874", ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", i64 16) ]
  %ptls_load8873 = load ptr, ptr %ptls_field, align 8
  %"box::Tuple878" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8873, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664" to i64)) #15
  %"box::Tuple878.tag_addr" = getelementptr inbounds ptr, ptr %"box::Tuple878", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", ptr %"box::Tuple878.tag_addr" unordered, align 8
  store atomic ptr @"ejl_inserted$jl_global_3975$false$4940591872", ptr %"box::Tuple878" unordered, align 8
  %i137 = getelementptr inbounds i8, ptr %"box::Tuple878", i64 8
  store i64 %value_phi170.size232.sroa.0.0.copyload, ptr %i137, align 8
  %i138 = getelementptr inbounds i8, ptr %"box::Tuple878", i64 16
  store atomic ptr @"ejl_inserted$jl_global_3976$false$4940591824", ptr %i138 unordered, align 8
  %i139 = getelementptr inbounds i8, ptr %"box::Tuple878", i64 24
  store i64 %"pred::Array.size175.sroa.0.0.copyload", ptr %i139, align 8
  store atomic ptr %"box::Tuple878", ptr %"new::LazyString874" release, align 8
  %"new::LazyString874.tag" = load atomic volatile i64, ptr %"new::LazyString874.tag_addr" unordered, align 8
  %parent_bits9090 = and i64 %"new::LazyString874.tag", 3
  %parent_old_marked9091 = icmp eq i64 %parent_bits9090, 3
  br i1 %parent_old_marked9091, label %may_trigger_wb9092, label %bb140

may_trigger_wb9092:                               ; preds = %L901
  %"box::Tuple878.tag" = load atomic volatile i64, ptr %"box::Tuple878.tag_addr" unordered, align 8
  %child_bit9094 = and i64 %"box::Tuple878.tag", 1
  %child_not_marked9095 = icmp eq i64 %child_bit9094, 0
  br i1 %child_not_marked9095, label %trigger_wb9096, label %bb140

trigger_wb9096:                                   ; preds = %may_trigger_wb9092
  call void @ijl_gc_queue_root(ptr nonnull %"new::LazyString874")
  br label %bb140

bb140:                                            ; preds = %trigger_wb9096, %may_trigger_wb9092, %L901
  %i141 = getelementptr inbounds i8, ptr %"new::LazyString874", i64 8
  %jl_nothing879 = load ptr, ptr @jl_nothing, align 8
  store atomic ptr %jl_nothing879, ptr %i141 release, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", i64 16) ]
  %"box::DimensionMismatch883" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8873, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" to i64)) #15
  %"box::DimensionMismatch883.tag_addr" = getelementptr inbounds ptr, ptr %"box::DimensionMismatch883", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", ptr %"box::DimensionMismatch883.tag_addr" unordered, align 8
  store atomic ptr %"new::LazyString874", ptr %"box::DimensionMismatch883" unordered, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::DimensionMismatch883")
  unreachable

L906:                                             ; preds = %L893, %L864
  %value_phi336.size397.sroa.0.0.copyload = phi i64 [ %"pred::Array.size175.sroa.0.0.copyload", %L864 ], [ %value_phi170.size232.sroa.0.0.copyload, %L893 ]
  %i142 = icmp eq i64 %"pred::Array.size175.sroa.2.0.copyload", %.size.sroa.2.0.copyload
  %i143 = icmp eq i64 %.size.sroa.2.0.copyload, 1
  %narrow1699 = or i1 %i143, %i142
  br i1 %narrow1699, label %L928, label %L915

L915:                                             ; preds = %L906
  %value_phi844.v.not = icmp eq i64 %"pred::Array.size175.sroa.2.0.copyload", 1
  br i1 %value_phi844.v.not, label %L928, label %L923

L923:                                             ; preds = %L915
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", i64 16) ]
  %ptls_load8880 = load ptr, ptr %ptls_field, align 8
  %"new::LazyString851" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8880, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" to i64)) #15
  %"new::LazyString851.tag_addr" = getelementptr inbounds ptr, ptr %"new::LazyString851", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", ptr %"new::LazyString851.tag_addr" unordered, align 8
  store <2 x ptr> zeroinitializer, ptr %"new::LazyString851", align 8
  store ptr %"new::LazyString851", ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", i64 16) ]
  %ptls_load8883 = load ptr, ptr %ptls_field, align 8
  %"box::Tuple855" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8883, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664" to i64)) #15
  %"box::Tuple855.tag_addr" = getelementptr inbounds ptr, ptr %"box::Tuple855", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", ptr %"box::Tuple855.tag_addr" unordered, align 8
  store atomic ptr @"ejl_inserted$jl_global_3975$false$4940591872", ptr %"box::Tuple855" unordered, align 8
  %i144 = getelementptr inbounds i8, ptr %"box::Tuple855", i64 8
  store i64 %.size.sroa.2.0.copyload, ptr %i144, align 8
  %i145 = getelementptr inbounds i8, ptr %"box::Tuple855", i64 16
  store atomic ptr @"ejl_inserted$jl_global_3976$false$4940591824", ptr %i145 unordered, align 8
  %i146 = getelementptr inbounds i8, ptr %"box::Tuple855", i64 24
  store i64 %"pred::Array.size175.sroa.2.0.copyload", ptr %i146, align 8
  store atomic ptr %"box::Tuple855", ptr %"new::LazyString851" release, align 8
  %"new::LazyString851.tag" = load atomic volatile i64, ptr %"new::LazyString851.tag_addr" unordered, align 8
  %parent_bits9098 = and i64 %"new::LazyString851.tag", 3
  %parent_old_marked9099 = icmp eq i64 %parent_bits9098, 3
  br i1 %parent_old_marked9099, label %may_trigger_wb9100, label %bb147

may_trigger_wb9100:                               ; preds = %L923
  %"box::Tuple855.tag" = load atomic volatile i64, ptr %"box::Tuple855.tag_addr" unordered, align 8
  %child_bit9102 = and i64 %"box::Tuple855.tag", 1
  %child_not_marked9103 = icmp eq i64 %child_bit9102, 0
  br i1 %child_not_marked9103, label %trigger_wb9104, label %bb147

trigger_wb9104:                                   ; preds = %may_trigger_wb9100
  call void @ijl_gc_queue_root(ptr nonnull %"new::LazyString851")
  br label %bb147

bb147:                                            ; preds = %trigger_wb9104, %may_trigger_wb9100, %L923
  %i148 = getelementptr inbounds i8, ptr %"new::LazyString851", i64 8
  %jl_nothing856 = load ptr, ptr @jl_nothing, align 8
  store atomic ptr %jl_nothing856, ptr %i148 release, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", i64 16) ]
  %"box::DimensionMismatch860" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8883, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" to i64)) #15
  %"box::DimensionMismatch860.tag_addr" = getelementptr inbounds ptr, ptr %"box::DimensionMismatch860", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", ptr %"box::DimensionMismatch860.tag_addr" unordered, align 8
  store atomic ptr %"new::LazyString851", ptr %"box::DimensionMismatch860" unordered, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::DimensionMismatch860")
  unreachable

L928:                                             ; preds = %L915, %L906
  %value_phi336.size397.sroa.2.0.copyload = phi i64 [ %"pred::Array.size175.sroa.2.0.copyload", %L906 ], [ %.size.sroa.2.0.copyload, %L915 ]
  %i149 = icmp eq i64 %"pred::Array.size175.sroa.3.0.copyload", %.size.sroa.3.0.copyload
  %i150 = icmp eq i64 %.size.sroa.3.0.copyload, 1
  %narrow1700 = or i1 %i150, %i149
  br i1 %narrow1700, label %L958, label %L937

L937:                                             ; preds = %L928
  %value_phi821.v.not = icmp eq i64 %"pred::Array.size175.sroa.3.0.copyload", 1
  br i1 %value_phi821.v.not, label %L958, label %L945

L945:                                             ; preds = %L937
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", i64 16) ]
  %ptls_load8890 = load ptr, ptr %ptls_field, align 8
  %"new::LazyString828" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8890, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" to i64)) #15
  %"new::LazyString828.tag_addr" = getelementptr inbounds ptr, ptr %"new::LazyString828", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", ptr %"new::LazyString828.tag_addr" unordered, align 8
  store <2 x ptr> zeroinitializer, ptr %"new::LazyString828", align 8
  store ptr %"new::LazyString828", ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", i64 16) ]
  %ptls_load8893 = load ptr, ptr %ptls_field, align 8
  %"box::Tuple832" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8893, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664" to i64)) #15
  %"box::Tuple832.tag_addr" = getelementptr inbounds ptr, ptr %"box::Tuple832", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", ptr %"box::Tuple832.tag_addr" unordered, align 8
  store atomic ptr @"ejl_inserted$jl_global_3975$false$4940591872", ptr %"box::Tuple832" unordered, align 8
  %i151 = getelementptr inbounds i8, ptr %"box::Tuple832", i64 8
  store i64 %.size.sroa.3.0.copyload, ptr %i151, align 8
  %i152 = getelementptr inbounds i8, ptr %"box::Tuple832", i64 16
  store atomic ptr @"ejl_inserted$jl_global_3976$false$4940591824", ptr %i152 unordered, align 8
  %i153 = getelementptr inbounds i8, ptr %"box::Tuple832", i64 24
  store i64 %"pred::Array.size175.sroa.3.0.copyload", ptr %i153, align 8
  store atomic ptr %"box::Tuple832", ptr %"new::LazyString828" release, align 8
  %"new::LazyString828.tag" = load atomic volatile i64, ptr %"new::LazyString828.tag_addr" unordered, align 8
  %parent_bits9106 = and i64 %"new::LazyString828.tag", 3
  %parent_old_marked9107 = icmp eq i64 %parent_bits9106, 3
  br i1 %parent_old_marked9107, label %may_trigger_wb9108, label %bb154

may_trigger_wb9108:                               ; preds = %L945
  %"box::Tuple832.tag" = load atomic volatile i64, ptr %"box::Tuple832.tag_addr" unordered, align 8
  %child_bit9110 = and i64 %"box::Tuple832.tag", 1
  %child_not_marked9111 = icmp eq i64 %child_bit9110, 0
  br i1 %child_not_marked9111, label %trigger_wb9112, label %bb154

trigger_wb9112:                                   ; preds = %may_trigger_wb9108
  call void @ijl_gc_queue_root(ptr nonnull %"new::LazyString828")
  br label %bb154

bb154:                                            ; preds = %trigger_wb9112, %may_trigger_wb9108, %L945
  %i155 = getelementptr inbounds i8, ptr %"new::LazyString828", i64 8
  %jl_nothing833 = load ptr, ptr @jl_nothing, align 8
  store atomic ptr %jl_nothing833, ptr %i155 release, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", i64 16) ]
  %"box::DimensionMismatch837" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8893, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" to i64)) #15
  %"box::DimensionMismatch837.tag_addr" = getelementptr inbounds ptr, ptr %"box::DimensionMismatch837", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", ptr %"box::DimensionMismatch837.tag_addr" unordered, align 8
  store atomic ptr %"new::LazyString828", ptr %"box::DimensionMismatch837" unordered, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::DimensionMismatch837")
  unreachable

L958:                                             ; preds = %L937, %L928
  %value_phi336.size397.sroa.3.0.copyload = phi i64 [ %"pred::Array.size175.sroa.3.0.copyload", %L928 ], [ %.size.sroa.3.0.copyload, %L937 ]
  %i156 = icmp ugt i64 %value_phi336.size397.sroa.0.0.copyload, 9223372036854775806
  %i157 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %value_phi336.size397.sroa.0.0.copyload, i64 %value_phi336.size397.sroa.2.0.copyload)
  %i158 = extractvalue { i64, i1 } %i157, 0
  %i159 = extractvalue { i64, i1 } %i157, 1
  %i160 = icmp ne i64 %value_phi336.size397.sroa.2.0.copyload, 0
  %i161 = icmp ugt i64 %value_phi336.size397.sroa.2.0.copyload, 9223372036854775806
  %i162 = or i1 %i156, %i161
  %i163 = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %i158, i64 %value_phi336.size397.sroa.3.0.copyload)
  %i164 = extractvalue { i64, i1 } %i163, 0
  %i165 = extractvalue { i64, i1 } %i163, 1
  %i166 = icmp ne i64 %value_phi336.size397.sroa.3.0.copyload, 0
  %.not3898 = and i1 %i160, %i166
  %i167 = or i1 %i159, %i165
  %i168 = icmp ugt i64 %value_phi336.size397.sroa.3.0.copyload, 9223372036854775806
  %i169 = or i1 %i162, %i168
  %i170 = and i1 %.not3898, %i167
  %i171 = or i1 %i169, %i170
  br i1 %i171, label %L983, label %L987

L983:                                             ; preds = %L958
  %i172 = call fastcc [1 x ptr] @julia_ArgumentError_4002()
  store ptr null, ptr %gc_slot_addr_18736, align 8
  %i173 = extractvalue [1 x ptr] %i172, 0
  store ptr %i173, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_ArgumentError_3969$false$4895492080", i64 16) ]
  %ptls_load8901 = load ptr, ptr %ptls_field, align 8
  %"box::ArgumentError203" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8901, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Core_ArgumentError_3969$false$4895492080" to i64)) #15
  %"box::ArgumentError203.tag_addr" = getelementptr inbounds ptr, ptr %"box::ArgumentError203", i64 -1
  store atomic ptr @"ejl_inserted$_Core_ArgumentError_3969$false$4895492080", ptr %"box::ArgumentError203.tag_addr" unordered, align 8
  store ptr %i173, ptr %"box::ArgumentError203", align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::ArgumentError203")
  unreachable

L987:                                             ; preds = %L958
  %memorynew_empty206 = icmp eq i64 %i164, 0
  br i1 %memorynew_empty206, label %L987.retval217_crit_edge, label %nonemptymem209

L987.retval217_crit_edge:                         ; preds = %L987
  br label %retval217

L1122:                                            ; preds = %retval217
  %i174 = mul i64 %.size.sroa.3.0.copyload, %.size.sroa.2.0.copyload
  %i175 = mul i64 %i174, %value_phi170.size232.sroa.0.0.copyload
  %i176 = icmp slt i64 %i175, 1
  br i1 %i176, label %L1776, label %L1141.preheader

L1141.preheader:                                  ; preds = %L1122
  %memoryref_data255 = load ptr, ptr %"new::Array91", align 8
  %memoryref_mem275 = load ptr, ptr %i322, align 8
  %memory_data_ptr263 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem275, i64 0, i32 1
  %memoryref_data282 = load ptr, ptr %"pred::Array", align 8
  %i177 = getelementptr inbounds { ptr, ptr }, ptr %"pred::Array", i64 0, i32 1
  %memoryref_mem302 = load ptr, ptr %i177, align 8
  %memory_data_ptr290 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem302, i64 0, i32 1
  %i178 = mul i64 %value_phi336.size397.sroa.3.0.copyload, %i158
  %i179 = add i64 %i178, 1
  %memory_len258 = load i64, ptr %memoryref_mem275, align 8
  %i180 = shl nuw nsw i64 %memory_len258, 1
  %memory_data264 = load ptr, ptr %memory_data_ptr263, align 8
  %i181 = ptrtoint ptr %memory_data264 to i64
  %memoryref_bytelen265 = shl nuw nsw i64 %memory_len258, 2
  %i182 = icmp eq i64 %memory_len258, 0
  br label %L1160

L1160:                                            ; preds = %load331, %L1141.preheader
  %value_phi251 = phi i64 [ %i336, %load331 ], [ 1, %L1141.preheader ]
  %i183 = add nsw i64 %value_phi251, -1
  %i184 = add nuw nsw i64 %i183, %memory_len258
  %memoryref_ovflw259.not = icmp ult i64 %i184, %i180
  %memoryref_byteoffset260 = shl i64 %i183, 2
  %memoryref_data_byteoffset261 = getelementptr i8, ptr %memoryref_data255, i64 %memoryref_byteoffset260
  %i185 = ptrtoint ptr %memoryref_data_byteoffset261 to i64
  %i186 = sub i64 %i185, %i181
  %memoryref_isinbounds266 = icmp ult i64 %i186, %memoryref_bytelen265
  %"memoryref_isinbounds&notovflw267" = and i1 %memoryref_ovflw259.not, %memoryref_isinbounds266
  br i1 %"memoryref_isinbounds&notovflw267", label %idxend273, label %oob268

L1182:                                            ; preds = %idxend273
  %memoryref_data279 = getelementptr inbounds i8, ptr %memoryref_data255, i64 %memoryref_byteoffset260
  %i187 = load float, ptr %memoryref_data279, align 4
  %memory_len285 = load i64, ptr %memoryref_mem302, align 8
  %i188 = add i64 %memory_len285, %i183
  %i189 = shl nuw nsw i64 %memory_len285, 1
  %memoryref_ovflw286.not = icmp ult i64 %i188, %i189
  %memoryref_data_byteoffset288 = getelementptr i8, ptr %memoryref_data282, i64 %memoryref_byteoffset260
  %memory_data291 = load ptr, ptr %memory_data_ptr290, align 8
  %i190 = ptrtoint ptr %memoryref_data_byteoffset288 to i64
  %i191 = ptrtoint ptr %memory_data291 to i64
  %i192 = sub i64 %i190, %i191
  %memoryref_bytelen292 = shl nuw nsw i64 %memory_len285, 2
  %memoryref_isinbounds293 = icmp ult i64 %i192, %memoryref_bytelen292
  %"memoryref_isinbounds&notovflw294" = and i1 %memoryref_ovflw286.not, %memoryref_isinbounds293
  br i1 %"memoryref_isinbounds&notovflw294", label %idxend300, label %oob295

L1203:                                            ; preds = %load305
  store i64 %i179, ptr %"new::Tuple428", align 8
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %"new::Array225", ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_3993(ptr %"new::Array225", ptr nocapture readonly %"new::Tuple428") #22
  unreachable

L1206:                                            ; preds = %load305
  %memory_len311 = load i64, ptr %memoryref_mem328, align 8
  %i193 = add i64 %memory_len311, %i183
  %i194 = shl nuw nsw i64 %memory_len311, 1
  %memoryref_ovflw312.not = icmp ult i64 %i193, %i194
  %memoryref_bytelen318 = shl nuw nsw i64 %memory_len311, 2
  %memoryref_isinbounds319 = icmp ult i64 %memoryref_byteoffset260, %memoryref_bytelen318
  %"memoryref_isinbounds&notovflw320" = and i1 %memoryref_ovflw312.not, %memoryref_isinbounds319
  br i1 %"memoryref_isinbounds&notovflw320", label %idxend326, label %oob321

L1224:                                            ; preds = %retval217
  %.not4467 = icmp eq i64 %"pred::Array.size175.sroa.3.0.copyload", 1
  %.not4468 = icmp eq i64 %"pred::Array.size175.sroa.2.0.copyload", 1
  %.not4469 = icmp eq i64 %"pred::Array.size175.sroa.0.0.copyload", 1
  %brmerge6337 = or i1 %.not4469, %narrow1698
  %"pred::Array.size175.sroa.0.0.copyload.mux" = select i1 %i136, i64 %"pred::Array.size175.sroa.0.0.copyload", i64 %value_phi170.size232.sroa.0.0.copyload
  br i1 %brmerge6337, label %L1291, label %L1286

L1286:                                            ; preds = %L1224
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", i64 16) ]
  %ptls_load8909 = load ptr, ptr %ptls_field, align 8
  %"new::LazyString803" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8909, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" to i64)) #15
  %"new::LazyString803.tag_addr" = getelementptr inbounds ptr, ptr %"new::LazyString803", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", ptr %"new::LazyString803.tag_addr" unordered, align 8
  store <2 x ptr> zeroinitializer, ptr %"new::LazyString803", align 8
  store ptr %"new::LazyString803", ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", i64 16) ]
  %ptls_load8912 = load ptr, ptr %ptls_field, align 8
  %"box::Tuple807" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8912, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664" to i64)) #15
  %"box::Tuple807.tag_addr" = getelementptr inbounds ptr, ptr %"box::Tuple807", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", ptr %"box::Tuple807.tag_addr" unordered, align 8
  store atomic ptr @"ejl_inserted$jl_global_3975$false$4940591872", ptr %"box::Tuple807" unordered, align 8
  %i195 = getelementptr inbounds i8, ptr %"box::Tuple807", i64 8
  store i64 %value_phi170.size232.sroa.0.0.copyload, ptr %i195, align 8
  %i196 = getelementptr inbounds i8, ptr %"box::Tuple807", i64 16
  store atomic ptr @"ejl_inserted$jl_global_3976$false$4940591824", ptr %i196 unordered, align 8
  %i197 = getelementptr inbounds i8, ptr %"box::Tuple807", i64 24
  store i64 %"pred::Array.size175.sroa.0.0.copyload", ptr %i197, align 8
  store atomic ptr %"box::Tuple807", ptr %"new::LazyString803" release, align 8
  %"new::LazyString803.tag" = load atomic volatile i64, ptr %"new::LazyString803.tag_addr" unordered, align 8
  %parent_bits9114 = and i64 %"new::LazyString803.tag", 3
  %parent_old_marked9115 = icmp eq i64 %parent_bits9114, 3
  br i1 %parent_old_marked9115, label %may_trigger_wb9116, label %bb198

may_trigger_wb9116:                               ; preds = %L1286
  %"box::Tuple807.tag" = load atomic volatile i64, ptr %"box::Tuple807.tag_addr" unordered, align 8
  %child_bit9118 = and i64 %"box::Tuple807.tag", 1
  %child_not_marked9119 = icmp eq i64 %child_bit9118, 0
  br i1 %child_not_marked9119, label %trigger_wb9120, label %bb198

trigger_wb9120:                                   ; preds = %may_trigger_wb9116
  call void @ijl_gc_queue_root(ptr nonnull %"new::LazyString803")
  br label %bb198

bb198:                                            ; preds = %trigger_wb9120, %may_trigger_wb9116, %L1286
  %i199 = getelementptr inbounds i8, ptr %"new::LazyString803", i64 8
  %jl_nothing808 = load ptr, ptr @jl_nothing, align 8
  store atomic ptr %jl_nothing808, ptr %i199 release, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", i64 16) ]
  %"box::DimensionMismatch812" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8912, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" to i64)) #15
  %"box::DimensionMismatch812.tag_addr" = getelementptr inbounds ptr, ptr %"box::DimensionMismatch812", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", ptr %"box::DimensionMismatch812.tag_addr" unordered, align 8
  store atomic ptr %"new::LazyString803", ptr %"box::DimensionMismatch812" unordered, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::DimensionMismatch812")
  unreachable

L1291:                                            ; preds = %L1224
  %brmerge6338 = or i1 %.not4468, %narrow1699
  %"pred::Array.size175.sroa.2.0.copyload.mux" = select i1 %i143, i64 %"pred::Array.size175.sroa.2.0.copyload", i64 %.size.sroa.2.0.copyload
  br i1 %brmerge6338, label %L1316, label %L1311

L1311:                                            ; preds = %L1291
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", i64 16) ]
  %ptls_load8919 = load ptr, ptr %ptls_field, align 8
  %"new::LazyString780" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8919, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" to i64)) #15
  %"new::LazyString780.tag_addr" = getelementptr inbounds ptr, ptr %"new::LazyString780", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", ptr %"new::LazyString780.tag_addr" unordered, align 8
  store <2 x ptr> zeroinitializer, ptr %"new::LazyString780", align 8
  store ptr %"new::LazyString780", ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", i64 16) ]
  %ptls_load8922 = load ptr, ptr %ptls_field, align 8
  %"box::Tuple784" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8922, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664" to i64)) #15
  %"box::Tuple784.tag_addr" = getelementptr inbounds ptr, ptr %"box::Tuple784", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", ptr %"box::Tuple784.tag_addr" unordered, align 8
  store atomic ptr @"ejl_inserted$jl_global_3975$false$4940591872", ptr %"box::Tuple784" unordered, align 8
  %i200 = getelementptr inbounds i8, ptr %"box::Tuple784", i64 8
  store i64 %.size.sroa.2.0.copyload, ptr %i200, align 8
  %i201 = getelementptr inbounds i8, ptr %"box::Tuple784", i64 16
  store atomic ptr @"ejl_inserted$jl_global_3976$false$4940591824", ptr %i201 unordered, align 8
  %i202 = getelementptr inbounds i8, ptr %"box::Tuple784", i64 24
  store i64 %"pred::Array.size175.sroa.2.0.copyload", ptr %i202, align 8
  store atomic ptr %"box::Tuple784", ptr %"new::LazyString780" release, align 8
  %"new::LazyString780.tag" = load atomic volatile i64, ptr %"new::LazyString780.tag_addr" unordered, align 8
  %parent_bits9122 = and i64 %"new::LazyString780.tag", 3
  %parent_old_marked9123 = icmp eq i64 %parent_bits9122, 3
  br i1 %parent_old_marked9123, label %may_trigger_wb9124, label %bb203

may_trigger_wb9124:                               ; preds = %L1311
  %"box::Tuple784.tag" = load atomic volatile i64, ptr %"box::Tuple784.tag_addr" unordered, align 8
  %child_bit9126 = and i64 %"box::Tuple784.tag", 1
  %child_not_marked9127 = icmp eq i64 %child_bit9126, 0
  br i1 %child_not_marked9127, label %trigger_wb9128, label %bb203

trigger_wb9128:                                   ; preds = %may_trigger_wb9124
  call void @ijl_gc_queue_root(ptr nonnull %"new::LazyString780")
  br label %bb203

bb203:                                            ; preds = %trigger_wb9128, %may_trigger_wb9124, %L1311
  %i204 = getelementptr inbounds i8, ptr %"new::LazyString780", i64 8
  %jl_nothing785 = load ptr, ptr @jl_nothing, align 8
  store atomic ptr %jl_nothing785, ptr %i204 release, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", i64 16) ]
  %"box::DimensionMismatch789" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8922, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" to i64)) #15
  %"box::DimensionMismatch789.tag_addr" = getelementptr inbounds ptr, ptr %"box::DimensionMismatch789", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", ptr %"box::DimensionMismatch789.tag_addr" unordered, align 8
  store atomic ptr %"new::LazyString780", ptr %"box::DimensionMismatch789" unordered, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::DimensionMismatch789")
  unreachable

L1316:                                            ; preds = %L1291
  %brmerge6339 = or i1 %.not4467, %narrow1700
  %"pred::Array.size175.sroa.3.0.copyload.mux" = select i1 %i150, i64 %"pred::Array.size175.sroa.3.0.copyload", i64 %.size.sroa.3.0.copyload
  br i1 %brmerge6339, label %L1516, label %L1336

L1336:                                            ; preds = %L1316
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", i64 16) ]
  %ptls_load8929 = load ptr, ptr %ptls_field, align 8
  %"new::LazyString757" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8929, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040" to i64)) #15
  %"new::LazyString757.tag_addr" = getelementptr inbounds ptr, ptr %"new::LazyString757", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_LazyString_3977$false$4810811040", ptr %"new::LazyString757.tag_addr" unordered, align 8
  store <2 x ptr> zeroinitializer, ptr %"new::LazyString757", align 8
  store ptr %"new::LazyString757", ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", i64 16) ]
  %ptls_load8932 = load ptr, ptr %ptls_field, align 8
  %"box::Tuple761" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8932, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664" to i64)) #15
  %"box::Tuple761.tag_addr" = getelementptr inbounds ptr, ptr %"box::Tuple761", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Tuple_3978$false$4846611664", ptr %"box::Tuple761.tag_addr" unordered, align 8
  store atomic ptr @"ejl_inserted$jl_global_3975$false$4940591872", ptr %"box::Tuple761" unordered, align 8
  %i205 = getelementptr inbounds i8, ptr %"box::Tuple761", i64 8
  store i64 %.size.sroa.3.0.copyload, ptr %i205, align 8
  %i206 = getelementptr inbounds i8, ptr %"box::Tuple761", i64 16
  store atomic ptr @"ejl_inserted$jl_global_3976$false$4940591824", ptr %i206 unordered, align 8
  %i207 = getelementptr inbounds i8, ptr %"box::Tuple761", i64 24
  store i64 %"pred::Array.size175.sroa.3.0.copyload", ptr %i207, align 8
  store atomic ptr %"box::Tuple761", ptr %"new::LazyString757" release, align 8
  %"new::LazyString757.tag" = load atomic volatile i64, ptr %"new::LazyString757.tag_addr" unordered, align 8
  %parent_bits9130 = and i64 %"new::LazyString757.tag", 3
  %parent_old_marked9131 = icmp eq i64 %parent_bits9130, 3
  br i1 %parent_old_marked9131, label %may_trigger_wb9132, label %bb208

may_trigger_wb9132:                               ; preds = %L1336
  %"box::Tuple761.tag" = load atomic volatile i64, ptr %"box::Tuple761.tag_addr" unordered, align 8
  %child_bit9134 = and i64 %"box::Tuple761.tag", 1
  %child_not_marked9135 = icmp eq i64 %child_bit9134, 0
  br i1 %child_not_marked9135, label %trigger_wb9136, label %bb208

trigger_wb9136:                                   ; preds = %may_trigger_wb9132
  call void @ijl_gc_queue_root(ptr nonnull %"new::LazyString757")
  br label %bb208

bb208:                                            ; preds = %trigger_wb9136, %may_trigger_wb9132, %L1336
  %i209 = getelementptr inbounds i8, ptr %"new::LazyString757", i64 8
  %jl_nothing762 = load ptr, ptr @jl_nothing, align 8
  store atomic ptr %jl_nothing762, ptr %i209 release, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", i64 16) ]
  %"box::DimensionMismatch766" = call noalias nonnull align 8 dereferenceable(16) ptr @ijl_gc_small_alloc(ptr %ptls_load8932, i32 424, i32 16, i64 ptrtoint (ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688" to i64)) #15
  %"box::DimensionMismatch766.tag_addr" = getelementptr inbounds ptr, ptr %"box::DimensionMismatch766", i64 -1
  store atomic ptr @"ejl_inserted$_Main_Base_DimensionMismatch_3979$false$4810810688", ptr %"box::DimensionMismatch766.tag_addr" unordered, align 8
  store atomic ptr %"new::LazyString757", ptr %"box::DimensionMismatch766" unordered, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_throw(ptr %"box::DimensionMismatch766")
  unreachable

L1516:                                            ; preds = %L1316
  %i210 = icmp eq i64 %"pred::Array.size175.sroa.0.0.copyload.mux", 0
  %i211 = icmp eq i64 %"pred::Array.size175.sroa.2.0.copyload.mux", 0
  %narrow1741.not = select i1 %i210, i1 true, i1 %i211
  %i212 = icmp eq i64 %"pred::Array.size175.sroa.3.0.copyload.mux", 0
  %or.cond1773 = select i1 %narrow1741.not, i1 true, i1 %i212
  br i1 %or.cond1773, label %L1776, label %L1529.preheader

L1529.preheader:                                  ; preds = %L1516
  %i213 = mul i64 %value_phi170.size232.sroa.0.0.copyload, %.size.sroa.2.0.copyload
  %i214 = mul i64 %i213, %.size.sroa.3.0.copyload
  %i215 = mul i64 %"pred::Array.size175.sroa.2.0.copyload", %"pred::Array.size175.sroa.0.0.copyload"
  %i216 = mul i64 %i215, %"pred::Array.size175.sroa.3.0.copyload"
  %i217 = getelementptr inbounds { ptr, ptr }, ptr %"pred::Array", i64 0, i32 1
  %i218 = icmp ne i64 %value_phi170.size232.sroa.0.0.copyload, 0
  %i219 = icmp ne i64 %"pred::Array.size175.sroa.0.0.copyload", 0
  %i220 = icmp ne i64 %value_phi336.size397.sroa.0.0.copyload, 0
  %.not1752.peel = icmp eq i64 %"pred::Array.size175.sroa.0.0.copyload.mux", 1
  br label %L1529

L1529:                                            ; preds = %L1767, %L1529.preheader
  %value_phi622 = phi i64 [ %i221, %L1767 ], [ 0, %L1529.preheader ]
  %i221 = add nuw i64 %value_phi622, 1
  %i222 = select i1 %i150, i64 1, i64 %i221
  %i223 = add i64 %i222, -1
  %i224 = icmp ult i64 %i223, %.size.sroa.3.0.copyload
  %i225 = mul i64 %i223, %i213
  %i226 = select i1 %.not4467, i64 1, i64 %i221
  %i227 = add i64 %i226, -1
  %i228 = icmp ult i64 %i227, %"pred::Array.size175.sroa.3.0.copyload"
  %i229 = mul i64 %i227, %i215
  %i230 = icmp ult i64 %value_phi622, %value_phi336.size397.sroa.3.0.copyload
  %i231 = mul i64 %value_phi622, %value_phi336.size397.sroa.2.0.copyload
  br label %L1531

L1531:                                            ; preds = %L1762, %L1529
  %value_phi623 = phi i64 [ 0, %L1529 ], [ %i232, %L1762 ]
  %i232 = add nuw i64 %value_phi623, 1
  %i233 = select i1 %i143, i64 1, i64 %i232
  %i234 = add i64 %i233, -1
  %i235 = icmp ult i64 %i234, %.size.sroa.2.0.copyload
  %i236 = and i1 %i224, %i235
  %i237 = mul i64 %i234, %value_phi170.size232.sroa.0.0.copyload
  %i238 = add i64 %i237, %i225
  %memoryref_data639 = load ptr, ptr %"new::Array91", align 8
  %memoryref_mem659 = load ptr, ptr %i322, align 8
  %memory_data_ptr647 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem659, i64 0, i32 1
  %i239 = select i1 %.not4468, i64 1, i64 %i232
  %i240 = add i64 %i239, -1
  %i241 = icmp ult i64 %i240, %"pred::Array.size175.sroa.2.0.copyload"
  %i242 = and i1 %i228, %i241
  %i243 = mul i64 %i240, %"pred::Array.size175.sroa.0.0.copyload"
  %i244 = add i64 %i243, %i229
  %memoryref_data678 = load ptr, ptr %"pred::Array", align 8
  %memoryref_mem698 = load ptr, ptr %i217, align 8
  %memory_data_ptr686 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem698, i64 0, i32 1
  %i245 = icmp ult i64 %value_phi623, %value_phi336.size397.sroa.2.0.copyload
  %i246 = and i1 %i230, %i245
  %memoryref_data712 = load ptr, ptr %"new::Array225", align 8
  %memoryref_mem732 = load ptr, ptr %i331, align 8
  %reass.add1780 = add i64 %value_phi623, %i231
  %reass.mul1781 = mul i64 %reass.add1780, %value_phi336.size397.sroa.0.0.copyload
  %memory_data_ptr720 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem732, i64 0, i32 1
  %i247 = and i1 %i218, %i236
  br i1 %i247, label %L1575.peel, label %L1572

L1575.peel:                                       ; preds = %L1531
  %.not1744.peel = icmp ult i64 %i238, %i214
  br i1 %.not1744.peel, label %L1608.peel, label %L1605.loopexit4471

L1608.peel:                                       ; preds = %L1575.peel
  %memory_len642.peel = load i64, ptr %memoryref_mem659, align 8
  %i248 = add i64 %memory_len642.peel, %i238
  %i249 = shl nuw nsw i64 %memory_len642.peel, 1
  %memoryref_ovflw643.not.peel = icmp ult i64 %i248, %i249
  %memoryref_byteoffset644.peel = shl i64 %i238, 2
  %memoryref_data_byteoffset645.peel = getelementptr i8, ptr %memoryref_data639, i64 %memoryref_byteoffset644.peel
  %memory_data648.peel = load ptr, ptr %memory_data_ptr647, align 8
  %i250 = ptrtoint ptr %memoryref_data_byteoffset645.peel to i64
  %i251 = ptrtoint ptr %memory_data648.peel to i64
  %i252 = sub i64 %i250, %i251
  %memoryref_bytelen649.peel = shl nuw nsw i64 %memory_len642.peel, 2
  %memoryref_isinbounds650.peel = icmp ult i64 %i252, %memoryref_bytelen649.peel
  %"memoryref_isinbounds&notovflw651.peel" = and i1 %memoryref_ovflw643.not.peel, %memoryref_isinbounds650.peel
  br i1 %"memoryref_isinbounds&notovflw651.peel", label %idxend657.peel, label %oob652.loopexit4472

idxend657.peel:                                   ; preds = %L1608.peel
  %i253 = icmp eq i64 %memory_len642.peel, 0
  br i1 %i253, label %oob661, label %load662.peel

load662.peel:                                     ; preds = %idxend657.peel
  %memoryref_data663.peel = getelementptr inbounds i8, ptr %memoryref_data639, i64 %memoryref_byteoffset644.peel
  %i254 = load float, ptr %memoryref_data663.peel, align 4
  %i255 = and i1 %i219, %i242
  br i1 %i255, label %L1655.peel, label %L1652

L1655.peel:                                       ; preds = %load662.peel
  %.not1747.peel = icmp ult i64 %i244, %i216
  br i1 %.not1747.peel, label %L1688.peel, label %L1685.loopexit4474

L1688.peel:                                       ; preds = %L1655.peel
  %memory_len681.peel = load i64, ptr %memoryref_mem698, align 8
  %i256 = add i64 %memory_len681.peel, %i244
  %i257 = shl nuw nsw i64 %memory_len681.peel, 1
  %memoryref_ovflw682.not.peel = icmp ult i64 %i256, %i257
  %memoryref_byteoffset683.peel = shl i64 %i244, 2
  %memoryref_data_byteoffset684.peel = getelementptr i8, ptr %memoryref_data678, i64 %memoryref_byteoffset683.peel
  %memory_data687.peel = load ptr, ptr %memory_data_ptr686, align 8
  %i258 = ptrtoint ptr %memoryref_data_byteoffset684.peel to i64
  %i259 = ptrtoint ptr %memory_data687.peel to i64
  %i260 = sub i64 %i258, %i259
  %memoryref_bytelen688.peel = shl nuw nsw i64 %memory_len681.peel, 2
  %memoryref_isinbounds689.peel = icmp ult i64 %i260, %memoryref_bytelen688.peel
  %"memoryref_isinbounds&notovflw690.peel" = and i1 %memoryref_ovflw682.not.peel, %memoryref_isinbounds689.peel
  br i1 %"memoryref_isinbounds&notovflw690.peel", label %idxend696.peel, label %oob691.loopexit4475

idxend696.peel:                                   ; preds = %L1688.peel
  %i261 = icmp eq i64 %memory_len681.peel, 0
  br i1 %i261, label %oob700, label %load701.peel

load701.peel:                                     ; preds = %idxend696.peel
  %memoryref_data702.peel = getelementptr inbounds i8, ptr %memoryref_data678, i64 %memoryref_byteoffset683.peel
  %i262 = load float, ptr %memoryref_data702.peel, align 4
  %i263 = fmul float %i254, %i262
  %i264 = and i1 %i220, %i246
  br i1 %i264, label %L1736.peel, label %L1733

L1736.peel:                                       ; preds = %load701.peel
  %memory_len715.peel = load i64, ptr %memoryref_mem732, align 8
  %i265 = add i64 %memory_len715.peel, %reass.mul1781
  %i266 = shl nuw nsw i64 %memory_len715.peel, 1
  %memoryref_ovflw716.not.peel = icmp ult i64 %i265, %i266
  %memoryref_byteoffset717.peel = shl i64 %reass.mul1781, 2
  %memoryref_data_byteoffset718.peel = getelementptr i8, ptr %memoryref_data712, i64 %memoryref_byteoffset717.peel
  %memory_data721.peel = load ptr, ptr %memory_data_ptr720, align 8
  %i267 = ptrtoint ptr %memoryref_data_byteoffset718.peel to i64
  %i268 = ptrtoint ptr %memory_data721.peel to i64
  %i269 = sub i64 %i267, %i268
  %memoryref_bytelen722.peel = shl nuw nsw i64 %memory_len715.peel, 2
  %memoryref_isinbounds723.peel = icmp ult i64 %i269, %memoryref_bytelen722.peel
  %"memoryref_isinbounds&notovflw724.peel" = and i1 %memoryref_ovflw716.not.peel, %memoryref_isinbounds723.peel
  br i1 %"memoryref_isinbounds&notovflw724.peel", label %idxend730.peel, label %oob725

idxend730.peel:                                   ; preds = %L1736.peel
  %i270 = icmp eq i64 %memory_len715.peel, 0
  br i1 %i270, label %oob734, label %load735.peel

load735.peel:                                     ; preds = %idxend730.peel
  %memoryref_data736.peel = getelementptr inbounds i8, ptr %memoryref_data712, i64 %memoryref_byteoffset717.peel
  store float %i263, ptr %memoryref_data736.peel, align 4
  br i1 %.not1752.peel, label %L1762, label %L1533

L1533:                                            ; preds = %load735, %load735.peel
  %i271 = phi i64 [ %i272, %load735 ], [ 1, %load735.peel ]
  %i272 = add nuw nsw i64 %i271, 1
  %i273 = select i1 %i136, i64 1, i64 %i272
  %i274 = add nsw i64 %i273, -1
  %i275 = icmp ult i64 %i274, %value_phi170.size232.sroa.0.0.copyload
  br i1 %i275, label %L1575, label %L1572

L1572:                                            ; preds = %L1533, %L1531
  %.lcssa2742 = phi i64 [ %i273, %L1533 ], [ 1, %L1531 ]
  %i276 = getelementptr inbounds i8, ptr %"new::Tuple628", i64 16
  %i277 = getelementptr inbounds i8, ptr %"new::Tuple628", i64 8
  store i64 %.lcssa2742, ptr %"new::Tuple628", align 8
  store i64 %i233, ptr %i277, align 8
  store i64 %i222, ptr %i276, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_3998(ptr %"new::Array91", ptr nocapture readonly %"new::Tuple628") #22
  unreachable

L1575:                                            ; preds = %L1533
  %i278 = add i64 %i273, %i238
  %i279 = add i64 %i278, -1
  %.not1744 = icmp ult i64 %i279, %i214
  br i1 %.not1744, label %L1608, label %L1605

L1605.loopexit4471:                               ; preds = %L1575.peel
  %i280 = add i64 %i238, 1
  br label %L1605

L1605:                                            ; preds = %L1605.loopexit4471, %L1575
  %.lcssa2838 = phi i64 [ %i280, %L1605.loopexit4471 ], [ %i278, %L1575 ]
  store i64 %.lcssa2838, ptr %"new::Tuple741", align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_3993(ptr %"new::Array91", ptr nocapture readonly %"new::Tuple741") #22
  unreachable

L1608:                                            ; preds = %L1575
  %i281 = add i64 %i279, %memory_len642.peel
  %memoryref_ovflw643.not = icmp ult i64 %i281, %i249
  %memoryref_byteoffset644 = shl i64 %i279, 2
  %memoryref_data_byteoffset645 = getelementptr i8, ptr %memoryref_data639, i64 %memoryref_byteoffset644
  %i282 = ptrtoint ptr %memoryref_data_byteoffset645 to i64
  %i283 = sub i64 %i282, %i251
  %memoryref_isinbounds650 = icmp ult i64 %i283, %memoryref_bytelen649.peel
  %"memoryref_isinbounds&notovflw651" = and i1 %memoryref_ovflw643.not, %memoryref_isinbounds650
  br i1 %"memoryref_isinbounds&notovflw651", label %load662, label %oob652

L1652:                                            ; preds = %load662, %load662.peel
  %.lcssa2849 = phi i64 [ %i367, %load662 ], [ 1, %load662.peel ]
  %i284 = getelementptr inbounds i8, ptr %"new::Tuple667", i64 16
  %i285 = getelementptr inbounds i8, ptr %"new::Tuple667", i64 8
  store i64 %.lcssa2849, ptr %"new::Tuple667", align 1
  store i64 %i239, ptr %i285, align 1
  store i64 %i226, ptr %i284, align 1
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_3998(ptr %"pred::Array", ptr nocapture readonly %"new::Tuple667") #22
  unreachable

L1655:                                            ; preds = %load662
  %i286 = add i64 %i367, %i244
  %i287 = add i64 %i286, -1
  %.not1747 = icmp ult i64 %i287, %i216
  br i1 %.not1747, label %L1688, label %L1685

L1685.loopexit4474:                               ; preds = %L1655.peel
  %i288 = add i64 %i244, 1
  br label %L1685

L1685:                                            ; preds = %L1685.loopexit4474, %L1655
  %.lcssa2857 = phi i64 [ %i288, %L1685.loopexit4474 ], [ %i286, %L1655 ]
  store i64 %.lcssa2857, ptr %"new::Tuple738", align 8
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_3993(ptr %"pred::Array", ptr nocapture readonly %"new::Tuple738") #22
  unreachable

L1688:                                            ; preds = %L1655
  %i289 = add i64 %i287, %memory_len681.peel
  %memoryref_ovflw682.not = icmp ult i64 %i289, %i257
  %memoryref_byteoffset683 = shl i64 %i287, 2
  %memoryref_data_byteoffset684 = getelementptr i8, ptr %memoryref_data678, i64 %memoryref_byteoffset683
  %i290 = ptrtoint ptr %memoryref_data_byteoffset684 to i64
  %i291 = sub i64 %i290, %i259
  %memoryref_isinbounds689 = icmp ult i64 %i291, %memoryref_bytelen688.peel
  %"memoryref_isinbounds&notovflw690" = and i1 %memoryref_ovflw682.not, %memoryref_isinbounds689
  br i1 %"memoryref_isinbounds&notovflw690", label %load701, label %oob691

L1733:                                            ; preds = %load701, %load701.peel
  %.lcssa2762 = phi i64 [ %i272, %load701 ], [ 1, %load701.peel ]
  %i292 = getelementptr inbounds i8, ptr %"new::Tuple703", i64 16
  %i293 = getelementptr inbounds i8, ptr %"new::Tuple703", i64 8
  store i64 %.lcssa2762, ptr %"new::Tuple703", align 1
  store i64 %i232, ptr %i293, align 1
  store i64 %i221, ptr %i292, align 1
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %"new::Array225", ptr %gc_slot_addr_08726, align 8
  call fastcc void @julia_throw_boundserror_3998(ptr %"new::Array225", ptr nocapture readonly %"new::Tuple703") #22
  unreachable

L1736:                                            ; preds = %load701
  %memoryref_offset714 = add i64 %i271, %reass.mul1781
  %i294 = add i64 %memoryref_offset714, %memory_len715.peel
  %memoryref_ovflw716.not = icmp ult i64 %i294, %i266
  %memoryref_byteoffset717 = shl i64 %memoryref_offset714, 2
  %memoryref_data_byteoffset718 = getelementptr i8, ptr %memoryref_data712, i64 %memoryref_byteoffset717
  %i295 = ptrtoint ptr %memoryref_data_byteoffset718 to i64
  %i296 = sub i64 %i295, %i268
  %memoryref_isinbounds723 = icmp ult i64 %i296, %memoryref_bytelen722.peel
  %"memoryref_isinbounds&notovflw724" = and i1 %memoryref_ovflw716.not, %memoryref_isinbounds723
  br i1 %"memoryref_isinbounds&notovflw724", label %load735, label %oob725

L1762:                                            ; preds = %load735, %load735.peel
  %.not1753 = icmp eq i64 %i232, %"pred::Array.size175.sroa.2.0.copyload.mux"
  br i1 %.not1753, label %L1767, label %L1531

L1767:                                            ; preds = %L1762
  %.not1754 = icmp eq i64 %i221, %"pred::Array.size175.sroa.3.0.copyload.mux"
  br i1 %.not1754, label %L1776, label %L1529

L1776:                                            ; preds = %load331, %L1767, %L1516, %L1122
  %i297 = mul i64 %value_phi336.size397.sroa.3.0.copyload, %i158
  switch i64 %i297, label %L1830 [
    i64 0, label %L1877
    i64 1, label %L1808
  ]

L1808:                                            ; preds = %L1776
  %memoryref_data340 = load ptr, ptr %"new::Array225", align 8
  %memoryref_mem358 = load ptr, ptr %i331, align 8
  %memory_len342 = load i64, ptr %memoryref_mem358, align 8
  %memory_data_ptr346 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem358, i64 0, i32 1
  %memory_data347 = load ptr, ptr %memory_data_ptr346, align 8
  %i298 = ptrtoint ptr %memoryref_data340 to i64
  %i299 = ptrtoint ptr %memory_data347 to i64
  %i300 = sub i64 %i298, %i299
  %memoryref_bytelen348 = shl nuw nsw i64 %memory_len342, 2
  %memoryref_isinbounds349 = icmp ult i64 %i300, %memoryref_bytelen348
  br i1 %memoryref_isinbounds349, label %idxend356, label %oob351

L1830:                                            ; preds = %L1776
  %memoryref_data368 = load ptr, ptr %"new::Array225", align 8
  %memoryref_mem386 = load ptr, ptr %i331, align 8
  %memory_len370 = load i64, ptr %memoryref_mem386, align 8
  %i301 = shl nuw nsw i64 %memory_len370, 1
  %memory_data_ptr374 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem386, i64 0, i32 1
  %memory_data375 = load ptr, ptr %memory_data_ptr374, align 8
  %i302 = ptrtoint ptr %memoryref_data368 to i64
  %i303 = ptrtoint ptr %memory_data375 to i64
  %i304 = sub i64 %i302, %i303
  %memoryref_bytelen376 = shl nuw nsw i64 %memory_len370, 2
  %memoryref_isinbounds377 = icmp ult i64 %i304, %memoryref_bytelen376
  br i1 %memoryref_isinbounds377, label %idxend384, label %oob379

L1859:                                            ; preds = %load422, %scalar.ph
  %value_phi3922719 = phi i64 [ %i305, %load422 ], [ %bc.resume.val, %scalar.ph ]
  %value_phi3912718 = phi float [ %i364, %load422 ], [ %bc.merge.rdx, %scalar.ph ]
  %i305 = add nuw nsw i64 %value_phi3922719, 1
  %i306 = add nuw nsw i64 %value_phi3922719, %memory_len370
  %memoryref_ovflw403.not = icmp ult i64 %i306, %i301
  %memoryref_byteoffset404 = shl i64 %value_phi3922719, 2
  %memoryref_data_byteoffset405 = getelementptr i8, ptr %memoryref_data368, i64 %memoryref_byteoffset404
  %i307 = ptrtoint ptr %memoryref_data_byteoffset405 to i64
  %i308 = sub i64 %i307, %i303
  %memoryref_isinbounds410 = icmp ult i64 %i308, %memoryref_bytelen376
  %"memoryref_isinbounds&notovflw411" = and i1 %memoryref_ovflw403.not, %memoryref_isinbounds410
  br i1 %"memoryref_isinbounds&notovflw411", label %load422, label %oob412

L1877:                                            ; preds = %load422, %load389, %load361, %L1776
  %value_phi337 = phi float [ %i338, %load361 ], [ 0.000000e+00, %L1776 ], [ %i340, %load389 ], [ %i364, %load422 ]
  %frame.prev9137 = load ptr, ptr %frame.prev, align 8
  store ptr %frame.prev9137, ptr %newly_emitted_pgc_stack, align 8
  ret float %value_phi337

nonemptymem:                                      ; preds = %L46
  %i309 = icmp ult i64 %"y::Array.size.0.copyload", 2305843009213693952
  br i1 %i309, label %pass, label %fail

fail:                                             ; preds = %nonemptymem
  call void @jl_argument_error(ptr nonnull @_j_str_invalid_GenericMemory_siz____3)
  unreachable

pass:                                             ; preds = %nonemptymem
  %i310 = shl nuw nsw i64 %"y::Array.size.0.copyload", 2
  %ptls_load5 = load ptr, ptr %ptls_field, align 8
  %"Memory{Int32}[]" = call noalias nonnull align 16 "enzyme_ReadOnlyOrThrow" "enzyme_inactive" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Integer}" ptr @jl_alloc_genericmemory_unchecked(ptr %ptls_load5, i64 %i310, ptr nonnull inttoptr (i64 4866807648 to ptr)) #24
  store i64 %"y::Array.size.0.copyload", ptr %"Memory{Int32}[]", align 8
  br label %retval

retval:                                           ; preds = %pass, %L46.retval_crit_edge
  %memoryref_mem47 = phi ptr [ %"Memory{Int32}[]", %pass ], [ @"ejl_inserted$jl_global_3957$false$4866807728", %L46.retval_crit_edge ]
  %memory_data_ptr = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem47, i64 0, i32 1
  %memory_data = load ptr, ptr %memory_data_ptr, align 8
  %gc_slot_addr_08726 = getelementptr inbounds ptr, ptr %gcframe9139, i64 2
  store ptr %memoryref_mem47, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Array_3959$false$4896072880", i64 16) ]
  %ptls_load8948 = load ptr, ptr %ptls_field, align 8
  %"new::Array" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8948, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_Array_3959$false$4896072880" to i64)) #15
  %"new::Array.tag_addr" = getelementptr inbounds ptr, ptr %"new::Array", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Array_3959$false$4896072880", ptr %"new::Array.tag_addr" unordered, align 8
  %i311 = getelementptr inbounds i8, ptr %"new::Array", i64 8
  store ptr %memory_data, ptr %"new::Array", align 8
  store ptr %memoryref_mem47, ptr %i311, align 8
  %"new::Array.size_ptr" = getelementptr inbounds i8, ptr %"new::Array", i64 16
  store i64 %"y::Array.size.0.copyload", ptr %"new::Array.size_ptr", align 8
  br i1 %memorynew_empty, label %L144, label %L79.preheader

L79.preheader:                                    ; preds = %retval
  %memoryref_data = load ptr, ptr %"y::Array", align 8
  %i312 = getelementptr inbounds { ptr, ptr }, ptr %"y::Array", i64 0, i32 1
  %memoryref_mem = load ptr, ptr %i312, align 8
  %memory_data_ptr19 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem, i64 0, i32 1
  %i313 = add nuw nsw i64 %"y::Array.size.0.copyload", 1
  %memory_len = load i64, ptr %memoryref_mem, align 8
  %i314 = shl nuw nsw i64 %memory_len, 1
  %memory_data20 = load ptr, ptr %memory_data_ptr19, align 8
  %i315 = ptrtoint ptr %memory_data20 to i64
  %memoryref_bytelen = shl nuw nsw i64 %memory_len, 2
  %i316 = icmp eq i64 %memory_len, 0
  br label %L79

oob:                                              ; preds = %L95
  store ptr %memoryref_mem, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3960$false$4867651680", i64 16) ]
  %ptls_load8952 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8952, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3960$false$4867651680" to i64)) #15
  %"box::GenericMemoryRef.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3960$false$4867651680", ptr %"box::GenericMemoryRef.tag_addr" unordered, align 8
  store ptr %memoryref_data, ptr %"box::GenericMemoryRef", align 8
  %.repack1656 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef", i64 0, i32 1
  store ptr %memoryref_mem, ptr %.repack1656, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef", i64 %value_phi14)
  unreachable

idxend:                                           ; preds = %L95
  br i1 %i316, label %oob24, label %load

oob24:                                            ; preds = %idxend
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem, i64 1)
  unreachable

load:                                             ; preds = %idxend
  %memoryref_data25 = getelementptr inbounds i8, ptr %memoryref_data, i64 %memoryref_byteoffset
  %i317 = load i32, ptr %memoryref_data25, align 4
  %.not1658 = icmp sgt i32 %i317, -1
  br i1 %.not1658, label %L126, label %L104

oob41:                                            ; preds = %L126
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824", i64 16) ]
  %"box::GenericMemoryRef44" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8948, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824" to i64)) #15
  %"box::GenericMemoryRef44.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef44", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824", ptr %"box::GenericMemoryRef44.tag_addr" unordered, align 8
  store ptr %memory_data, ptr %"box::GenericMemoryRef44", align 8
  %.repack1659 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef44", i64 0, i32 1
  store ptr %memoryref_mem47, ptr %.repack1659, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef44", i64 %value_phi14)
  unreachable

idxend45:                                         ; preds = %L126
  %i318 = icmp eq i64 %memory_len31, 0
  br i1 %i318, label %oob49, label %load50

oob49:                                            ; preds = %idxend45
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem47, i64 1)
  unreachable

load50:                                           ; preds = %idxend45
  %memoryref_data51 = getelementptr inbounds i8, ptr %memory_data, i64 %memoryref_byteoffset
  store i32 %i317, ptr %memoryref_data51, align 4
  %.not1661 = icmp eq i64 %value_phi14, %"y::Array.size.0.copyload"
  %i319 = add nuw nsw i64 %value_phi14, 1
  br i1 %.not1661, label %L144, label %L79

nonemptymem78:                                    ; preds = %L259
  %i320 = icmp ult i64 %i41, 2305843009213693952
  br i1 %i320, label %pass80, label %fail79

fail79:                                           ; preds = %nonemptymem78
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @jl_argument_error(ptr nonnull @_j_str_invalid_GenericMemory_siz____3)
  unreachable

pass80:                                           ; preds = %nonemptymem78
  %i321 = shl nuw nsw i64 %i41, 2
  %ptls_load83 = load ptr, ptr %ptls_field, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  %gc_slot_addr_18735 = getelementptr inbounds ptr, ptr %gcframe9139, i64 3
  store ptr %i25, ptr %gc_slot_addr_18735, align 8
  %"Memory{Float32}[]" = call noalias nonnull align 16 dereferenceable(16) "enzyme_ReadOnlyOrThrow" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@float}" ptr @jl_alloc_genericmemory_unchecked(ptr %ptls_load83, i64 %i321, ptr nonnull inttoptr (i64 5309578512 to ptr)) #24
  store i64 %i41, ptr %"Memory{Float32}[]", align 8
  br label %retval84

retval84:                                         ; preds = %pass80, %L259.retval84_crit_edge
  %memoryref_mem162 = phi ptr [ %"Memory{Float32}[]", %pass80 ], [ @"ejl_inserted$jl_global_3970$false$4627012928", %L259.retval84_crit_edge ]
  %memory_data_ptr85 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem162, i64 0, i32 1
  %memory_data86 = load ptr, ptr %memory_data_ptr85, align 8
  %gc_slot_addr_18736 = getelementptr inbounds ptr, ptr %gcframe9139, i64 3
  store ptr %i25, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem162, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Array_3972$false$5309578256", i64 16) ]
  %ptls_load8966 = load ptr, ptr %ptls_field, align 8
  %"new::Array91" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8966, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Array_3972$false$5309578256" to i64)) #15
  %"new::Array91.tag_addr" = getelementptr inbounds ptr, ptr %"new::Array91", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Array_3972$false$5309578256", ptr %"new::Array91.tag_addr" unordered, align 8
  %i322 = getelementptr inbounds i8, ptr %"new::Array91", i64 8
  store ptr %memory_data86, ptr %"new::Array91", align 8
  store ptr %memoryref_mem162, ptr %i322, align 8
  %"new::Array91.size_ptr" = getelementptr inbounds i8, ptr %"new::Array91", i64 16
  store i64 %value_phi170.size232.sroa.0.0.copyload, ptr %"new::Array91.size_ptr", align 8
  %"new::Tuple88.sroa.2.0.new::Array91.size_ptr.sroa_idx" = getelementptr inbounds i8, ptr %"new::Array91", i64 24
  store i64 %.size.sroa.2.0.copyload, ptr %"new::Tuple88.sroa.2.0.new::Array91.size_ptr.sroa_idx", align 8
  %"new::Tuple88.sroa.3.0.new::Array91.size_ptr.sroa_idx" = getelementptr inbounds i8, ptr %"new::Array91", i64 32
  store i64 %.size.sroa.3.0.copyload, ptr %"new::Tuple88.sroa.3.0.new::Array91.size_ptr.sroa_idx", align 8
  %.size93.sroa.0.0.copyload = load i64, ptr %.size_ptr, align 8
  %.size93.sroa.2.0.copyload = load i64, ptr %.size.sroa.2.0..size_ptr.sroa_idx, align 8
  %.size93.sroa.3.0.copyload = load i64, ptr %.size.sroa.3.0..size_ptr.sroa_idx, align 8
  %.not1666 = icmp eq i64 %.size93.sroa.0.0.copyload, %i10
  %.not1667 = icmp eq i64 %.size93.sroa.2.0.copyload, 1
  %or.cond = select i1 %.not1666, i1 %.not1667, i1 false
  %.not1668 = icmp eq i64 %.size93.sroa.3.0.copyload, 1
  %or.cond1774 = select i1 %or.cond, i1 %.not1668, i1 false
  br i1 %or.cond1774, label %L381, label %L505

oob130:                                           ; preds = %L461
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem137, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824", i64 16) ]
  %ptls_load8971 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef134" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8971, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824" to i64)) #15
  %"box::GenericMemoryRef134.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef134", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824", ptr %"box::GenericMemoryRef134.tag_addr" unordered, align 8
  store ptr %memoryref_data117, ptr %"box::GenericMemoryRef134", align 8
  %.repack1674 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef134", i64 0, i32 1
  store ptr %memoryref_mem137, ptr %.repack1674, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef134", i64 %value_phi111)
  unreachable

idxend135:                                        ; preds = %L461
  %i323 = icmp eq i64 %memory_len120, 0
  br i1 %i323, label %oob139, label %load140

oob139:                                           ; preds = %idxend135
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem137, i64 1)
  unreachable

load140:                                          ; preds = %idxend135
  %memoryref_data141 = getelementptr inbounds i8, ptr %memoryref_data117, i64 %memoryref_byteoffset122
  %i324 = load i32, ptr %memoryref_data141, align 4
  %i325 = icmp eq i32 %i324, %i58
  %i326 = uitofp i1 %i325 to float
  %exitcond3460.not = icmp eq i64 %value_phi111, %i56
  br i1 %exitcond3460.not, label %L484, label %L487

oob156:                                           ; preds = %L487
  store ptr null, ptr %gc_slot_addr_18736, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load8977 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef159" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8977, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef159.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef159", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef159.tag_addr" unordered, align 8
  store ptr %memory_data86, ptr %"box::GenericMemoryRef159", align 8
  %.repack1677 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef159", i64 0, i32 1
  store ptr %memoryref_mem162, ptr %.repack1677, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef159", i64 %value_phi111)
  unreachable

idxend160:                                        ; preds = %L487
  %i327 = icmp eq i64 %memory_len146, 0
  br i1 %i327, label %oob164, label %load165

oob164:                                           ; preds = %idxend160
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem162, i64 1)
  unreachable

load165:                                          ; preds = %idxend160
  %memoryref_data166 = getelementptr inbounds i8, ptr %memory_data86, i64 %memoryref_byteoffset122
  store float %i326, ptr %memoryref_data166, align 4
  %.not1679 = icmp eq i64 %value_phi111, %i10
  %i328 = add nuw nsw i64 %value_phi111, 1
  br i1 %.not1679, label %L864.loopexit, label %L400

nonemptymem209:                                   ; preds = %L987
  %i329 = icmp ult i64 %i164, 2305843009213693952
  br i1 %i329, label %pass212, label %fail211

fail211:                                          ; preds = %nonemptymem209
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @jl_argument_error(ptr nonnull @_j_str_invalid_GenericMemory_siz____3)
  unreachable

pass212:                                          ; preds = %nonemptymem209
  %i330 = shl nuw nsw i64 %i164, 2
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  store ptr %"new::Array91", ptr %gc_slot_addr_18736, align 8
  %"Memory{Float32}[]216" = call noalias nonnull align 16 dereferenceable(16) "enzyme_ReadOnlyOrThrow" "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Pointer, [-1,8,-1]:Float@float}" ptr @jl_alloc_genericmemory_unchecked(ptr %ptls_load8966, i64 %i330, ptr nonnull inttoptr (i64 5309578512 to ptr)) #24
  store i64 %i164, ptr %"Memory{Float32}[]216", align 8
  br label %retval217

retval217:                                        ; preds = %pass212, %L987.retval217_crit_edge
  %memoryref_mem328 = phi ptr [ %"Memory{Float32}[]216", %pass212 ], [ @"ejl_inserted$jl_global_3970$false$4627012928", %L987.retval217_crit_edge ]
  %memory_data_ptr218 = getelementptr inbounds { i64, ptr }, ptr %memoryref_mem328, i64 0, i32 1
  %memory_data219 = load ptr, ptr %memory_data_ptr218, align 8
  store ptr %"new::Array91", ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem328, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_Array_3972$false$5309578256", i64 16) ]
  %ptls_load8990 = load ptr, ptr %ptls_field, align 8
  %"new::Array225" = call noalias nonnull align 8 dereferenceable(48) ptr @ijl_gc_small_alloc(ptr %ptls_load8990, i32 520, i32 48, i64 ptrtoint (ptr @"ejl_inserted$_Core_Array_3972$false$5309578256" to i64)) #15
  %"new::Array225.tag_addr" = getelementptr inbounds ptr, ptr %"new::Array225", i64 -1
  store atomic ptr @"ejl_inserted$_Core_Array_3972$false$5309578256", ptr %"new::Array225.tag_addr" unordered, align 8
  %i331 = getelementptr inbounds i8, ptr %"new::Array225", i64 8
  store ptr %memory_data219, ptr %"new::Array225", align 8
  store ptr %memoryref_mem328, ptr %i331, align 8
  %"new::Array225.size_ptr" = getelementptr inbounds i8, ptr %"new::Array225", i64 16
  store i64 %value_phi336.size397.sroa.0.0.copyload, ptr %"new::Array225.size_ptr", align 8
  %"new::Tuple221.sroa.2.0.new::Array225.size_ptr.sroa_idx" = getelementptr inbounds i8, ptr %"new::Array225", i64 24
  store i64 %value_phi336.size397.sroa.2.0.copyload, ptr %"new::Tuple221.sroa.2.0.new::Array225.size_ptr.sroa_idx", align 8
  %"new::Tuple221.sroa.3.0.new::Array225.size_ptr.sroa_idx" = getelementptr inbounds i8, ptr %"new::Array225", i64 32
  store i64 %value_phi336.size397.sroa.3.0.copyload, ptr %"new::Tuple221.sroa.3.0.new::Array225.size_ptr.sroa_idx", align 8
  %or.cond1768 = and i1 %i135, %i142
  %or.cond1775 = and i1 %or.cond1768, %i149
  br i1 %or.cond1775, label %L1122, label %L1224

oob268:                                           ; preds = %L1160
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem275, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load8995 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef272" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load8995, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef272.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef272", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef272.tag_addr" unordered, align 8
  store ptr %memoryref_data255, ptr %"box::GenericMemoryRef272", align 8
  %.repack1710 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef272", i64 0, i32 1
  store ptr %memoryref_mem275, ptr %.repack1710, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef272", i64 %value_phi251)
  unreachable

idxend273:                                        ; preds = %L1160
  br i1 %i182, label %oob277, label %L1182

oob277:                                           ; preds = %idxend273
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem275, i64 1)
  unreachable

oob295:                                           ; preds = %L1182
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem302, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9003 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef299" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9003, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef299.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef299", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef299.tag_addr" unordered, align 8
  store ptr %memoryref_data282, ptr %"box::GenericMemoryRef299", align 8
  %.repack1713 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef299", i64 0, i32 1
  store ptr %memoryref_mem302, ptr %.repack1713, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef299", i64 %value_phi251)
  unreachable

idxend300:                                        ; preds = %L1182
  %i332 = icmp eq i64 %memory_len285, 0
  br i1 %i332, label %oob304, label %load305

oob304:                                           ; preds = %idxend300
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem302, i64 1)
  unreachable

load305:                                          ; preds = %idxend300
  %memoryref_data306 = getelementptr inbounds i8, ptr %memoryref_data282, i64 %memoryref_byteoffset260
  %i333 = load float, ptr %memoryref_data306, align 4
  %i334 = fmul float %i187, %i333
  %exitcond3467.not = icmp eq i64 %value_phi251, %i179
  br i1 %exitcond3467.not, label %L1203, label %L1206

oob321:                                           ; preds = %L1206
  store ptr null, ptr %gc_slot_addr_18736, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9009 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef325" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9009, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef325.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef325", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef325.tag_addr" unordered, align 8
  store ptr %memory_data219, ptr %"box::GenericMemoryRef325", align 8
  %.repack1716 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef325", i64 0, i32 1
  store ptr %memoryref_mem328, ptr %.repack1716, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef325", i64 %value_phi251)
  unreachable

idxend326:                                        ; preds = %L1206
  %i335 = icmp eq i64 %memory_len311, 0
  br i1 %i335, label %oob330, label %load331

oob330:                                           ; preds = %idxend326
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem328, i64 1)
  unreachable

load331:                                          ; preds = %idxend326
  %memoryref_data332 = getelementptr inbounds i8, ptr %memory_data219, i64 %memoryref_byteoffset260
  store float %i334, ptr %memoryref_data332, align 4
  %.not1718 = icmp eq i64 %i175, %value_phi251
  %i336 = add nuw nsw i64 %value_phi251, 1
  br i1 %.not1718, label %L1776, label %L1160

oob351:                                           ; preds = %L1808
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem358, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9017 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef355" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9017, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef355.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef355", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef355.tag_addr" unordered, align 8
  store ptr %memoryref_data340, ptr %"box::GenericMemoryRef355", align 8
  %.repack1757 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef355", i64 0, i32 1
  store ptr %memoryref_mem358, ptr %.repack1757, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef355", i64 1)
  unreachable

idxend356:                                        ; preds = %L1808
  %i337 = icmp eq i64 %memory_len342, 0
  br i1 %i337, label %oob360, label %load361

oob360:                                           ; preds = %idxend356
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem358, i64 1)
  unreachable

load361:                                          ; preds = %idxend356
  %i338 = load float, ptr %memoryref_data340, align 4
  br label %L1877

oob379:                                           ; preds = %L1830
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem386, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9025 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef383" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9025, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef383.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef383", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef383.tag_addr" unordered, align 8
  store ptr %memoryref_data368, ptr %"box::GenericMemoryRef383", align 8
  %.repack1759 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef383", i64 0, i32 1
  store ptr %memoryref_mem386, ptr %.repack1759, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef383", i64 1)
  unreachable

idxend384:                                        ; preds = %L1830
  %i339 = icmp eq i64 %memory_len370, 0
  br i1 %i339, label %oob388, label %load389

oob388:                                           ; preds = %idxend384
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem386, i64 1)
  unreachable

load389:                                          ; preds = %idxend384
  %i340 = load float, ptr %memoryref_data368, align 4
  %.not17612717 = icmp sgt i64 %i297, 1
  br i1 %.not17612717, label %L1859.preheader, label %L1877

L1859.preheader:                                  ; preds = %load389
  %i341 = add nsw i64 %i297, -2
  %i342 = add i64 %i302, 4
  %i343 = sub i64 %i342, %i303
  %umax = call i64 @llvm.umax.i64(i64 %memoryref_bytelen376, i64 %i343)
  %i344 = add i64 %umax, %i303
  %i345 = xor i64 %i302, -1
  %i346 = add i64 %i344, %i345
  %i347 = lshr i64 %i346, 2
  %i348 = add nuw nsw i64 %memory_len370, 1
  %umax8653 = call i64 @llvm.umax.i64(i64 %i301, i64 %i348)
  %i349 = xor i64 %memory_len370, -1
  %i350 = add nsw i64 %umax8653, %i349
  %umin = call i64 @llvm.umin.i64(i64 %i347, i64 %i350)
  %umin8654 = call i64 @llvm.umin.i64(i64 %i341, i64 %umin)
  %min.iters.check = icmp ult i64 %umin8654, 8
  br i1 %min.iters.check, label %scalar.ph, label %vector.ph

vector.ph:                                        ; preds = %L1859.preheader
  %i351 = add nuw nsw i64 %umin8654, 1
  %n.mod.vf = and i64 %i351, 7
  %i352 = icmp eq i64 %n.mod.vf, 0
  %i353 = select i1 %i352, i64 8, i64 %n.mod.vf
  %n.vec = sub nsw i64 %i351, %i353
  %ind.end = add nsw i64 %n.vec, 1
  %i354 = insertelement <4 x float> <float poison, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, float %i340, i64 0
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind = phi <4 x i64> [ <i64 1, i64 2, i64 3, i64 4>, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %vec.phi = phi <4 x float> [ %i354, %vector.ph ], [ %i359, %vector.body ]
  %vec.phi8656 = phi <4 x float> [ <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %vector.ph ], [ %i360, %vector.body ]
  %i355 = extractelement <4 x i64> %vec.ind, i64 0
  %i356 = shl i64 %i355, 2
  %i357 = getelementptr inbounds i8, ptr %memoryref_data368, i64 %i356
  %i358 = getelementptr inbounds float, ptr %i357, i64 4
  %wide.load = load <4 x float>, ptr %i357, align 4
  %wide.load8657 = load <4 x float>, ptr %i358, align 4
  %i359 = fadd reassoc contract <4 x float> %vec.phi, %wide.load
  %i360 = fadd reassoc contract <4 x float> %vec.phi8656, %wide.load8657
  %index.next = add nuw i64 %index, 8
  %vec.ind.next = add <4 x i64> %vec.ind, <i64 8, i64 8, i64 8, i64 8>
  %i361 = icmp eq i64 %index.next, %n.vec
  br i1 %i361, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %bin.rdx = fadd reassoc contract <4 x float> %i360, %i359
  %i362 = call reassoc contract float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %bin.rdx)
  br label %scalar.ph

scalar.ph:                                        ; preds = %middle.block, %L1859.preheader
  %bc.resume.val = phi i64 [ %ind.end, %middle.block ], [ 1, %L1859.preheader ]
  %bc.merge.rdx = phi float [ %i362, %middle.block ], [ %i340, %L1859.preheader ]
  br label %L1859

oob412:                                           ; preds = %L1859
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem386, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9037 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef416" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9037, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef416.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef416", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef416.tag_addr" unordered, align 8
  store ptr %memoryref_data368, ptr %"box::GenericMemoryRef416", align 8
  %.repack1763 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef416", i64 0, i32 1
  store ptr %memoryref_mem386, ptr %.repack1763, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef416", i64 %i305)
  unreachable

load422:                                          ; preds = %L1859
  %memoryref_data423 = getelementptr inbounds i8, ptr %memoryref_data368, i64 %memoryref_byteoffset404
  %i363 = load float, ptr %memoryref_data423, align 4
  %i364 = fadd reassoc contract float %value_phi3912718, %i363
  %exitcond3468.not = icmp eq i64 %i305, %i297
  br i1 %exitcond3468.not, label %L1877, label %L1859

oob652.loopexit4472:                              ; preds = %L1608.peel
  %i365 = add nuw i64 %i238, 1
  br label %oob652

oob652:                                           ; preds = %oob652.loopexit4472, %L1608
  %.lcssa2839 = phi i64 [ %i365, %oob652.loopexit4472 ], [ %i278, %L1608 ]
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem659, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9043 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef656" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9043, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef656.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef656", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef656.tag_addr" unordered, align 8
  store ptr %memoryref_data639, ptr %"box::GenericMemoryRef656", align 8
  %.repack1745 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef656", i64 0, i32 1
  store ptr %memoryref_mem659, ptr %.repack1745, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef656", i64 %.lcssa2839)
  unreachable

oob661:                                           ; preds = %idxend657.peel
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem659, i64 1)
  unreachable

load662:                                          ; preds = %L1608
  %memoryref_data663 = getelementptr inbounds i8, ptr %memoryref_data639, i64 %memoryref_byteoffset644
  %i366 = load float, ptr %memoryref_data663, align 4
  %i367 = select i1 %.not4469, i64 1, i64 %i272
  %i368 = add nsw i64 %i367, -1
  %i369 = icmp ult i64 %i368, %"pred::Array.size175.sroa.0.0.copyload"
  br i1 %i369, label %L1655, label %L1652

oob691.loopexit4475:                              ; preds = %L1688.peel
  %i370 = add nuw i64 %i244, 1
  br label %oob691

oob691:                                           ; preds = %oob691.loopexit4475, %L1688
  %.lcssa2858 = phi i64 [ %i370, %oob691.loopexit4475 ], [ %i286, %L1688 ]
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem698, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9051 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef695" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9051, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef695.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef695", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef695.tag_addr" unordered, align 8
  store ptr %memoryref_data678, ptr %"box::GenericMemoryRef695", align 8
  %.repack1748 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef695", i64 0, i32 1
  store ptr %memoryref_mem698, ptr %.repack1748, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef695", i64 %.lcssa2858)
  unreachable

oob700:                                           ; preds = %idxend696.peel
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem698, i64 1)
  unreachable

load701:                                          ; preds = %L1688
  %memoryref_data702 = getelementptr inbounds i8, ptr %memoryref_data678, i64 %memoryref_byteoffset683
  %i371 = load float, ptr %memoryref_data702, align 4
  %i372 = fmul float %i366, %i371
  %exitcond6335.not = icmp eq i64 %i271, %value_phi336.size397.sroa.0.0.copyload
  br i1 %exitcond6335.not, label %L1733, label %L1736

oob725:                                           ; preds = %L1736, %L1736.peel
  %.lcssa2763 = phi i64 [ %i272, %L1736 ], [ 1, %L1736.peel ]
  %i373 = add i64 %.lcssa2763, %reass.mul1781
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem732, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9059 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef729" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9059, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef729.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef729", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef729.tag_addr" unordered, align 8
  store ptr %memoryref_data712, ptr %"box::GenericMemoryRef729", align 8
  %.repack1750 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef729", i64 0, i32 1
  store ptr %memoryref_mem732, ptr %.repack1750, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef729", i64 %i373)
  unreachable

oob734:                                           ; preds = %idxend730.peel
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem732, i64 1)
  unreachable

load735:                                          ; preds = %L1736
  %memoryref_data736 = getelementptr inbounds i8, ptr %memoryref_data712, i64 %memoryref_byteoffset717
  store float %i372, ptr %memoryref_data736, align 4
  %.not1752 = icmp eq i64 %i272, %"pred::Array.size175.sroa.0.0.copyload.mux"
  br i1 %.not1752, label %L1762, label %L1533

oob1007.loopexit4482:                             ; preds = %L773.peel
  %i374 = add nuw i64 %i101, 1
  br label %oob1007

oob1007:                                          ; preds = %oob1007.loopexit4482, %L773
  %.lcssa3248 = phi i64 [ %i374, %oob1007.loopexit4482 ], [ %i126, %L773 ]
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem1014.peel, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824", i64 16) ]
  %ptls_load9067 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef1011" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9067, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824" to i64)) #15
  %"box::GenericMemoryRef1011.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef1011", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3964$false$4867651824", ptr %"box::GenericMemoryRef1011.tag_addr" unordered, align 8
  store ptr %memoryref_data994.peel, ptr %"box::GenericMemoryRef1011", align 8
  %.repack1691 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef1011", i64 0, i32 1
  store ptr %memoryref_mem1014.peel, ptr %.repack1691, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef1011", i64 %.lcssa3248)
  unreachable

oob1016:                                          ; preds = %idxend1012.peel
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem1014.peel, i64 1)
  unreachable

load1017:                                         ; preds = %L773
  %memoryref_data1018 = getelementptr inbounds i8, ptr %memoryref_data994.peel, i64 %memoryref_byteoffset999
  %i375 = load i32, ptr %memoryref_data1018, align 4
  %i376 = icmp eq i32 %i375, %i122
  %i377 = uitofp i1 %i376 to float
  %exitcond.not6336 = icmp eq i64 %i118, %value_phi170.size232.sroa.0.0.copyload
  br i1 %exitcond.not6336, label %L821, label %L824

oob1041:                                          ; preds = %L824, %L824.peel
  %.lcssa3159 = phi i64 [ %i119, %L824 ], [ 1, %L824.peel ]
  %i378 = add i64 %.lcssa3159, %reass.mul
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr %memoryref_mem1048, ptr %gc_slot_addr_08726, align 8
  call void @llvm.assume(i1 true) [ "align"(ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", i64 16) ]
  %ptls_load9075 = load ptr, ptr %ptls_field, align 8
  %"box::GenericMemoryRef1045" = call noalias nonnull align 8 dereferenceable(32) ptr @ijl_gc_small_alloc(ptr %ptls_load9075, i32 472, i32 32, i64 ptrtoint (ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448" to i64)) #15
  %"box::GenericMemoryRef1045.tag_addr" = getelementptr inbounds ptr, ptr %"box::GenericMemoryRef1045", i64 -1
  store atomic ptr @"ejl_inserted$_Core_GenericMemoryRef_3973$false$5309578448", ptr %"box::GenericMemoryRef1045.tag_addr" unordered, align 8
  store ptr %memoryref_data1028, ptr %"box::GenericMemoryRef1045", align 8
  %.repack1693 = getelementptr inbounds { ptr, ptr }, ptr %"box::GenericMemoryRef1045", i64 0, i32 1
  store ptr %memoryref_mem1048, ptr %.repack1693, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %"box::GenericMemoryRef1045", i64 %i378)
  unreachable

oob1050:                                          ; preds = %idxend1046.peel
  store ptr null, ptr %gc_slot_addr_18736, align 8
  store ptr null, ptr %gc_slot_addr_08726, align 8
  call void @ijl_bounds_error_int(ptr %memoryref_mem1048, i64 1)
  unreachable

load1051:                                         ; preds = %L824
  %memoryref_data1052 = getelementptr inbounds i8, ptr %memoryref_data1028, i64 %memoryref_byteoffset1033
  store float %i377, ptr %memoryref_data1052, align 4
  %.not1695 = icmp eq i64 %i119, %.size93.sroa.0.0.copyload.mux
  br i1 %.not1695, label %L850, label %L624
}

; Function Attrs: nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @julia.safepoint(ptr) local_unnamed_addr #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64) #3

; Function Attrs: noreturn
declare void @jl_argument_error(ptr) local_unnamed_addr #4

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc") memory(inaccessiblemem: readwrite)
declare noalias nonnull dereferenceable(16) ptr @jl_alloc_genericmemory_unchecked(ptr, i64, ptr) local_unnamed_addr #5

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite)
declare noalias nonnull ptr @julia.gc_alloc_obj(ptr, i64, ptr) local_unnamed_addr #6

; Function Attrs: noreturn
declare void @ijl_bounds_error_int(ptr, i64) local_unnamed_addr #4

; Function Attrs: nofree norecurse nosync nounwind speculatable willreturn memory(argmem: read)
declare noundef nonnull ptr @julia.gc_loaded(ptr nocapture noundef nonnull readnone, ptr noundef nonnull readnone) local_unnamed_addr #7

; Function Attrs: nofree
declare nonnull ptr @ijl_invoke(ptr, ptr noalias nocapture noundef readonly, i32, ptr) #8

; Function Attrs: nofree
declare nonnull ptr @julia.call2(ptr, ptr, ptr, ...) local_unnamed_addr #8

; Function Attrs: nofree nounwind willreturn memory(read, inaccessiblemem: readwrite)
declare nonnull align 8 dereferenceable(4) ptr @ijl_box_uint32(i32 zeroext) local_unnamed_addr #9

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #10

; Function Attrs: noreturn
declare void @ijl_throw(ptr) local_unnamed_addr #11

; Function Attrs: nofree norecurse nounwind memory(inaccessiblemem: readwrite)
declare void @julia.write_barrier(ptr readonly, ...) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare noalias nonnull align 8 dereferenceable(8) ptr @ijl_box_int64(i64 signext) local_unnamed_addr #13

declare void @julia_throw_boundserror_3998(ptr, ptr)

declare void @julia_throw_boundserror_4004(ptr, ptr)

declare void @julia_throw_boundserror_4006(ptr, ptr)

declare void @julia_throw_boundserror_4015(ptr, ptr)

declare void @julia_throw_boundserror_4012(ptr, ptr)

declare void @julia_throw_boundserror_4008(ptr, i64)

declare void @julia_throw_boundserror_4037(ptr, ptr)

declare void @julia_throw_boundserror_3993(ptr, ptr)

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #3

declare void @julia__throw_dmrs_4039(i64, ptr)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #14

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #14

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #3

; Function Attrs: nounwind willreturn allockind("alloc") allocsize(2) memory(argmem: read, inaccessiblemem: readwrite)
declare noalias nonnull ptr @ijl_gc_small_alloc(ptr, i32, i32, i64) #15

declare noalias nonnull ptr @julia.new_gc_frame(i32)

declare void @julia.push_gc_frame(ptr, i32)

declare ptr @julia.get_gc_frame_slot(ptr, i32)

declare void @julia.pop_gc_frame(ptr)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #16

; Function Attrs: nounwind willreturn allockind("alloc") allocsize(1) memory(argmem: read, inaccessiblemem: readwrite)
declare noalias nonnull ptr @julia.gc_alloc_bytes(ptr, i64, i64) #17

; Function Attrs: memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @julia.queue_gc_root(ptr) #18

; Function Attrs: memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @ijl_gc_queue_root(ptr) #18

; Function Attrs: nounwind willreturn allockind("alloc") allocsize(1) memory(argmem: read, inaccessiblemem: readwrite)
declare noalias nonnull ptr @ijl_gc_big_alloc(ptr, i64, i64) #17

; Function Attrs: nounwind willreturn allockind("alloc") allocsize(1) memory(argmem: read, inaccessiblemem: readwrite)
declare noalias nonnull ptr @ijl_gc_alloc_typed(ptr, i64, i64) #17

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #19

declare fastcc [1 x ptr] @julia_ArgumentError_4002()

declare fastcc ptr @julia_reshape_4017(ptr)

attributes #0 = { alwaysinline "enzyme_ta_norecur" "enzymejl_mi"="5312053392" "enzymejl_rt"="4896071888" "enzymejl_world"="38715" "frame-pointer"="all" }
attributes #1 = { nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="38715" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { noreturn "enzymejl_world"="38715" }
attributes #5 = { mustprogress nofree nounwind willreturn allockind("alloc") memory(inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" "enzyme_no_escaping_allocation" "enzymejl_world"="38715" }
attributes #6 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) memory(inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" "enzyme_no_escaping_allocation" "enzymejl_world"="38715" }
attributes #7 = { nofree norecurse nosync nounwind speculatable willreturn memory(argmem: read) "enzyme_nocache" "enzyme_shouldrecompute" "enzymejl_world"="38715" }
attributes #8 = { nofree "enzymejl_world"="38715" }
attributes #9 = { nofree nounwind willreturn memory(read, inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="38715" }
attributes #10 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #11 = { noreturn "enzyme_no_escaping_allocation" "enzymejl_world"="38715" }
attributes #12 = { nofree norecurse nounwind memory(inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="38715" }
attributes #13 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_LocalReadOnlyOrThrow" "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="38715" }
attributes #14 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #15 = { nounwind willreturn allockind("alloc") allocsize(2) memory(argmem: read, inaccessiblemem: readwrite) }
attributes #16 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #17 = { nounwind willreturn allockind("alloc") allocsize(1) memory(argmem: read, inaccessiblemem: readwrite) }
attributes #18 = { memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #19 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #20 = { nounwind memory(none) }
attributes #21 = { nounwind willreturn memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }
attributes #22 = { noreturn }
attributes #23 = { nounwind willreturn memory(read, inaccessiblemem: readwrite) }
attributes #24 = { nounwind willreturn allockind("alloc") memory(inaccessiblemem: readwrite) "enzyme_no_escaping_allocation" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}