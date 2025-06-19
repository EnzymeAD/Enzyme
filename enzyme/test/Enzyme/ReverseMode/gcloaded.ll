; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,loop(loop-deletion),%simplifycfg,correlated-propagation,adce,instsimplify)" -enzyme-preopt=false -S | FileCheck %s; fi

; ModuleID = 'start'
source_filename = "start"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"
target triple = "arm64-apple-darwin24.0.0"

; Function Attrs: nofree memory(none)
declare {}*** @julia.get_pgcstack() local_unnamed_addr #0

; Function Attrs: nofree memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @julia.safepoint(i64*) local_unnamed_addr #1

; Function Attrs: nofree norecurse nosync nounwind speculatable willreturn memory(none)
declare noundef nonnull {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* nocapture noundef nonnull readnone, {} addrspace(10)** noundef nonnull readnone) local_unnamed_addr #2

; Function Attrs: nofree memory(read, argmem: none, inaccessiblemem: none)
define internal fastcc "enzyme_type"="{[-1]:Float@double}" double @julia__mapreduce_833({} addrspace(10)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4730247120" "enzymejl_parmtype_ref"="2" %0) unnamed_addr #3 !dbg !8 {
top:
  %pgcstack = call {}*** @julia.get_pgcstack()
  %ptls_field16 = getelementptr inbounds {}**, {}*** %pgcstack, i64 2
  %1 = bitcast {}*** %ptls_field16 to i64***
  %ptls_load1718 = load i64**, i64*** %1, align 8, !tbaa !12
  %2 = getelementptr inbounds i64*, i64** %ptls_load1718, i64 2
  %safepoint = load i64*, i64** %2, align 8, !tbaa !16
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %safepoint) #6, !dbg !18
  fence syncscope("singlethread") seq_cst
  %3 = bitcast {} addrspace(10)* %0 to i8 addrspace(10)*, !dbg !19
  %4 = addrspacecast i8 addrspace(10)* %3 to i8 addrspace(11)*, !dbg !19
  %5 = getelementptr inbounds i8, i8 addrspace(11)* %4, i64 16, !dbg !19
  %6 = bitcast i8 addrspace(11)* %5 to i64 addrspace(11)*, !dbg !19
  %7 = load i64, i64 addrspace(11)* %6, align 8, !dbg !19, !tbaa !29, !alias.scope !32, !noalias !35, !enzyme_type !40, !enzymejl_source_type_Int64 !11, !enzymejl_byref_BITS_VALUE !11, !enzyme_inactive !11
  switch i64 %7, label %L29 [
    i64 0, label %common.ret
    i64 1, label %L23
  ], !dbg !42

common.ret:                                       ; preds = %L74, %L45, %top, %L97, %L23
  %common.ret.op = phi double [ %18, %L23 ], [ %42, %L97 ], [ 0.000000e+00, %top ], [ %35, %L45 ], [ %41, %L74 ]
  ret double %common.ret.op, !dbg !43

L23:                                              ; preds = %top
  %8 = bitcast {} addrspace(10)* %0 to { i8*, {} addrspace(10)* } addrspace(10)*, !dbg !44
  %9 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %8 to { i8*, {} addrspace(10)* } addrspace(11)*, !dbg !44
  %10 = bitcast {} addrspace(10)* %0 to {} addrspace(10)** addrspace(10)*, !dbg !44
  %11 = addrspacecast {} addrspace(10)** addrspace(10)* %10 to {} addrspace(10)** addrspace(11)*, !dbg !44
  %12 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %11, align 8, !dbg !44, !tbaa !48, !alias.scope !32, !noalias !35, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !11, !enzymejl_source_type_Ptr\7BFloat64\7D !11, !enzyme_nocache !11
  %13 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %9, i64 0, i32 1, !dbg !44
  %14 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %13, align 8, !dbg !44, !tbaa !48, !alias.scope !32, !noalias !35, !dereferenceable_or_null !53, !align !54, !enzyme_type !55, !enzymejl_source_type_Memory\7BFloat64\7D !11, !enzymejl_byref_MUT_REF !11
  %15 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %14, {} addrspace(10)** noundef %12), !dbg !44
  %16 = bitcast {} addrspace(10)* addrspace(13)* %15 to double addrspace(13)*, !dbg !44
  %17 = load double, double addrspace(13)* %16, align 8, !dbg !44, !tbaa !57, !alias.scope !60, !noalias !61, !enzyme_type !62, !enzymejl_byref_BITS_VALUE !11, !enzymejl_source_type_Float64 !11
  %18 = fmul double %17, %17, !dbg !63
  br label %common.ret

L29:                                              ; preds = %top
  %19 = icmp sgt i64 %7, 15, !dbg !72
  br i1 %19, label %L97, label %L45, !dbg !75

L45:                                              ; preds = %L29
  %20 = bitcast {} addrspace(10)* %0 to { i8*, {} addrspace(10)* } addrspace(10)*, !dbg !76
  %21 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %20 to { i8*, {} addrspace(10)* } addrspace(11)*, !dbg !76
  %22 = bitcast {} addrspace(10)* %0 to {} addrspace(10)** addrspace(10)*, !dbg !76
  %23 = addrspacecast {} addrspace(10)** addrspace(10)* %22 to {} addrspace(10)** addrspace(11)*, !dbg !76
  %24 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %23, align 8, !dbg !76, !tbaa !48, !alias.scope !32, !noalias !35, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !11, !enzymejl_source_type_Ptr\7BFloat64\7D !11, !enzyme_nocache !11
  %25 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %21, i64 0, i32 1, !dbg !76
  %26 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %25, align 8, !dbg !76, !tbaa !48, !alias.scope !32, !noalias !35, !enzyme_type !55, !enzymejl_source_type_Memory\7BFloat64\7D !11, !enzymejl_byref_MUT_REF !11
  %27 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %26, {} addrspace(10)** noundef %24), !dbg !76
  %28 = bitcast {} addrspace(10)* addrspace(13)* %27 to double addrspace(13)*, !dbg !76
  %29 = load double, double addrspace(13)* %28, align 8, !dbg !76, !tbaa !57, !alias.scope !60, !noalias !61, !enzyme_type !62, !enzymejl_byref_BITS_VALUE !11, !enzymejl_source_type_Float64 !11
  %30 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %27, i64 1, !dbg !78
  %31 = bitcast {} addrspace(10)* addrspace(13)* %30 to double addrspace(13)*, !dbg !78
  %32 = load double, double addrspace(13)* %31, align 8, !dbg !78, !tbaa !57, !alias.scope !60, !noalias !61, !enzyme_type !62, !enzymejl_byref_BITS_VALUE !11, !enzymejl_source_type_Float64 !11
  %33 = fmul double %29, %29, !dbg !80
  %34 = fmul double %32, %32, !dbg !80
  %35 = fadd double %33, %34, !dbg !83
  %.not2021 = icmp sgt i64 %7, 2, !dbg !87
  br i1 %.not2021, label %L74, label %common.ret, !dbg !88

L74:                                              ; preds = %L45, %L74
  %value_phi223 = phi i64 [ %36, %L74 ], [ 2, %L45 ]
  %value_phi22 = phi double [ %41, %L74 ], [ %35, %L45 ]
  %36 = add nuw nsw i64 %value_phi223, 1, !dbg !89
  %37 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %27, i64 %value_phi223, !dbg !92
  %38 = bitcast {} addrspace(10)* addrspace(13)* %37 to double addrspace(13)*, !dbg !92
  %39 = load double, double addrspace(13)* %38, align 8, !dbg !92, !tbaa !57, !alias.scope !60, !noalias !61
  %40 = fmul double %39, %39, !dbg !93
  %41 = fadd double %value_phi22, %40, !dbg !96
  %exitcond.not = icmp eq i64 %36, %7, !dbg !87
  br i1 %exitcond.not, label %common.ret, label %L74, !dbg !88

L97:                                              ; preds = %L29
  %42 = call fastcc double @julia_mapreduce_impl_852({} addrspace(10)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) %0, i64 noundef signext 1, i64 noundef signext %7) #7, !dbg !98
  br label %common.ret
}

; Function Attrs: nofree memory(read, argmem: none, inaccessiblemem: none)
define "enzyme_type"="{[-1]:Float@double}" double @julia_sumsq2_826({} addrspace(10)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4730247120" "enzymejl_parmtype_ref"="2" %0) local_unnamed_addr #4 !dbg !101 {
top:
  %pgcstack = call {}*** @julia.get_pgcstack()
  %ptls_field3 = getelementptr inbounds {}**, {}*** %pgcstack, i64 2
  %1 = bitcast {}*** %ptls_field3 to i64***
  %ptls_load45 = load i64**, i64*** %1, align 8, !tbaa !12
  %2 = getelementptr inbounds i64*, i64** %ptls_load45, i64 2
  %safepoint = load i64*, i64** %2, align 8, !tbaa !16
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %safepoint), !dbg !103
  fence syncscope("singlethread") seq_cst
  %3 = call fastcc double @julia__mapreduce_833({} addrspace(10)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) %0) #7, !dbg !104
  ret double %3, !dbg !104
}

; Function Attrs: nounwind
declare void @__enzyme_autodiff(i8*, ...)

define void @test_derivative({} addrspace(10)* %in, {} addrspace(10)* %din) {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double ({} addrspace(10)*)* @julia_sumsq2_826 to i8*), metadata !"enzyme_dup", {} addrspace(10)* %in, {} addrspace(10)* %din)
  ret void
}

; Function Attrs: nofree noinline memory(read, argmem: none, inaccessiblemem: none)
define internal fastcc "enzyme_type"="{[-1]:Float@double}" double @julia_mapreduce_impl_852({} addrspace(10)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,8,0]:Integer, [-1,8,1]:Integer, [-1,8,2]:Integer, [-1,8,3]:Integer, [-1,8,4]:Integer, [-1,8,5]:Integer, [-1,8,6]:Integer, [-1,8,7]:Integer, [-1,8,8]:Pointer, [-1,8,8,-1]:Float@double, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" "enzymejl_parmtype"="4730247120" "enzymejl_parmtype_ref"="2" %0, i64 signext "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" "enzymejl_parmtype"="4784121872" "enzymejl_parmtype_ref"="0" %1, i64 signext "enzyme_inactive" "enzyme_type"="{[-1]:Integer}" "enzymejl_parmtype"="4784121872" "enzymejl_parmtype_ref"="0" %2) unnamed_addr #5 !dbg !119 {
top:
  %pgcstack = call {}*** @julia.get_pgcstack()
  %ptls_field16 = getelementptr inbounds {}**, {}*** %pgcstack, i64 2
  %3 = bitcast {}*** %ptls_field16 to i64***
  %ptls_load1718 = load i64**, i64*** %3, align 8, !tbaa !12
  %4 = getelementptr inbounds i64*, i64** %ptls_load1718, i64 2
  %safepoint = load i64*, i64** %4, align 8, !tbaa !16
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(i64* %safepoint) #6, !dbg !120
  fence syncscope("singlethread") seq_cst
  %.not = icmp eq i64 %2, %1, !dbg !121
  br i1 %.not, label %L17, label %L23, !dbg !124

L17:                                              ; preds = %top
  %5 = bitcast {} addrspace(10)* %0 to { i8*, {} addrspace(10)* } addrspace(10)*, !dbg !125
  %6 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %5 to { i8*, {} addrspace(10)* } addrspace(11)*, !dbg !125
  %7 = bitcast {} addrspace(10)* %0 to {} addrspace(10)** addrspace(10)*, !dbg !125
  %8 = addrspacecast {} addrspace(10)** addrspace(10)* %7 to {} addrspace(10)** addrspace(11)*, !dbg !125
  %9 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %8, align 8, !dbg !125, !tbaa !48, !alias.scope !32, !noalias !35, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !11, !enzymejl_source_type_Ptr\7BFloat64\7D !11, !enzyme_nocache !11
  %10 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %6, i64 0, i32 1, !dbg !125
  %11 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %10, align 8, !dbg !125, !tbaa !48, !alias.scope !32, !noalias !35, !dereferenceable_or_null !53, !align !54, !enzyme_type !55, !enzymejl_source_type_Memory\7BFloat64\7D !11, !enzymejl_byref_MUT_REF !11
  %12 = add i64 %2, -1, !dbg !125
  %13 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %11, {} addrspace(10)** noundef %9), !dbg !125
  %14 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %13, i64 %12, !dbg !125
  %15 = bitcast {} addrspace(10)* addrspace(13)* %14 to double addrspace(13)*, !dbg !125
  %16 = load double, double addrspace(13)* %15, align 8, !dbg !125, !tbaa !57, !alias.scope !60, !noalias !61
  %17 = fmul double %16, %16, !dbg !128
  br label %common.ret

common.ret:                                       ; preds = %L132, %L86, %L40, %L17
  %common.ret.op = phi double [ %17, %L17 ], [ %53, %L132 ], [ %36, %L40 ], [ %47, %L86 ]
  ret double %common.ret.op, !dbg !135

L23:                                              ; preds = %top
  %18 = sub i64 %2, %1, !dbg !136
  %.not19 = icmp slt i64 %18, 1024, !dbg !139
  br i1 %.not19, label %L40, label %L132, !dbg !138

L40:                                              ; preds = %L23
  %19 = bitcast {} addrspace(10)* %0 to { i8*, {} addrspace(10)* } addrspace(10)*, !dbg !141
  %20 = addrspacecast { i8*, {} addrspace(10)* } addrspace(10)* %19 to { i8*, {} addrspace(10)* } addrspace(11)*, !dbg !141
  %21 = bitcast {} addrspace(10)* %0 to {} addrspace(10)** addrspace(10)*, !dbg !141
  %22 = addrspacecast {} addrspace(10)** addrspace(10)* %21 to {} addrspace(10)** addrspace(11)*, !dbg !141
  %23 = load {} addrspace(10)**, {} addrspace(10)** addrspace(11)* %22, align 8, !dbg !141, !tbaa !48, !alias.scope !32, !noalias !35, !enzyme_type !50, !enzymejl_byref_BITS_VALUE !11, !enzymejl_source_type_Ptr\7BFloat64\7D !11, !enzyme_nocache !11
  %24 = getelementptr inbounds { i8*, {} addrspace(10)* }, { i8*, {} addrspace(10)* } addrspace(11)* %20, i64 0, i32 1, !dbg !141
  %25 = load {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %24, align 8, !dbg !141, !tbaa !48, !alias.scope !32, !noalias !35, !enzyme_type !55, !enzymejl_source_type_Memory\7BFloat64\7D !11, !enzymejl_byref_MUT_REF !11
  %26 = add i64 %1, -1, !dbg !141
  %27 = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" {} addrspace(10)* addrspace(13)* @julia.gc_loaded({} addrspace(10)* noundef %25, {} addrspace(10)** noundef %23), !dbg !141
  %28 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %27, i64 %26, !dbg !141
  %29 = bitcast {} addrspace(10)* addrspace(13)* %28 to double addrspace(13)*, !dbg !141
  %30 = load double, double addrspace(13)* %29, align 8, !dbg !141, !tbaa !57, !alias.scope !60, !noalias !61
  %31 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %27, i64 %1, !dbg !143
  %32 = bitcast {} addrspace(10)* addrspace(13)* %31 to double addrspace(13)*, !dbg !143
  %33 = load double, double addrspace(13)* %32, align 8, !dbg !143, !tbaa !57, !alias.scope !60, !noalias !61
  %34 = fmul double %30, %30, !dbg !145
  %35 = fmul double %33, %33, !dbg !145
  %36 = fadd double %34, %35, !dbg !148
  %37 = add i64 %1, 2, !dbg !152
  %.not20 = icmp sgt i64 %37, %2, !dbg !158
  %38 = add i64 %1, 1
  %spec.select = select i1 %.not20, i64 %38, i64 %2, !dbg !163
  %39 = sub i64 %spec.select, %37, !dbg !170
  %40 = icmp ugt i64 %39, 9223372036854775806, !dbg !176
  br i1 %40, label %common.ret, label %L86, !dbg !177

L86:                                              ; preds = %L86, %L40
  %value_phi324 = phi i64 [ %41, %L86 ], [ 0, %L40 ]
  %value_phi223 = phi double [ %47, %L86 ], [ %36, %L40 ]
  %41 = add nuw nsw i64 %value_phi324, 1, !dbg !178
  %42 = add i64 %value_phi324, %38, !dbg !182
  %43 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %27, i64 %42, !dbg !182
  %44 = bitcast {} addrspace(10)* addrspace(13)* %43 to double addrspace(13)*, !dbg !182
  %45 = load double, double addrspace(13)* %44, align 8, !dbg !182, !tbaa !57, !alias.scope !60, !noalias !61
  %46 = fmul double %45, %45, !dbg !186
  %47 = fadd reassoc contract double %value_phi223, %46, !dbg !189
  %exitcond.not = icmp eq i64 %value_phi324, %39, !dbg !191
  br i1 %exitcond.not, label %common.ret, label %L86, !dbg !192, !llvm.loop !193

L132:                                             ; preds = %L23
  %48 = ashr i64 %18, 1, !dbg !194
  %49 = add i64 %48, %1, !dbg !198
  %50 = call fastcc double @julia_mapreduce_impl_852({} addrspace(10)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) %0, i64 signext %1, i64 signext %49) #7, !dbg !199
  %51 = add i64 %49, 1, !dbg !200
  %52 = call fastcc double @julia_mapreduce_impl_852({} addrspace(10)* nocapture nofree noundef nonnull readonly align 8 dereferenceable(24) %0, i64 signext %51, i64 signext %2) #7, !dbg !201
  %53 = fadd double %50, %52, !dbg !202
  br label %common.ret
}

attributes #0 = { nofree memory(none) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzyme_shouldrecompute" "enzymejl_world"="26728" }
attributes #1 = { nofree memory(argmem: readwrite, inaccessiblemem: readwrite) "enzyme_inactive" "enzyme_no_escaping_allocation" "enzymejl_world"="26728" }
attributes #2 = { nofree norecurse nosync nounwind speculatable willreturn memory(none) "enzyme_nocache" "enzyme_shouldrecompute" "enzymejl_world"="26728" }
attributes #3 = { nofree memory(read, argmem: none, inaccessiblemem: none) "enzyme_ta_norecur" "enzymejl_mi"="4539698384" "enzymejl_rt"="4784121072" "enzymejl_world"="26728" }
attributes #4 = { nofree memory(read, argmem: none, inaccessiblemem: none) "enzyme_ta_norecur" "enzymejl_mi"="4538285520" "enzymejl_rt"="4784121072" "enzymejl_world"="26728" }
attributes #5 = { nofree noinline memory(read, argmem: none, inaccessiblemem: none) "enzyme_parmremove"="3" "enzyme_ta_norecur" "enzymejl_mi"="4544092048" "enzymejl_rt"="4784121072" "enzymejl_world"="26728" }
attributes #6 = { memory(readwrite) }
attributes #7 = { nofree }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2, !4, !5, !6, !7}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
!3 = !DIFile(filename: "julia", directory: ".")
!4 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
!5 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
!6 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
!7 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
!8 = distinct !DISubprogram(name: "_mapreduce", linkageName: "julia__mapreduce_833", scope: null, file: !9, line: 425, type: !10, scopeLine: 425, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!9 = !DIFile(filename: "reduce.jl", directory: ".")
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !{!13, !13, i64 0}
!13 = !{!"jtbaa_gcframe", !14, i64 0}
!14 = !{!"jtbaa", !15, i64 0}
!15 = !{!"jtbaa"}
!16 = !{!17, !17, i64 0, i64 0}
!17 = !{!"jtbaa_const", !14, i64 0}
!18 = !DILocation(line: 425, scope: !8)
!19 = !DILocation(line: 194, scope: !20, inlinedAt: !22)
!20 = distinct !DISubprogram(name: "size;", linkageName: "size", scope: !21, file: !21, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!21 = !DIFile(filename: "array.jl", directory: ".")
!22 = !DILocation(line: 98, scope: !23, inlinedAt: !25)
!23 = distinct !DISubprogram(name: "axes;", linkageName: "axes", scope: !24, file: !24, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!24 = !DIFile(filename: "abstractarray.jl", directory: ".")
!25 = !DILocation(line: 494, scope: !26, inlinedAt: !28)
!26 = distinct !DISubprogram(name: "LinearIndices;", linkageName: "LinearIndices", scope: !27, file: !27, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!27 = !DIFile(filename: "indices.jl", directory: ".")
!28 = !DILocation(line: 426, scope: !8)
!29 = !{!30, !30, i64 0}
!30 = !{!"jtbaa_arraysize", !31, i64 0}
!31 = !{!"jtbaa_array", !14, i64 0}
!32 = !{!33}
!33 = !{!"jnoalias_typemd", !34}
!34 = !{!"jnoalias"}
!35 = !{!36, !37, !38, !39}
!36 = !{!"jnoalias_gcframe", !34}
!37 = !{!"jnoalias_stack", !34}
!38 = !{!"jnoalias_data", !34}
!39 = !{!"jnoalias_const", !34}
!40 = !{!"Unknown", i32 -1, !41}
!41 = !{!"Integer"}
!42 = !DILocation(line: 428, scope: !8)
!43 = !DILocation(line: 0, scope: !8)
!44 = !DILocation(line: 917, scope: !45, inlinedAt: !47)
!45 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !46, file: !46, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!46 = !DIFile(filename: "essentials.jl", directory: ".")
!47 = !DILocation(line: 431, scope: !8)
!48 = !{!49, !49, i64 0}
!49 = !{!"jtbaa_arrayptr", !31, i64 0}
!50 = !{!"Unknown", i32 -1, !51}
!51 = !{!"Pointer", i32 -1, !52}
!52 = !{!"Float@double"}
!53 = !{i64 16}
!54 = !{i64 8}
!55 = !{!"Unknown", i32 -1, !56}
!56 = !{!"Pointer", i32 0, !41, i32 1, !41, i32 2, !41, i32 3, !41, i32 4, !41, i32 5, !41, i32 6, !41, i32 7, !41, i32 8, !51}
!57 = !{!58, !58, i64 0}
!58 = !{!"jtbaa_arraybuf", !59, i64 0}
!59 = !{!"jtbaa_data", !14, i64 0}
!60 = !{!38}
!61 = !{!36, !37, !33, !39}
!62 = !{!"Unknown", i32 -1, !52}
!63 = !DILocation(line: 493, scope: !64, inlinedAt: !66)
!64 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !65, file: !65, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!65 = !DIFile(filename: "float.jl", directory: ".")
!66 = !DILocation(line: 189, scope: !67, inlinedAt: !69)
!67 = distinct !DISubprogram(name: "abs2;", linkageName: "abs2", scope: !68, file: !68, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!68 = !DIFile(filename: "number.jl", directory: ".")
!69 = !DILocation(line: 421, scope: !70, inlinedAt: !71)
!70 = distinct !DISubprogram(name: "mapreduce_first;", linkageName: "mapreduce_first", scope: !9, file: !9, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!71 = !DILocation(line: 432, scope: !8)
!72 = !DILocation(line: 83, scope: !73, inlinedAt: !75)
!73 = distinct !DISubprogram(name: "<;", linkageName: "<", scope: !74, file: !74, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!74 = !DIFile(filename: "int.jl", directory: ".")
!75 = !DILocation(line: 433, scope: !8)
!76 = !DILocation(line: 917, scope: !45, inlinedAt: !77)
!77 = !DILocation(line: 435, scope: !8)
!78 = !DILocation(line: 917, scope: !45, inlinedAt: !79)
!79 = !DILocation(line: 436, scope: !8)
!80 = !DILocation(line: 493, scope: !64, inlinedAt: !81)
!81 = !DILocation(line: 189, scope: !67, inlinedAt: !82)
!82 = !DILocation(line: 437, scope: !8)
!83 = !DILocation(line: 491, scope: !84, inlinedAt: !85)
!84 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !65, file: !65, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!85 = !DILocation(line: 27, scope: !86, inlinedAt: !82)
!86 = distinct !DISubprogram(name: "add_sum;", linkageName: "add_sum", scope: !9, file: !9, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!87 = !DILocation(line: 83, scope: !73, inlinedAt: !88)
!88 = !DILocation(line: 438, scope: !8)
!89 = !DILocation(line: 87, scope: !90, inlinedAt: !91)
!90 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !74, file: !74, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!91 = !DILocation(line: 439, scope: !8)
!92 = !DILocation(line: 917, scope: !45, inlinedAt: !91)
!93 = !DILocation(line: 493, scope: !64, inlinedAt: !94)
!94 = !DILocation(line: 189, scope: !67, inlinedAt: !95)
!95 = !DILocation(line: 440, scope: !8)
!96 = !DILocation(line: 491, scope: !84, inlinedAt: !97)
!97 = !DILocation(line: 27, scope: !86, inlinedAt: !95)
!98 = !DILocation(line: 277, scope: !99, inlinedAt: !100)
!99 = distinct !DISubprogram(name: "mapreduce_impl;", linkageName: "mapreduce_impl", scope: !9, file: !9, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, retainedNodes: !11)
!100 = !DILocation(line: 444, scope: !8)
!101 = distinct !DISubprogram(name: "sumsq2", linkageName: "julia_sumsq2_826", scope: null, file: !102, line: 11, type: !10, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !11)
!102 = !DIFile(filename: "/Users/wmoses/git/Enzyme.jl/t2.jl", directory: ".")
!103 = !DILocation(line: 11, scope: !101)
!104 = !DILocation(line: 337, scope: !105, inlinedAt: !107)
!105 = distinct !DISubprogram(name: "_mapreduce_dim;", linkageName: "_mapreduce_dim", scope: !106, file: !106, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !11)
!106 = !DIFile(filename: "reducedim.jl", directory: ".")
!107 = !DILocation(line: 329, scope: !108, inlinedAt: !109)
!108 = distinct !DISubprogram(name: "#mapreduce#926;", linkageName: "#mapreduce#926", scope: !106, file: !106, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !11)
!109 = !DILocation(line: 329, scope: !110, inlinedAt: !111)
!110 = distinct !DISubprogram(name: "mapreduce;", linkageName: "mapreduce", scope: !106, file: !106, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !11)
!111 = !DILocation(line: 987, scope: !112, inlinedAt: !113)
!112 = distinct !DISubprogram(name: "#_sum#936;", linkageName: "#_sum#936", scope: !106, file: !106, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !11)
!113 = !DILocation(line: 987, scope: !114, inlinedAt: !115)
!114 = distinct !DISubprogram(name: "_sum;", linkageName: "_sum", scope: !106, file: !106, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !11)
!115 = !DILocation(line: 983, scope: !116, inlinedAt: !117)
!116 = distinct !DISubprogram(name: "#sum#934;", linkageName: "#sum#934", scope: !106, file: !106, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !11)
!117 = !DILocation(line: 983, scope: !118, inlinedAt: !103)
!118 = distinct !DISubprogram(name: "sum;", linkageName: "sum", scope: !106, file: !106, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !11)
!119 = distinct !DISubprogram(name: "mapreduce_impl", linkageName: "julia_mapreduce_impl_852", scope: null, file: !9, line: 253, type: !10, scopeLine: 253, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!120 = !DILocation(line: 253, scope: !119)
!121 = !DILocation(line: 639, scope: !122, inlinedAt: !124)
!122 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !123, file: !123, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!123 = !DIFile(filename: "promotion.jl", directory: ".")
!124 = !DILocation(line: 255, scope: !119)
!125 = !DILocation(line: 917, scope: !126, inlinedAt: !127)
!126 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !46, file: !46, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!127 = !DILocation(line: 256, scope: !119)
!128 = !DILocation(line: 493, scope: !129, inlinedAt: !130)
!129 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !65, file: !65, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!130 = !DILocation(line: 189, scope: !131, inlinedAt: !132)
!131 = distinct !DISubprogram(name: "abs2;", linkageName: "abs2", scope: !68, file: !68, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!132 = !DILocation(line: 421, scope: !133, inlinedAt: !134)
!133 = distinct !DISubprogram(name: "mapreduce_first;", linkageName: "mapreduce_first", scope: !9, file: !9, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!134 = !DILocation(line: 257, scope: !119)
!135 = !DILocation(line: 0, scope: !119)
!136 = !DILocation(line: 86, scope: !137, inlinedAt: !138)
!137 = distinct !DISubprogram(name: "-;", linkageName: "-", scope: !74, file: !74, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!138 = !DILocation(line: 258, scope: !119)
!139 = !DILocation(line: 83, scope: !140, inlinedAt: !138)
!140 = distinct !DISubprogram(name: "<;", linkageName: "<", scope: !74, file: !74, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!141 = !DILocation(line: 917, scope: !126, inlinedAt: !142)
!142 = !DILocation(line: 260, scope: !119)
!143 = !DILocation(line: 917, scope: !126, inlinedAt: !144)
!144 = !DILocation(line: 261, scope: !119)
!145 = !DILocation(line: 493, scope: !129, inlinedAt: !146)
!146 = !DILocation(line: 189, scope: !131, inlinedAt: !147)
!147 = !DILocation(line: 262, scope: !119)
!148 = !DILocation(line: 491, scope: !149, inlinedAt: !150)
!149 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !65, file: !65, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!150 = !DILocation(line: 27, scope: !151, inlinedAt: !147)
!151 = distinct !DISubprogram(name: "add_sum;", linkageName: "add_sum", scope: !9, file: !9, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!152 = !DILocation(line: 87, scope: !153, inlinedAt: !154)
!153 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !74, file: !74, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!154 = !DILocation(line: 69, scope: !155, inlinedAt: !157)
!155 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !156, file: !156, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!156 = !DIFile(filename: "simdloop.jl", directory: ".")
!157 = !DILocation(line: 263, scope: !119)
!158 = !DILocation(line: 514, scope: !159, inlinedAt: !160)
!159 = distinct !DISubprogram(name: "<=;", linkageName: "<=", scope: !74, file: !74, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!160 = !DILocation(line: 426, scope: !161, inlinedAt: !163)
!161 = distinct !DISubprogram(name: ">=;", linkageName: ">=", scope: !162, file: !162, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!162 = !DIFile(filename: "operators.jl", directory: ".")
!163 = !DILocation(line: 419, scope: !164, inlinedAt: !166)
!164 = distinct !DISubprogram(name: "unitrange_last;", linkageName: "unitrange_last", scope: !165, file: !165, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!165 = !DIFile(filename: "range.jl", directory: ".")
!166 = !DILocation(line: 408, scope: !167, inlinedAt: !168)
!167 = distinct !DISubprogram(name: "UnitRange;", linkageName: "UnitRange", scope: !165, file: !165, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!168 = !DILocation(line: 5, scope: !169, inlinedAt: !154)
!169 = distinct !DISubprogram(name: "Colon;", linkageName: "Colon", scope: !165, file: !165, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!170 = !DILocation(line: 86, scope: !137, inlinedAt: !171)
!171 = !DILocation(line: 768, scope: !172, inlinedAt: !173)
!172 = distinct !DISubprogram(name: "length;", linkageName: "length", scope: !165, file: !165, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!173 = !DILocation(line: 51, scope: !174, inlinedAt: !175)
!174 = distinct !DISubprogram(name: "simd_inner_length;", linkageName: "simd_inner_length", scope: !156, file: !156, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!175 = !DILocation(line: 71, scope: !155, inlinedAt: !157)
!176 = !DILocation(line: 83, scope: !140, inlinedAt: !177)
!177 = !DILocation(line: 72, scope: !155, inlinedAt: !157)
!178 = !DILocation(line: 87, scope: !153, inlinedAt: !179)
!179 = !DILocation(line: 54, scope: !180, inlinedAt: !181)
!180 = distinct !DISubprogram(name: "simd_index;", linkageName: "simd_index", scope: !156, file: !156, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!181 = !DILocation(line: 76, scope: !155, inlinedAt: !157)
!182 = !DILocation(line: 917, scope: !126, inlinedAt: !183)
!183 = !DILocation(line: 264, scope: !184, inlinedAt: !185)
!184 = distinct !DISubprogram(name: "macro expansion;", linkageName: "macro expansion", scope: !9, file: !9, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!185 = !DILocation(line: 77, scope: !155, inlinedAt: !157)
!186 = !DILocation(line: 493, scope: !129, inlinedAt: !187)
!187 = !DILocation(line: 189, scope: !131, inlinedAt: !188)
!188 = !DILocation(line: 265, scope: !184, inlinedAt: !185)
!189 = !DILocation(line: 491, scope: !149, inlinedAt: !190)
!190 = !DILocation(line: 27, scope: !151, inlinedAt: !188)
!191 = !DILocation(line: 83, scope: !140, inlinedAt: !192)
!192 = !DILocation(line: 75, scope: !155, inlinedAt: !157)
!193 = distinct !{!193}
!194 = !DILocation(line: 527, scope: !195, inlinedAt: !196)
!195 = distinct !DISubprogram(name: ">>;", linkageName: ">>", scope: !74, file: !74, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !11)
!196 = !DILocation(line: 534, scope: !195, inlinedAt: !197)
!197 = !DILocation(line: 270, scope: !119)
!198 = !DILocation(line: 87, scope: !153, inlinedAt: !197)
!199 = !DILocation(line: 271, scope: !119)
!200 = !DILocation(line: 87, scope: !153, inlinedAt: !201)
!201 = !DILocation(line: 272, scope: !119)
!202 = !DILocation(line: 491, scope: !149, inlinedAt: !203)
!203 = !DILocation(line: 27, scope: !151, inlinedAt: !204)
!204 = !DILocation(line: 273, scope: !119)
