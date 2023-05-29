;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; ModuleID = 'start'
source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

@_j_const1 = private unnamed_addr constant [2 x i8] c"\01\00"
@_j_str3 = private unnamed_addr constant [11 x i8] c"typeassert\00"

; Function Attrs: nofree readnone
declare {}*** @julia.get_pgcstack() local_unnamed_addr #0

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata %0, metadata %1, metadata %2) #1

; Function Attrs: inaccessiblememonly allocsize(1)
declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** %0, i64 %1, {} addrspace(10)* %2) local_unnamed_addr #2

; Function Attrs: inaccessiblememonly nofree norecurse nounwind
declare void @julia.write_barrier({} addrspace(10)* readonly %0, ...) local_unnamed_addr #3

; Function Attrs: nofree
declare nonnull {} addrspace(10)* @ijl_invoke({} addrspace(10)* %0, {} addrspace(10)** nocapture readonly %1, i32 %2, {} addrspace(10)* %3) #4

declare nonnull {} addrspace(10)* @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* %0, {} addrspace(10)* %1, {} addrspace(10)* %2, ...) local_unnamed_addr #5

; Function Attrs: noreturn
declare void @ijl_throw({} addrspace(12)* %0) local_unnamed_addr #6

; Function Attrs: nofree norecurse nounwind readnone
declare nonnull {} addrspace(10)* @julia.typeof({} addrspace(10)* %0) local_unnamed_addr #7

; Function Attrs: noreturn
declare void @ijl_type_error(i8* %0, {} addrspace(10)* %1, {} addrspace(12)* %2) local_unnamed_addr #6

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg %0, i8* nocapture %1) #8

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg %0, i8* nocapture %1) #8

; Function Attrs: nofree nounwind readnone
declare nonnull {}* @julia.pointer_from_objref({} addrspace(11)* %0) local_unnamed_addr #9

; Function Attrs: nofree
declare nonnull {} addrspace(10)* @ijl_box_uint32(i32 zeroext %0) local_unnamed_addr #10

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.ctlz.i32(i32 %0, i1 immarg %1) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.cttz.i32(i32 %0, i1 immarg %1) #1

; Function Attrs: inaccessiblememonly
declare noalias {} addrspace(10)* @ijl_alloc_array_2d({} addrspace(10)* %0, i64 %1, i64 %2) local_unnamed_addr #11

; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn
declare void @dgemm_64_(i8* %0, i8* %1, i8* nocapture readonly %2, i8* nocapture readonly %3, i8* nocapture readonly %4, i8* nocapture readonly %5, i64 %6, i8* %7, i64 %8, i8* %9, i8* nocapture readonly %10, i64 %11, i8* %12, i64 %13, i64 %14) local_unnamed_addr #12

declare void @__enzyme_autodiff(...)

define void @caller() {
entry:
  call void (...) @__enzyme_autodiff(void ({} addrspace(10)*, {} addrspace(10)*)* @julia_foo_801_inner.1, metadata !"enzyme_dup", i8* null, i8* null, metadata !"enzyme_dup", i8* null, i8* null)
  ret void
}

define void @julia_foo_801_inner.1({} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %0, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %1) local_unnamed_addr #13 !dbg !41 {
entry:
  %2 = call {}*** @julia.get_pgcstack()
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !54, metadata !DIExpression(DW_OP_deref)), !dbg !55
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !53, metadata !DIExpression(DW_OP_deref)), !dbg !55
  call void @llvm.dbg.value(metadata {} addrspace(10)* %0, metadata !53, metadata !DIExpression(DW_OP_deref)), !dbg !55
  call void @llvm.dbg.value(metadata {} addrspace(10)* %1, metadata !54, metadata !DIExpression(DW_OP_deref)), !dbg !55
  %3 = bitcast {} addrspace(10)* %0 to {} addrspace(10)* addrspace(10)*, !dbg !57
  %4 = addrspacecast {} addrspace(10)* addrspace(10)* %3 to {} addrspace(10)* addrspace(11)*, !dbg !57
  %5 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %4, i64 3, !dbg !57
  %6 = bitcast {} addrspace(10)* addrspace(11)* %5 to i64 addrspace(11)*, !dbg !57
  %7 = load i64, i64 addrspace(11)* %6, align 8, !dbg !57, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %8 = bitcast {} addrspace(10)* %1 to {} addrspace(10)* addrspace(10)*, !dbg !57
  %9 = addrspacecast {} addrspace(10)* addrspace(10)* %8 to {} addrspace(10)* addrspace(11)*, !dbg !57
  %10 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %9, i64 4, !dbg !57
  %11 = bitcast {} addrspace(10)* addrspace(11)* %10 to i64 addrspace(11)*, !dbg !57
  %12 = load i64, i64 addrspace(11)* %11, align 16, !dbg !57, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %13 = call noalias nonnull {} addrspace(10)* @ijl_alloc_array_2d({} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140733404252304 to {}*) to {} addrspace(10)*), i64 %7, i64 %12) #11, !dbg !76
  call fastcc void @julia_gemm_wrapper__804({} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %13, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %0, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %1) #26, !dbg !81
  ret void, !dbg !84
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata %0, metadata %1, metadata %2) #14

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p13i8.i64(i8 addrspace(13)* nocapture writeonly %0, i8 %1, i64 %2, i1 immarg %3) #15

; Function Attrs: nofree nosync readnone
define internal fastcc [1 x {} addrspace(10)*] @julia_ArgumentError_864() unnamed_addr #16 !dbg !85 {
top:
  %0 = call {}*** @julia.get_pgcstack()
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !98, metadata !DIExpression(DW_OP_deref)), !dbg !99
  call void @llvm.dbg.value(metadata {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772048 to {}*) to {} addrspace(10)*), metadata !98, metadata !DIExpression(DW_OP_deref)), !dbg !99
  ret [1 x {} addrspace(10)*] undef, !dbg !100
}

define internal fastcc void @julia_matmul2x2__878({} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %0, {} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %1, {} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %2) unnamed_addr #17 !dbg !101 {
top:
  %3 = call {}*** @julia.get_pgcstack()
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !115, metadata !DIExpression(DW_OP_deref)), !dbg !117
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !114, metadata !DIExpression(DW_OP_deref)), !dbg !117
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !111, metadata !DIExpression(DW_OP_deref)), !dbg !117
  call void @llvm.dbg.value(metadata {} addrspace(10)* %0, metadata !111, metadata !DIExpression(DW_OP_deref)), !dbg !117
  call void @llvm.dbg.value(metadata i32 1308622848, metadata !112, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata i32 1308622848, metadata !113, metadata !DIExpression()), !dbg !117
  call void @llvm.dbg.value(metadata {} addrspace(10)* %1, metadata !114, metadata !DIExpression(DW_OP_deref)), !dbg !117
  call void @llvm.dbg.value(metadata {} addrspace(10)* %2, metadata !115, metadata !DIExpression(DW_OP_deref)), !dbg !117
  call void @llvm.dbg.declare(metadata [2 x i8] addrspace(11)* addrspacecast ([2 x i8]* @_j_const1 to [2 x i8] addrspace(11)*), metadata !116, metadata !DIExpression(DW_OP_deref)), !dbg !118
  %4 = bitcast {} addrspace(10)* %2 to {} addrspace(10)* addrspace(10)*, !dbg !119
  %5 = addrspacecast {} addrspace(10)* addrspace(10)* %4 to {} addrspace(10)* addrspace(11)*, !dbg !119
  %6 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %5, i64 3, !dbg !119
  %7 = bitcast {} addrspace(10)* addrspace(11)* %6 to i64 addrspace(11)*, !dbg !119
  %8 = load i64, i64 addrspace(11)* %7, align 8, !dbg !119, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %9 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %5, i64 4, !dbg !119
  %10 = bitcast {} addrspace(10)* addrspace(11)* %9 to i64 addrspace(11)*, !dbg !119
  %11 = load i64, i64 addrspace(11)* %10, align 16, !dbg !119, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %12 = bitcast {} addrspace(10)* %1 to {} addrspace(10)* addrspace(10)*, !dbg !119
  %13 = addrspacecast {} addrspace(10)* addrspace(10)* %12 to {} addrspace(10)* addrspace(11)*, !dbg !119
  %14 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %13, i64 3, !dbg !119
  %15 = bitcast {} addrspace(10)* addrspace(11)* %14 to i64 addrspace(11)*, !dbg !119
  %16 = load i64, i64 addrspace(11)* %15, align 8, !dbg !119, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %17 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %13, i64 4, !dbg !119
  %18 = bitcast {} addrspace(10)* addrspace(11)* %17 to i64 addrspace(11)*, !dbg !119
  %19 = load i64, i64 addrspace(11)* %18, align 16, !dbg !119, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.not = icmp ne i64 %16, %8, !dbg !122
  %20 = icmp ne i64 %19, %11
  %spec.select = select i1 %.not, i1 true, i1 %20, !dbg !130
  br i1 %spec.select, label %top.L56_crit_edge, label %L19, !dbg !121

top.L56_crit_edge:                                ; preds = %top
  %.phi.trans.insert = bitcast {} addrspace(10)* %0 to {} addrspace(10)* addrspace(10)*
  %.phi.trans.insert97 = addrspacecast {} addrspace(10)* addrspace(10)* %.phi.trans.insert to {} addrspace(10)* addrspace(11)*
  %.phi.trans.insert98 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %.phi.trans.insert97, i64 3
  %.phi.trans.insert99 = bitcast {} addrspace(10)* addrspace(11)* %.phi.trans.insert98 to i64 addrspace(11)*
  %.pre = load i64, i64 addrspace(11)* %.phi.trans.insert99, align 8, !dbg !131, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.phi.trans.insert102 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %.phi.trans.insert97, i64 4
  %.phi.trans.insert103 = bitcast {} addrspace(10)* addrspace(11)* %.phi.trans.insert102 to i64 addrspace(11)*
  %.pre104 = load i64, i64 addrspace(11)* %.phi.trans.insert103, align 16, !dbg !131, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  br label %L56, !dbg !121

L19:                                              ; preds = %top
  %21 = bitcast {} addrspace(10)* %0 to {} addrspace(10)* addrspace(10)*, !dbg !119
  %22 = addrspacecast {} addrspace(10)* addrspace(10)* %21 to {} addrspace(10)* addrspace(11)*, !dbg !119
  %23 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %22, i64 3, !dbg !119
  %24 = bitcast {} addrspace(10)* addrspace(11)* %23 to i64 addrspace(11)*, !dbg !119
  %25 = load i64, i64 addrspace(11)* %24, align 8, !dbg !119, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %26 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %22, i64 4, !dbg !119
  %27 = bitcast {} addrspace(10)* addrspace(11)* %26 to i64 addrspace(11)*, !dbg !119
  %28 = load i64, i64 addrspace(11)* %27, align 16, !dbg !119, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.not69 = icmp eq i64 %8, %25, !dbg !122
  %29 = icmp eq i64 %11, %28
  %spec.select94 = select i1 %.not69, i1 %29, i1 false, !dbg !130
  %.not70 = icmp eq i64 %25, 2
  %or.cond = and i1 %.not70, %spec.select94, !dbg !121
  %.not93 = icmp eq i64 %28, 2
  %or.cond96 = select i1 %or.cond, i1 %.not93, i1 false, !dbg !121
  br i1 %or.cond96, label %L91, label %L56, !dbg !121

L56:                                              ; preds = %L19, %top.L56_crit_edge
  %30 = phi i64 [ %.pre104, %top.L56_crit_edge ], [ %28, %L19 ], !dbg !131
  %31 = phi i64 [ %.pre, %top.L56_crit_edge ], [ %25, %L19 ], !dbg !131
  %current_task571 = getelementptr inbounds {}**, {}*** %3, i64 -13, !dbg !133
  %current_task5 = bitcast {}*** %current_task571 to {}**, !dbg !133
  %32 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task5, i64 noundef 16, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737038955296 to {}*) to {} addrspace(10)*)) #27, !dbg !133
  %33 = bitcast {} addrspace(10)* %32 to {} addrspace(10)* addrspace(10)*, !dbg !133
  %34 = addrspacecast {} addrspace(10)* addrspace(10)* %33 to {} addrspace(10)* addrspace(11)*, !dbg !133
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %34, align 8, !dbg !133, !tbaa !136, !alias.scope !140, !noalias !141
  %35 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %34, i64 1, !dbg !133
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %35, align 8, !dbg !133, !tbaa !136, !alias.scope !140, !noalias !141
  %36 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task5, i64 noundef 72, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140733407852880 to {}*) to {} addrspace(10)*)) #27, !dbg !133
  %37 = bitcast {} addrspace(10)* %36 to { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)*, !dbg !133
  %.repack = bitcast {} addrspace(10)* %36 to {} addrspace(10)* addrspace(10)*, !dbg !133
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048917232 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  %.repack72.repack = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 1, i64 0, !dbg !133
  store i64 %16, i64 addrspace(10)* %.repack72.repack, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  %.repack72.repack82 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 1, i64 1, !dbg !133
  store i64 %19, i64 addrspace(10)* %.repack72.repack82, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  %.repack74 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 2, !dbg !133
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048917200 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack74, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  %.repack76.repack = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 3, i64 0, !dbg !133
  store i64 %8, i64 addrspace(10)* %.repack76.repack, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  %.repack76.repack84 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 3, i64 1, !dbg !133
  store i64 %11, i64 addrspace(10)* %.repack76.repack84, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  %.repack78 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 4, !dbg !133
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048917168 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack78, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  %.repack80.repack = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 5, i64 0, !dbg !133
  store i64 %31, i64 addrspace(10)* %.repack80.repack, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  %.repack80.repack86 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 5, i64 1, !dbg !133
  store i64 %30, i64 addrspace(10)* %.repack80.repack86, align 8, !dbg !133, !tbaa !144, !alias.scope !140, !noalias !141
  store atomic {} addrspace(10)* %36, {} addrspace(10)* addrspace(11)* %34 release, align 8, !dbg !133, !tbaa !136, !alias.scope !140, !noalias !141
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %32, {} addrspace(10)* nonnull %36) #28, !dbg !133
  %38 = bitcast {} addrspace(10)* %32 to i8 addrspace(10)*, !dbg !133
  %39 = addrspacecast i8 addrspace(10)* %38 to i8 addrspace(11)*, !dbg !133
  %40 = getelementptr inbounds i8, i8 addrspace(11)* %39, i64 8, !dbg !133
  %41 = bitcast i8 addrspace(11)* %40 to {} addrspace(10)* addrspace(11)*, !dbg !133
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(11)* %41 release, align 8, !dbg !133, !tbaa !136, !alias.scope !140, !noalias !141
  %42 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %41 acquire, align 8, !dbg !146, !tbaa !136, !alias.scope !140, !noalias !156, !nonnull !50
  %43 = addrspacecast {} addrspace(10)* %42 to {} addrspace(11)*, !dbg !157
  %.not88 = icmp eq {} addrspace(11)* %43, addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(11)*), !dbg !157
  br i1 %.not88, label %L70, label %L87, !dbg !157

L70:                                              ; preds = %L56
  %44 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task5, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737021929984 to {}*) to {} addrspace(10)*)) #27, !dbg !158
  %45 = bitcast {} addrspace(10)* %44 to {} addrspace(10)* addrspace(10)*, !dbg !158
  store {} addrspace(10)* %32, {} addrspace(10)* addrspace(10)* %45, align 8, !dbg !158, !tbaa !144, !alias.scope !140, !noalias !141
  %46 = call nonnull {} addrspace(10)* ({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* noundef nonnull @ijl_invoke, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737062426592 to {}*) to {} addrspace(10)*), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140736998151488 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226350624 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140736999550464 to {}*) to {} addrspace(10)*), {} addrspace(10)* nonnull %44) #29, !dbg !158
  %47 = cmpxchg {} addrspace(10)* addrspace(11)* %41, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* %46 acq_rel acquire, align 8, !dbg !162, !tbaa !136, !alias.scope !140, !noalias !156
  %48 = extractvalue { {} addrspace(10)*, i1 } %47, 0, !dbg !162
  %49 = extractvalue { {} addrspace(10)*, i1 } %47, 1, !dbg !162
  br i1 %49, label %xchg_wb, label %L80, !dbg !162

L80:                                              ; preds = %L70
  %50 = call {} addrspace(10)* @julia.typeof({} addrspace(10)* %48) #30, !dbg !165
  %51 = icmp eq {} addrspace(10)* %50, addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), !dbg !165
  br i1 %51, label %L87, label %fail, !dbg !165

L87:                                              ; preds = %xchg_wb, %L80, %L56
  %value_phi8 = phi {} addrspace(10)* [ %46, %xchg_wb ], [ %42, %L56 ], [ %48, %L80 ]
  %52 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task5, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737046502288 to {}*) to {} addrspace(10)*)) #27, !dbg !132
  %53 = bitcast {} addrspace(10)* %52 to {} addrspace(10)* addrspace(10)*, !dbg !132
  store {} addrspace(10)* %value_phi8, {} addrspace(10)* addrspace(10)* %53, align 8, !dbg !132, !tbaa !144, !alias.scope !140, !noalias !141
  %54 = addrspacecast {} addrspace(10)* %52 to {} addrspace(12)*, !dbg !132
  call void @ijl_throw({} addrspace(12)* %54) #31, !dbg !132
  unreachable, !dbg !132

L91:                                              ; preds = %L19
  switch i32 1308622848, label %L109 [
    i32 1409286144, label %L95
    i32 1124073472, label %L104
  ], !dbg !166

L95:                                              ; preds = %L91
  unreachable

L104:                                             ; preds = %L91
  unreachable

L109:                                             ; preds = %L91
  %55 = bitcast {} addrspace(10)* %1 to double addrspace(13)* addrspace(10)*, !dbg !167
  %56 = addrspacecast double addrspace(13)* addrspace(10)* %55 to double addrspace(13)* addrspace(11)*, !dbg !167
  %57 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %56, align 16, !dbg !167, !tbaa !63, !invariant.load !50, !alias.scope !171, !noalias !71, !nonnull !50
  br label %L113, !dbg !167

L113:                                             ; preds = %L109
  %58 = add nuw nsw i64 %8, 1, !dbg !167
  %59 = getelementptr inbounds double, double addrspace(13)* %57, i64 %58, !dbg !167
  %60 = getelementptr inbounds double, double addrspace(13)* %57, i64 1, !dbg !167
  %61 = getelementptr inbounds double, double addrspace(13)* %57, i64 %8, !dbg !167
  %value_phi13 = load double, double addrspace(13)* %57, align 8, !dbg !117, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi12 = load double, double addrspace(13)* %61, align 8, !dbg !117, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi11 = load double, double addrspace(13)* %60, align 8, !dbg !117, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi10 = load double, double addrspace(13)* %59, align 8, !dbg !117, !tbaa !172, !alias.scope !140, !noalias !156
  switch i32 1308622848, label %L135 [
    i32 1409286144, label %L121
    i32 1124073472, label %L130
  ], !dbg !174

L121:                                             ; preds = %L113
  unreachable

L130:                                             ; preds = %L113
  unreachable

L135:                                             ; preds = %L113
  %62 = bitcast {} addrspace(10)* %2 to double addrspace(13)* addrspace(10)*, !dbg !175
  %63 = addrspacecast double addrspace(13)* addrspace(10)* %62 to double addrspace(13)* addrspace(11)*, !dbg !175
  %64 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %63, align 16, !dbg !175, !tbaa !63, !invariant.load !50, !alias.scope !171, !noalias !71, !nonnull !50
  %65 = add nuw nsw i64 %8, 1, !dbg !177
  br label %L139, !dbg !177

L139:                                             ; preds = %L135
  %66 = getelementptr inbounds double, double addrspace(13)* %64, i64 %65, !dbg !177
  %67 = getelementptr inbounds double, double addrspace(13)* %64, i64 1, !dbg !177
  %68 = getelementptr inbounds double, double addrspace(13)* %64, i64 %8, !dbg !175
  %value_phi17 = load double, double addrspace(13)* %64, align 8, !dbg !117, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi16 = load double, double addrspace(13)* %68, align 8, !dbg !117, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi15 = load double, double addrspace(13)* %67, align 8, !dbg !117, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi14 = load double, double addrspace(13)* %66, align 8, !dbg !117, !tbaa !172, !alias.scope !140, !noalias !156
  %69 = fmul double %value_phi13, %value_phi17, !dbg !179
  %70 = fmul double %value_phi12, %value_phi15, !dbg !179
  %71 = fadd double %69, %70, !dbg !183
  %72 = bitcast {} addrspace(10)* %0 to double addrspace(13)* addrspace(10)*, !dbg !185
  %73 = addrspacecast double addrspace(13)* addrspace(10)* %72 to double addrspace(13)* addrspace(11)*, !dbg !185
  %74 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %73, align 8, !dbg !185, !tbaa !63, !invariant.load !50, !alias.scope !171, !noalias !71, !nonnull !50
  store double %71, double addrspace(13)* %74, align 8, !dbg !185, !tbaa !172, !alias.scope !140, !noalias !141
  %75 = fmul double %value_phi13, %value_phi16, !dbg !192
  %76 = fmul double %value_phi12, %value_phi14, !dbg !192
  %77 = fadd double %75, %76, !dbg !194
  %78 = getelementptr inbounds double, double addrspace(13)* %74, i64 %8, !dbg !195
  store double %77, double addrspace(13)* %78, align 8, !dbg !195, !tbaa !172, !alias.scope !140, !noalias !141
  %79 = fmul double %value_phi11, %value_phi17, !dbg !198
  %80 = fmul double %value_phi10, %value_phi15, !dbg !198
  %81 = fadd double %79, %80, !dbg !200
  %82 = getelementptr inbounds double, double addrspace(13)* %74, i64 1, !dbg !201
  store double %81, double addrspace(13)* %82, align 8, !dbg !201, !tbaa !172, !alias.scope !140, !noalias !141
  %83 = fmul double %value_phi11, %value_phi16, !dbg !204
  %84 = fmul double %value_phi10, %value_phi14, !dbg !204
  %85 = fadd double %83, %84, !dbg !206
  %86 = getelementptr inbounds double, double addrspace(13)* %74, i64 %65, !dbg !207
  store double %85, double addrspace(13)* %86, align 8, !dbg !207, !tbaa !172, !alias.scope !140, !noalias !141
  ret void, !dbg !210

xchg_wb:                                          ; preds = %L70
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %32, {} addrspace(10)* nonnull %46) #28, !dbg !162
  br label %L87, !dbg !165

fail:                                             ; preds = %L80
  %87 = addrspacecast {} addrspace(10)* %48 to {} addrspace(12)*, !dbg !165
  call void @ijl_type_error(i8* noundef getelementptr inbounds ([11 x i8], [11 x i8]* @_j_str3, i64 0, i64 0), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), {} addrspace(12)* %87) #31, !dbg !165
  unreachable, !dbg !165
}

define internal fastcc void @julia_matmul3x3__876({} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %0, {} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %1, {} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %2) unnamed_addr #18 !dbg !211 {
top:
  %3 = call {}*** @julia.get_pgcstack()
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !221, metadata !DIExpression(DW_OP_deref)), !dbg !223
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !220, metadata !DIExpression(DW_OP_deref)), !dbg !223
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !217, metadata !DIExpression(DW_OP_deref)), !dbg !223
  call void @llvm.dbg.value(metadata {} addrspace(10)* %0, metadata !217, metadata !DIExpression(DW_OP_deref)), !dbg !223
  call void @llvm.dbg.value(metadata i32 1308622848, metadata !218, metadata !DIExpression()), !dbg !223
  call void @llvm.dbg.value(metadata i32 1308622848, metadata !219, metadata !DIExpression()), !dbg !223
  call void @llvm.dbg.value(metadata {} addrspace(10)* %1, metadata !220, metadata !DIExpression(DW_OP_deref)), !dbg !223
  call void @llvm.dbg.value(metadata {} addrspace(10)* %2, metadata !221, metadata !DIExpression(DW_OP_deref)), !dbg !223
  call void @llvm.dbg.declare(metadata [2 x i8] addrspace(11)* addrspacecast ([2 x i8]* @_j_const1 to [2 x i8] addrspace(11)*), metadata !222, metadata !DIExpression(DW_OP_deref)), !dbg !224
  %4 = bitcast {} addrspace(10)* %2 to {} addrspace(10)* addrspace(10)*, !dbg !225
  %5 = addrspacecast {} addrspace(10)* addrspace(10)* %4 to {} addrspace(10)* addrspace(11)*, !dbg !225
  %6 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %5, i64 3, !dbg !225
  %7 = bitcast {} addrspace(10)* addrspace(11)* %6 to i64 addrspace(11)*, !dbg !225
  %8 = load i64, i64 addrspace(11)* %7, align 8, !dbg !225, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %9 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %5, i64 4, !dbg !225
  %10 = bitcast {} addrspace(10)* addrspace(11)* %9 to i64 addrspace(11)*, !dbg !225
  %11 = load i64, i64 addrspace(11)* %10, align 16, !dbg !225, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %12 = bitcast {} addrspace(10)* %1 to {} addrspace(10)* addrspace(10)*, !dbg !225
  %13 = addrspacecast {} addrspace(10)* addrspace(10)* %12 to {} addrspace(10)* addrspace(11)*, !dbg !225
  %14 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %13, i64 3, !dbg !225
  %15 = bitcast {} addrspace(10)* addrspace(11)* %14 to i64 addrspace(11)*, !dbg !225
  %16 = load i64, i64 addrspace(11)* %15, align 8, !dbg !225, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %17 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %13, i64 4, !dbg !225
  %18 = bitcast {} addrspace(10)* addrspace(11)* %17 to i64 addrspace(11)*, !dbg !225
  %19 = load i64, i64 addrspace(11)* %18, align 16, !dbg !225, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.not = icmp ne i64 %16, %8, !dbg !228
  %20 = icmp ne i64 %19, %11
  %spec.select = select i1 %.not, i1 true, i1 %20, !dbg !234
  br i1 %spec.select, label %top.L56_crit_edge, label %L19, !dbg !227

top.L56_crit_edge:                                ; preds = %top
  %.phi.trans.insert = bitcast {} addrspace(10)* %0 to {} addrspace(10)* addrspace(10)*
  %.phi.trans.insert142 = addrspacecast {} addrspace(10)* addrspace(10)* %.phi.trans.insert to {} addrspace(10)* addrspace(11)*
  %.phi.trans.insert143 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %.phi.trans.insert142, i64 3
  %.phi.trans.insert144 = bitcast {} addrspace(10)* addrspace(11)* %.phi.trans.insert143 to i64 addrspace(11)*
  %.pre = load i64, i64 addrspace(11)* %.phi.trans.insert144, align 8, !dbg !235, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.phi.trans.insert147 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %.phi.trans.insert142, i64 4
  %.phi.trans.insert148 = bitcast {} addrspace(10)* addrspace(11)* %.phi.trans.insert147 to i64 addrspace(11)*
  %.pre149 = load i64, i64 addrspace(11)* %.phi.trans.insert148, align 16, !dbg !235, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  br label %L56, !dbg !227

L19:                                              ; preds = %top
  %21 = bitcast {} addrspace(10)* %0 to {} addrspace(10)* addrspace(10)*, !dbg !225
  %22 = addrspacecast {} addrspace(10)* addrspace(10)* %21 to {} addrspace(10)* addrspace(11)*, !dbg !225
  %23 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %22, i64 3, !dbg !225
  %24 = bitcast {} addrspace(10)* addrspace(11)* %23 to i64 addrspace(11)*, !dbg !225
  %25 = load i64, i64 addrspace(11)* %24, align 8, !dbg !225, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %26 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %22, i64 4, !dbg !225
  %27 = bitcast {} addrspace(10)* addrspace(11)* %26 to i64 addrspace(11)*, !dbg !225
  %28 = load i64, i64 addrspace(11)* %27, align 16, !dbg !225, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.not114 = icmp eq i64 %8, %25, !dbg !228
  %29 = icmp eq i64 %11, %28
  %spec.select139 = select i1 %.not114, i1 %29, i1 false, !dbg !234
  %.not115 = icmp eq i64 %25, 3
  %or.cond = and i1 %.not115, %spec.select139, !dbg !227
  %.not138 = icmp eq i64 %28, 3
  %or.cond141 = select i1 %or.cond, i1 %.not138, i1 false, !dbg !227
  br i1 %or.cond141, label %L91, label %L56, !dbg !227

L56:                                              ; preds = %L19, %top.L56_crit_edge
  %30 = phi i64 [ %.pre149, %top.L56_crit_edge ], [ %28, %L19 ], !dbg !235
  %31 = phi i64 [ %.pre, %top.L56_crit_edge ], [ %25, %L19 ], !dbg !235
  %current_task5116 = getelementptr inbounds {}**, {}*** %3, i64 -13, !dbg !237
  %current_task5 = bitcast {}*** %current_task5116 to {}**, !dbg !237
  %32 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task5, i64 noundef 16, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737038955296 to {}*) to {} addrspace(10)*)) #27, !dbg !237
  %33 = bitcast {} addrspace(10)* %32 to {} addrspace(10)* addrspace(10)*, !dbg !237
  %34 = addrspacecast {} addrspace(10)* addrspace(10)* %33 to {} addrspace(10)* addrspace(11)*, !dbg !237
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %34, align 8, !dbg !237, !tbaa !136, !alias.scope !140, !noalias !239
  %35 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %34, i64 1, !dbg !237
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %35, align 8, !dbg !237, !tbaa !136, !alias.scope !140, !noalias !239
  %36 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task5, i64 noundef 72, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140733407852880 to {}*) to {} addrspace(10)*)) #27, !dbg !237
  %37 = bitcast {} addrspace(10)* %36 to { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)*, !dbg !237
  %.repack = bitcast {} addrspace(10)* %36 to {} addrspace(10)* addrspace(10)*, !dbg !237
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737050799904 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  %.repack117.repack = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 1, i64 0, !dbg !237
  store i64 %16, i64 addrspace(10)* %.repack117.repack, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  %.repack117.repack127 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 1, i64 1, !dbg !237
  store i64 %19, i64 addrspace(10)* %.repack117.repack127, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  %.repack119 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 2, !dbg !237
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737050799872 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack119, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  %.repack121.repack = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 3, i64 0, !dbg !237
  store i64 %8, i64 addrspace(10)* %.repack121.repack, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  %.repack121.repack129 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 3, i64 1, !dbg !237
  store i64 %11, i64 addrspace(10)* %.repack121.repack129, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  %.repack123 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 4, !dbg !237
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737050799840 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack123, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  %.repack125.repack = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 5, i64 0, !dbg !237
  store i64 %31, i64 addrspace(10)* %.repack125.repack, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  %.repack125.repack131 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64], {} addrspace(10)*, [2 x i64] } addrspace(10)* %37, i64 0, i32 5, i64 1, !dbg !237
  store i64 %30, i64 addrspace(10)* %.repack125.repack131, align 8, !dbg !237, !tbaa !144, !alias.scope !140, !noalias !239
  store atomic {} addrspace(10)* %36, {} addrspace(10)* addrspace(11)* %34 release, align 8, !dbg !237, !tbaa !136, !alias.scope !140, !noalias !239
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %32, {} addrspace(10)* nonnull %36) #28, !dbg !237
  %38 = bitcast {} addrspace(10)* %32 to i8 addrspace(10)*, !dbg !237
  %39 = addrspacecast i8 addrspace(10)* %38 to i8 addrspace(11)*, !dbg !237
  %40 = getelementptr inbounds i8, i8 addrspace(11)* %39, i64 8, !dbg !237
  %41 = bitcast i8 addrspace(11)* %40 to {} addrspace(10)* addrspace(11)*, !dbg !237
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(11)* %41 release, align 8, !dbg !237, !tbaa !136, !alias.scope !140, !noalias !239
  %42 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %41 acquire, align 8, !dbg !242, !tbaa !136, !alias.scope !140, !noalias !156, !nonnull !50
  %43 = addrspacecast {} addrspace(10)* %42 to {} addrspace(11)*, !dbg !250
  %.not133 = icmp eq {} addrspace(11)* %43, addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(11)*), !dbg !250
  br i1 %.not133, label %L70, label %L87, !dbg !250

L70:                                              ; preds = %L56
  %44 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task5, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737021929984 to {}*) to {} addrspace(10)*)) #27, !dbg !251
  %45 = bitcast {} addrspace(10)* %44 to {} addrspace(10)* addrspace(10)*, !dbg !251
  store {} addrspace(10)* %32, {} addrspace(10)* addrspace(10)* %45, align 8, !dbg !251, !tbaa !144, !alias.scope !140, !noalias !239
  %46 = call nonnull {} addrspace(10)* ({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* noundef nonnull @ijl_invoke, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737062426592 to {}*) to {} addrspace(10)*), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140736998151488 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226350624 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140736999550464 to {}*) to {} addrspace(10)*), {} addrspace(10)* nonnull %44) #29, !dbg !251
  %47 = cmpxchg {} addrspace(10)* addrspace(11)* %41, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* %46 acq_rel acquire, align 8, !dbg !254, !tbaa !136, !alias.scope !140, !noalias !156
  %48 = extractvalue { {} addrspace(10)*, i1 } %47, 0, !dbg !254
  %49 = extractvalue { {} addrspace(10)*, i1 } %47, 1, !dbg !254
  br i1 %49, label %xchg_wb, label %L80, !dbg !254

L80:                                              ; preds = %L70
  %50 = call {} addrspace(10)* @julia.typeof({} addrspace(10)* %48) #30, !dbg !257
  %51 = icmp eq {} addrspace(10)* %50, addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), !dbg !257
  br i1 %51, label %L87, label %fail, !dbg !257

L87:                                              ; preds = %xchg_wb, %L80, %L56
  %value_phi8 = phi {} addrspace(10)* [ %46, %xchg_wb ], [ %42, %L56 ], [ %48, %L80 ]
  %52 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task5, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737046502288 to {}*) to {} addrspace(10)*)) #27, !dbg !236
  %53 = bitcast {} addrspace(10)* %52 to {} addrspace(10)* addrspace(10)*, !dbg !236
  store {} addrspace(10)* %value_phi8, {} addrspace(10)* addrspace(10)* %53, align 8, !dbg !236, !tbaa !144, !alias.scope !140, !noalias !239
  %54 = addrspacecast {} addrspace(10)* %52 to {} addrspace(12)*, !dbg !236
  call void @ijl_throw({} addrspace(12)* %54) #31, !dbg !236
  unreachable, !dbg !236

L91:                                              ; preds = %L19
  switch i32 1308622848, label %L119 [
    i32 1409286144, label %L95
    i32 1124073472, label %L109
  ], !dbg !258

L95:                                              ; preds = %L91
  unreachable

L109:                                             ; preds = %L91
  unreachable

L119:                                             ; preds = %L91
  %55 = bitcast {} addrspace(10)* %1 to double addrspace(13)* addrspace(10)*, !dbg !259
  %56 = addrspacecast double addrspace(13)* addrspace(10)* %55 to double addrspace(13)* addrspace(11)*, !dbg !259
  %57 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %56, align 16, !dbg !259, !tbaa !63, !invariant.load !50, !alias.scope !262, !noalias !71, !nonnull !50
  %58 = shl nuw i64 %8, 1, !dbg !259
  br label %L128, !dbg !263

L128:                                             ; preds = %L119
  %59 = add nuw i64 %58, 2, !dbg !263
  %60 = getelementptr inbounds double, double addrspace(13)* %57, i64 %59, !dbg !263
  %61 = add nuw i64 %8, 2, !dbg !263
  %62 = getelementptr inbounds double, double addrspace(13)* %57, i64 %61, !dbg !263
  %63 = getelementptr inbounds double, double addrspace(13)* %57, i64 2, !dbg !263
  %64 = or i64 %58, 1, !dbg !265
  %65 = getelementptr inbounds double, double addrspace(13)* %57, i64 %64, !dbg !265
  %66 = add nuw nsw i64 %8, 1, !dbg !265
  %67 = getelementptr inbounds double, double addrspace(13)* %57, i64 %66, !dbg !265
  %68 = getelementptr inbounds double, double addrspace(13)* %57, i64 1, !dbg !265
  %69 = getelementptr inbounds double, double addrspace(13)* %57, i64 %58, !dbg !259
  %70 = getelementptr inbounds double, double addrspace(13)* %57, i64 %8, !dbg !259
  %value_phi18 = load double, double addrspace(13)* %57, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi17 = load double, double addrspace(13)* %70, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi16 = load double, double addrspace(13)* %69, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi15 = load double, double addrspace(13)* %68, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi14 = load double, double addrspace(13)* %67, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi13 = load double, double addrspace(13)* %65, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi12 = load double, double addrspace(13)* %63, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi11 = load double, double addrspace(13)* %62, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi10 = load double, double addrspace(13)* %60, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  switch i32 1308622848, label %L165 [
    i32 1409286144, label %L141
    i32 1124073472, label %L155
  ], !dbg !267

L141:                                             ; preds = %L128
  unreachable

L155:                                             ; preds = %L128
  unreachable

L165:                                             ; preds = %L128
  %71 = bitcast {} addrspace(10)* %2 to double addrspace(13)* addrspace(10)*, !dbg !268
  %72 = addrspacecast double addrspace(13)* addrspace(10)* %71 to double addrspace(13)* addrspace(11)*, !dbg !268
  %73 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %72, align 16, !dbg !268, !tbaa !63, !invariant.load !50, !alias.scope !262, !noalias !71, !nonnull !50
  %74 = shl nuw i64 %8, 1, !dbg !268
  %75 = add nuw nsw i64 %8, 1, !dbg !270
  %76 = or i64 %74, 1, !dbg !270
  %77 = add nuw i64 %8, 2, !dbg !272
  %78 = add nuw i64 %74, 2, !dbg !272
  br label %L174, !dbg !272

L174:                                             ; preds = %L165
  %79 = getelementptr inbounds double, double addrspace(13)* %73, i64 %78, !dbg !272
  %80 = getelementptr inbounds double, double addrspace(13)* %73, i64 %77, !dbg !272
  %81 = getelementptr inbounds double, double addrspace(13)* %73, i64 2, !dbg !272
  %82 = getelementptr inbounds double, double addrspace(13)* %73, i64 %76, !dbg !270
  %83 = getelementptr inbounds double, double addrspace(13)* %73, i64 %75, !dbg !270
  %84 = getelementptr inbounds double, double addrspace(13)* %73, i64 1, !dbg !270
  %85 = getelementptr inbounds double, double addrspace(13)* %73, i64 %74, !dbg !268
  %86 = getelementptr inbounds double, double addrspace(13)* %73, i64 %8, !dbg !268
  %value_phi27 = load double, double addrspace(13)* %73, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi26 = load double, double addrspace(13)* %86, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi25 = load double, double addrspace(13)* %85, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi24 = load double, double addrspace(13)* %84, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi23 = load double, double addrspace(13)* %83, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi22 = load double, double addrspace(13)* %82, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi21 = load double, double addrspace(13)* %81, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi20 = load double, double addrspace(13)* %80, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %value_phi19 = load double, double addrspace(13)* %79, align 8, !dbg !223, !tbaa !172, !alias.scope !140, !noalias !156
  %87 = fmul double %value_phi18, %value_phi27, !dbg !274
  %88 = fmul double %value_phi17, %value_phi24, !dbg !274
  %89 = fmul double %value_phi16, %value_phi21, !dbg !274
  %90 = fadd double %87, %88, !dbg !277
  %91 = fadd double %90, %89, !dbg !277
  %92 = bitcast {} addrspace(10)* %0 to double addrspace(13)* addrspace(10)*, !dbg !282
  %93 = addrspacecast double addrspace(13)* addrspace(10)* %92 to double addrspace(13)* addrspace(11)*, !dbg !282
  %94 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %93, align 8, !dbg !282, !tbaa !63, !invariant.load !50, !alias.scope !262, !noalias !71, !nonnull !50
  store double %91, double addrspace(13)* %94, align 8, !dbg !282, !tbaa !172, !alias.scope !140, !noalias !239
  %95 = fmul double %value_phi18, %value_phi26, !dbg !288
  %96 = fmul double %value_phi17, %value_phi23, !dbg !288
  %97 = fmul double %value_phi16, %value_phi20, !dbg !288
  %98 = fadd double %95, %96, !dbg !290
  %99 = fadd double %98, %97, !dbg !290
  %100 = getelementptr inbounds double, double addrspace(13)* %94, i64 %8, !dbg !292
  store double %99, double addrspace(13)* %100, align 8, !dbg !292, !tbaa !172, !alias.scope !140, !noalias !239
  %101 = fmul double %value_phi18, %value_phi25, !dbg !295
  %102 = fmul double %value_phi17, %value_phi22, !dbg !295
  %103 = fmul double %value_phi16, %value_phi19, !dbg !295
  %104 = fadd double %101, %102, !dbg !297
  %105 = fadd double %104, %103, !dbg !297
  %106 = getelementptr inbounds double, double addrspace(13)* %94, i64 %74, !dbg !299
  store double %105, double addrspace(13)* %106, align 8, !dbg !299, !tbaa !172, !alias.scope !140, !noalias !239
  %107 = fmul double %value_phi15, %value_phi27, !dbg !302
  %108 = fmul double %value_phi14, %value_phi24, !dbg !302
  %109 = fmul double %value_phi13, %value_phi21, !dbg !302
  %110 = fadd double %107, %108, !dbg !304
  %111 = fadd double %110, %109, !dbg !304
  %112 = getelementptr inbounds double, double addrspace(13)* %94, i64 1, !dbg !306
  store double %111, double addrspace(13)* %112, align 8, !dbg !306, !tbaa !172, !alias.scope !140, !noalias !239
  %113 = fmul double %value_phi15, %value_phi26, !dbg !309
  %114 = fmul double %value_phi14, %value_phi23, !dbg !309
  %115 = fmul double %value_phi13, %value_phi20, !dbg !309
  %116 = fadd double %113, %114, !dbg !311
  %117 = fadd double %116, %115, !dbg !311
  %118 = getelementptr inbounds double, double addrspace(13)* %94, i64 %75, !dbg !313
  store double %117, double addrspace(13)* %118, align 8, !dbg !313, !tbaa !172, !alias.scope !140, !noalias !239
  %119 = fmul double %value_phi15, %value_phi25, !dbg !316
  %120 = fmul double %value_phi14, %value_phi22, !dbg !316
  %121 = fmul double %value_phi13, %value_phi19, !dbg !316
  %122 = fadd double %119, %120, !dbg !318
  %123 = fadd double %122, %121, !dbg !318
  %124 = getelementptr inbounds double, double addrspace(13)* %94, i64 %76, !dbg !320
  store double %123, double addrspace(13)* %124, align 8, !dbg !320, !tbaa !172, !alias.scope !140, !noalias !239
  %125 = fmul double %value_phi12, %value_phi27, !dbg !323
  %126 = fmul double %value_phi11, %value_phi24, !dbg !323
  %127 = fmul double %value_phi10, %value_phi21, !dbg !323
  %128 = fadd double %125, %126, !dbg !325
  %129 = fadd double %128, %127, !dbg !325
  %130 = getelementptr inbounds double, double addrspace(13)* %94, i64 2, !dbg !327
  store double %129, double addrspace(13)* %130, align 8, !dbg !327, !tbaa !172, !alias.scope !140, !noalias !239
  %131 = fmul double %value_phi12, %value_phi26, !dbg !330
  %132 = fmul double %value_phi11, %value_phi23, !dbg !330
  %133 = fmul double %value_phi10, %value_phi20, !dbg !330
  %134 = fadd double %131, %132, !dbg !332
  %135 = fadd double %134, %133, !dbg !332
  %136 = getelementptr inbounds double, double addrspace(13)* %94, i64 %77, !dbg !334
  store double %135, double addrspace(13)* %136, align 8, !dbg !334, !tbaa !172, !alias.scope !140, !noalias !239
  %137 = fmul double %value_phi12, %value_phi25, !dbg !337
  %138 = fmul double %value_phi11, %value_phi22, !dbg !337
  %139 = fmul double %value_phi10, %value_phi19, !dbg !337
  %140 = fadd double %137, %138, !dbg !339
  %141 = fadd double %140, %139, !dbg !339
  %142 = getelementptr inbounds double, double addrspace(13)* %94, i64 %78, !dbg !341
  store double %141, double addrspace(13)* %142, align 8, !dbg !341, !tbaa !172, !alias.scope !140, !noalias !239
  ret void, !dbg !344

xchg_wb:                                          ; preds = %L70
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %32, {} addrspace(10)* nonnull %46) #28, !dbg !254
  br label %L87, !dbg !257

fail:                                             ; preds = %L80
  %143 = addrspacecast {} addrspace(10)* %48 to {} addrspace(12)*, !dbg !257
  call void @ijl_type_error(i8* noundef getelementptr inbounds ([11 x i8], [11 x i8]* @_j_str3, i64 0, i64 0), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), {} addrspace(12)* %143) #31, !dbg !257
  unreachable, !dbg !257
}

define internal fastcc void @julia_gemm__880({} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %0, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %1, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %2) unnamed_addr #19 !dbg !345 {
top:
  %3 = alloca i8, align 1
  %4 = alloca i8, align 1
  %5 = alloca i64, align 16
  %6 = bitcast i64* %5 to i8*
  %7 = alloca i64, align 16
  %8 = bitcast i64* %7 to i8*
  %9 = alloca i64, align 16
  %10 = bitcast i64* %9 to i8*
  %11 = alloca i64, align 16
  %12 = bitcast i64* %11 to i8*
  %13 = alloca i64, align 16
  %14 = bitcast i64* %13 to i8*
  %15 = alloca i64, align 16
  %16 = bitcast i64* %15 to i8*
  %17 = alloca i64, align 16
  %18 = bitcast i64* %17 to i8*
  %19 = alloca i64, align 16
  %20 = bitcast i64* %19 to i8*
  %21 = call {}*** @julia.get_pgcstack()
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !358, metadata !DIExpression(DW_OP_deref)), !dbg !359
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !356, metadata !DIExpression(DW_OP_deref)), !dbg !359
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !355, metadata !DIExpression(DW_OP_deref)), !dbg !359
  call void @llvm.dbg.value(metadata i32 1308622848, metadata !352, metadata !DIExpression()), !dbg !359
  call void @llvm.dbg.value(metadata i32 1308622848, metadata !353, metadata !DIExpression()), !dbg !359
  call void @llvm.dbg.value(metadata double 1.000000e+00, metadata !354, metadata !DIExpression()), !dbg !359
  call void @llvm.dbg.value(metadata {} addrspace(10)* %0, metadata !355, metadata !DIExpression(DW_OP_deref)), !dbg !359
  call void @llvm.dbg.value(metadata {} addrspace(10)* %1, metadata !356, metadata !DIExpression(DW_OP_deref)), !dbg !359
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !357, metadata !DIExpression()), !dbg !359
  call void @llvm.dbg.value(metadata {} addrspace(10)* %2, metadata !358, metadata !DIExpression(DW_OP_deref)), !dbg !359
  %22 = bitcast {} addrspace(10)* %0 to {} addrspace(10)* addrspace(10)*, !dbg !360
  %23 = addrspacecast {} addrspace(10)* addrspace(10)* %22 to {} addrspace(10)* addrspace(11)*, !dbg !360
  %24 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %23, i64 3, !dbg !360
  %25 = bitcast {} addrspace(10)* addrspace(11)* %24 to i64 addrspace(11)*, !dbg !360
  %26 = load i64, i64 addrspace(11)* %25, align 8, !dbg !360, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %27 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %23, i64 4, !dbg !363
  %28 = bitcast {} addrspace(10)* addrspace(11)* %27 to i64 addrspace(11)*, !dbg !363
  %29 = load i64, i64 addrspace(11)* %28, align 16, !dbg !363, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %30 = bitcast {} addrspace(10)* %1 to {} addrspace(10)* addrspace(10)*, !dbg !365
  %31 = addrspacecast {} addrspace(10)* addrspace(10)* %30 to {} addrspace(10)* addrspace(11)*, !dbg !365
  %32 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %31, i64 3, !dbg !365
  %33 = bitcast {} addrspace(10)* addrspace(11)* %32 to i64 addrspace(11)*, !dbg !365
  %34 = load i64, i64 addrspace(11)* %33, align 8, !dbg !365, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %35 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %31, i64 4, !dbg !367
  %36 = bitcast {} addrspace(10)* addrspace(11)* %35 to i64 addrspace(11)*, !dbg !367
  %37 = load i64, i64 addrspace(11)* %36, align 16, !dbg !367, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %38 = icmp eq i64 %29, %34, !dbg !369
  br i1 %38, label %L37, label %arraysize18.L144_crit_edge, !dbg !373

L37:                                              ; preds = %top
  %39 = bitcast {} addrspace(10)* %2 to {} addrspace(10)* addrspace(10)*, !dbg !374
  %40 = addrspacecast {} addrspace(10)* addrspace(10)* %39 to {} addrspace(10)* addrspace(11)*, !dbg !374
  %41 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %40, i64 3, !dbg !374
  %42 = bitcast {} addrspace(10)* addrspace(11)* %41 to i64 addrspace(11)*, !dbg !374
  %43 = load i64, i64 addrspace(11)* %42, align 8, !dbg !374, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %44 = icmp eq i64 %26, %43, !dbg !369
  br i1 %44, label %L42, label %L144, !dbg !373

L42:                                              ; preds = %L37
  %45 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %40, i64 4, !dbg !374
  %46 = bitcast {} addrspace(10)* addrspace(11)* %45 to i64 addrspace(11)*, !dbg !374
  %47 = load i64, i64 addrspace(11)* %46, align 16, !dbg !374, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %48 = icmp eq i64 %37, %47, !dbg !369
  br i1 %48, label %L47, label %L144, !dbg !373

L47:                                              ; preds = %L42
  br i1 false, label %L56, label %L51, !dbg !375

L51:                                              ; preds = %L47
  br label %L69, !dbg !375

L56:                                              ; preds = %L47
  unreachable

L62:                                              ; No predecessors!
  unreachable

L69:                                              ; preds = %L51
  call void @llvm.lifetime.start.p0i8(i64 noundef 1, i8* noundef nonnull %3)
  store i8 78, i8* %3, align 1, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  br i1 false, label %L82, label %L77, !dbg !375

L77:                                              ; preds = %L69
  br label %L95, !dbg !375

L82:                                              ; preds = %L69
  unreachable

L88:                                              ; No predecessors!
  unreachable

L95:                                              ; preds = %L77
  %49 = addrspacecast {} addrspace(10)* %2 to {} addrspace(11)*, !dbg !374
  call void @llvm.lifetime.start.p0i8(i64 noundef 1, i8* noundef nonnull %4)
  store i8 78, i8* %4, align 1, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull %6)
  store i64 %26, i64* %5, align 16, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull %8)
  store i64 %37, i64* %7, align 16, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull %10)
  store i64 %29, i64* %9, align 16, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull %12)
  %50 = bitcast i64* %11 to double*, !dbg !379
  store double 1.000000e+00, double* %50, align 16, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  %51 = addrspacecast {} addrspace(10)* %0 to {} addrspace(11)*, !dbg !391
  %52 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %23, i64 3, !dbg !391
  %53 = bitcast {} addrspace(10)* addrspace(11)* %52 to i64 addrspace(11)*, !dbg !391
  %54 = load i64, i64 addrspace(11)* %53, align 8, !dbg !391, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.not83 = icmp eq i64 %54, 0, !dbg !397
  %55 = select i1 %.not83, i64 1, i64 %54, !dbg !402
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull %14)
  store i64 %55, i64* %13, align 16, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  %56 = addrspacecast {} addrspace(10)* %1 to {} addrspace(11)*, !dbg !391
  %57 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %31, i64 3, !dbg !391
  %58 = bitcast {} addrspace(10)* addrspace(11)* %57 to i64 addrspace(11)*, !dbg !391
  %59 = load i64, i64 addrspace(11)* %58, align 8, !dbg !391, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.not84 = icmp eq i64 %59, 0, !dbg !397
  %60 = select i1 %.not84, i64 1, i64 %59, !dbg !402
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull %16)
  store i64 %60, i64* %15, align 16, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull %18)
  %61 = bitcast i64* %17 to double*, !dbg !379
  store double 0.000000e+00, double* %61, align 16, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  %62 = icmp ugt i64 %26, 1, !dbg !402
  %63 = select i1 %62, i64 %26, i64 1, !dbg !402
  call void @llvm.lifetime.start.p0i8(i64 noundef 8, i8* noundef nonnull %20)
  store i64 %63, i64* %19, align 16, !dbg !379, !tbaa !136, !alias.scope !140, !noalias !388
  %64 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* noundef %51) #30, !dbg !404
  %65 = bitcast {}* %64 to i8**, !dbg !404
  %66 = load i8*, i8** %65, align 8, !dbg !404, !tbaa !63, !invariant.load !50, !alias.scope !68, !noalias !71, !nonnull !50
  %67 = ptrtoint i8* %66 to i64, !dbg !404
  %68 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* noundef %56) #30, !dbg !404
  %69 = bitcast {}* %68 to i8**, !dbg !404
  %70 = load i8*, i8** %69, align 8, !dbg !404, !tbaa !63, !invariant.load !50, !alias.scope !68, !noalias !71, !nonnull !50
  %71 = ptrtoint i8* %70 to i64, !dbg !404
  %72 = call nonnull {}* @julia.pointer_from_objref({} addrspace(11)* noundef %49) #30, !dbg !404
  %73 = bitcast {}* %72 to i8**, !dbg !404
  %74 = load i8*, i8** %73, align 8, !dbg !404, !tbaa !63, !invariant.load !50, !alias.scope !68, !noalias !71, !nonnull !50
  %75 = ptrtoint i8* %74 to i64, !dbg !404
  call void @dgemm_64_(i8* noundef nonnull %3, i8* noundef nonnull %4, i8* noundef nonnull %6, i8* noundef nonnull %8, i8* noundef nonnull %10, i8* noundef nonnull %12, i64 %67, i8* noundef nonnull %14, i64 %71, i8* noundef nonnull %16, i8* noundef nonnull %18, i64 %75, i8* noundef nonnull %20, i64 noundef 1, i64 noundef 1) [ "jl_roots"({} addrspace(10)* null, {} addrspace(10)* %2, {} addrspace(10)* null, {} addrspace(10)* null, {} addrspace(10)* %1, {} addrspace(10)* null, {} addrspace(10)* %0, {} addrspace(10)* null, {} addrspace(10)* null, {} addrspace(10)* null, {} addrspace(10)* null, {} addrspace(10)* null, {} addrspace(10)* null) ], !dbg !387
  ret void, !dbg !407

L144:                                             ; preds = %arraysize18.L144_crit_edge, %L42, %L37
  %76 = phi i64 [ %.pre, %arraysize18.L144_crit_edge ], [ %26, %L42 ], [ %43, %L37 ], !dbg !408
  %77 = bitcast {} addrspace(10)* %2 to {} addrspace(10)* addrspace(10)*, !dbg !408
  %.pre-phi58.pre-phi.pre-phi.pre-phi.pre-phi = addrspacecast {} addrspace(10)* addrspace(10)* %77 to {} addrspace(10)* addrspace(11)*, !dbg !408
  %78 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %.pre-phi58.pre-phi.pre-phi.pre-phi.pre-phi, i64 4, !dbg !408
  %79 = bitcast {} addrspace(10)* addrspace(11)* %78 to i64 addrspace(11)*, !dbg !408
  %80 = load i64, i64 addrspace(11)* %79, align 16, !dbg !408, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %current_task1961 = getelementptr inbounds {}**, {}*** %21, i64 -13, !dbg !410
  %current_task19 = bitcast {}*** %current_task1961 to {}**, !dbg !410
  %81 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task19, i64 noundef 16, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737038955296 to {}*) to {} addrspace(10)*)) #27, !dbg !410
  %82 = bitcast {} addrspace(10)* %81 to {} addrspace(10)* addrspace(10)*, !dbg !410
  %83 = addrspacecast {} addrspace(10)* addrspace(10)* %82 to {} addrspace(10)* addrspace(11)*, !dbg !410
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %83, align 8, !dbg !410, !tbaa !136, !alias.scope !140, !noalias !388
  %84 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %83, i64 1, !dbg !410
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %84, align 8, !dbg !410, !tbaa !136, !alias.scope !140, !noalias !388
  %85 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task19, i64 noundef 88, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140733424830864 to {}*) to {} addrspace(10)*)) #27, !dbg !410
  %86 = bitcast {} addrspace(10)* %85 to { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)*, !dbg !410
  %.repack = bitcast {} addrspace(10)* %85 to {} addrspace(10)* addrspace(10)*, !dbg !410
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737065511344 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack62 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 1, !dbg !410
  store i64 %26, i64 addrspace(10)* %.repack62, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack64 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 2, !dbg !410
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737065511312 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack64, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack66 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 3, !dbg !410
  store i64 %29, i64 addrspace(10)* %.repack66, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack68 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 4, !dbg !410
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737065511280 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack68, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack70 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 5, !dbg !410
  store i64 %34, i64 addrspace(10)* %.repack70, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack72 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 6, !dbg !410
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737065511312 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack72, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack74 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 7, !dbg !410
  store i64 %37, i64 addrspace(10)* %.repack74, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack76 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 8, !dbg !410
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737065511248 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack76, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack78.repack = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 9, i64 0, !dbg !410
  store i64 %76, i64 addrspace(10)* %.repack78.repack, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  %.repack78.repack80 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, [2 x i64] } addrspace(10)* %86, i64 0, i32 9, i64 1, !dbg !410
  store i64 %80, i64 addrspace(10)* %.repack78.repack80, align 8, !dbg !410, !tbaa !144, !alias.scope !140, !noalias !388
  store atomic {} addrspace(10)* %85, {} addrspace(10)* addrspace(11)* %83 release, align 8, !dbg !410, !tbaa !136, !alias.scope !140, !noalias !388
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %81, {} addrspace(10)* nonnull %85) #28, !dbg !410
  %87 = bitcast {} addrspace(10)* %81 to i8 addrspace(10)*, !dbg !410
  %88 = addrspacecast i8 addrspace(10)* %87 to i8 addrspace(11)*, !dbg !410
  %89 = getelementptr inbounds i8, i8 addrspace(11)* %88, i64 8, !dbg !410
  %90 = bitcast i8 addrspace(11)* %89 to {} addrspace(10)* addrspace(11)*, !dbg !410
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(11)* %90 release, align 8, !dbg !410, !tbaa !136, !alias.scope !140, !noalias !388
  %91 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %90 acquire, align 8, !dbg !412, !tbaa !136, !alias.scope !140, !noalias !156, !nonnull !50
  %92 = addrspacecast {} addrspace(10)* %91 to {} addrspace(11)*, !dbg !420
  %.not82 = icmp eq {} addrspace(11)* %92, addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(11)*), !dbg !420
  br i1 %.not82, label %L152, label %L169, !dbg !420

L152:                                             ; preds = %L144
  %93 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task19, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737021929984 to {}*) to {} addrspace(10)*)) #27, !dbg !421
  %94 = bitcast {} addrspace(10)* %93 to {} addrspace(10)* addrspace(10)*, !dbg !421
  store {} addrspace(10)* %81, {} addrspace(10)* addrspace(10)* %94, align 8, !dbg !421, !tbaa !144, !alias.scope !140, !noalias !388
  %95 = call nonnull {} addrspace(10)* ({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* noundef nonnull @ijl_invoke, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737062426592 to {}*) to {} addrspace(10)*), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140736998151488 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226350624 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140736999550464 to {}*) to {} addrspace(10)*), {} addrspace(10)* nonnull %93) #29, !dbg !421
  %96 = cmpxchg {} addrspace(10)* addrspace(11)* %90, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* %95 acq_rel acquire, align 8, !dbg !424, !tbaa !136, !alias.scope !140, !noalias !156
  %97 = extractvalue { {} addrspace(10)*, i1 } %96, 0, !dbg !424
  %98 = extractvalue { {} addrspace(10)*, i1 } %96, 1, !dbg !424
  br i1 %98, label %xchg_wb, label %L162, !dbg !424

L162:                                             ; preds = %L152
  %99 = call {} addrspace(10)* @julia.typeof({} addrspace(10)* %97) #30, !dbg !427
  %100 = icmp eq {} addrspace(10)* %99, addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), !dbg !427
  br i1 %100, label %L169, label %fail24, !dbg !427

L169:                                             ; preds = %xchg_wb, %L162, %L144
  %value_phi22 = phi {} addrspace(10)* [ %95, %xchg_wb ], [ %91, %L144 ], [ %97, %L162 ]
  %101 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task19, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737046502288 to {}*) to {} addrspace(10)*)) #27, !dbg !409
  %102 = bitcast {} addrspace(10)* %101 to {} addrspace(10)* addrspace(10)*, !dbg !409
  store {} addrspace(10)* %value_phi22, {} addrspace(10)* addrspace(10)* %102, align 8, !dbg !409, !tbaa !144, !alias.scope !140, !noalias !388
  %103 = addrspacecast {} addrspace(10)* %101 to {} addrspace(12)*, !dbg !409
  call void @ijl_throw({} addrspace(12)* %103) #31, !dbg !409
  unreachable, !dbg !409

arraysize18.L144_crit_edge:                       ; preds = %top
  %104 = bitcast {} addrspace(10)* %2 to {} addrspace(10)* addrspace(10)*
  %.phi.trans.insert55 = addrspacecast {} addrspace(10)* addrspace(10)* %104 to {} addrspace(10)* addrspace(11)*
  %.phi.trans.insert56 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %.phi.trans.insert55, i64 3
  %.phi.trans.insert57 = bitcast {} addrspace(10)* addrspace(11)* %.phi.trans.insert56 to i64 addrspace(11)*
  %.pre = load i64, i64 addrspace(11)* %.phi.trans.insert57, align 8, !dbg !408, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  br label %L144, !dbg !373

xchg_wb:                                          ; preds = %L152
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %81, {} addrspace(10)* nonnull %95) #28, !dbg !424
  br label %L169, !dbg !427

fail24:                                           ; preds = %L162
  %105 = addrspacecast {} addrspace(10)* %97 to {} addrspace(12)*, !dbg !427
  call void @ijl_type_error(i8* noundef getelementptr inbounds ([11 x i8], [11 x i8]* @_j_str3, i64 0, i64 0), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), {} addrspace(12)* %105) #31, !dbg !427
  unreachable, !dbg !427
}

define internal fastcc void @julia_gemm_wrapper__804({} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %0, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %1, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %2) unnamed_addr #20 !dbg !428 {
top:
  %3 = call {}*** @julia.get_pgcstack()
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !438, metadata !DIExpression(DW_OP_deref)), !dbg !440
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !437, metadata !DIExpression(DW_OP_deref)), !dbg !440
  call void @llvm.dbg.value(metadata {} addrspace(10)* null, metadata !434, metadata !DIExpression(DW_OP_deref)), !dbg !440
  call void @llvm.dbg.value(metadata {} addrspace(10)* %0, metadata !434, metadata !DIExpression(DW_OP_deref)), !dbg !440
  call void @llvm.dbg.value(metadata i32 1308622848, metadata !435, metadata !DIExpression()), !dbg !440
  call void @llvm.dbg.value(metadata i32 1308622848, metadata !436, metadata !DIExpression()), !dbg !440
  call void @llvm.dbg.value(metadata {} addrspace(10)* %1, metadata !437, metadata !DIExpression(DW_OP_deref)), !dbg !440
  call void @llvm.dbg.value(metadata {} addrspace(10)* %2, metadata !438, metadata !DIExpression(DW_OP_deref)), !dbg !440
  call void @llvm.dbg.declare(metadata [2 x i8] addrspace(11)* addrspacecast ([2 x i8]* @_j_const1 to [2 x i8] addrspace(11)*), metadata !439, metadata !DIExpression(DW_OP_deref)), !dbg !441
  %4 = bitcast {} addrspace(10)* %1 to {} addrspace(10)* addrspace(10)*, !dbg !442
  %5 = addrspacecast {} addrspace(10)* addrspace(10)* %4 to {} addrspace(10)* addrspace(11)*, !dbg !442
  %6 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %5, i64 3, !dbg !442
  %7 = bitcast {} addrspace(10)* addrspace(11)* %6 to i64 addrspace(11)*, !dbg !442
  %8 = load i64, i64 addrspace(11)* %7, align 8, !dbg !442, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %9 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %5, i64 4, !dbg !442
  %10 = bitcast {} addrspace(10)* addrspace(11)* %9 to i64 addrspace(11)*, !dbg !442
  %11 = load i64, i64 addrspace(11)* %10, align 16, !dbg !442, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %12 = bitcast {} addrspace(10)* %2 to {} addrspace(10)* addrspace(10)*, !dbg !447
  %13 = addrspacecast {} addrspace(10)* addrspace(10)* %12 to {} addrspace(10)* addrspace(11)*, !dbg !447
  %14 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %13, i64 3, !dbg !447
  %15 = bitcast {} addrspace(10)* addrspace(11)* %14 to i64 addrspace(11)*, !dbg !447
  %16 = load i64, i64 addrspace(11)* %15, align 8, !dbg !447, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %17 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %13, i64 4, !dbg !447
  %18 = bitcast {} addrspace(10)* addrspace(11)* %17 to i64 addrspace(11)*, !dbg !447
  %19 = load i64, i64 addrspace(11)* %18, align 16, !dbg !447, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %20 = icmp eq i64 %11, %16, !dbg !450
  br i1 %20, label %L64, label %L38, !dbg !454

L38:                                              ; preds = %top
  %current_task1994 = getelementptr inbounds {}**, {}*** %3, i64 -13, !dbg !455
  %current_task19 = bitcast {}*** %current_task1994 to {}**, !dbg !455
  %21 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task19, i64 noundef 16, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737038955296 to {}*) to {} addrspace(10)*)) #27, !dbg !455
  %22 = bitcast {} addrspace(10)* %21 to {} addrspace(10)* addrspace(10)*, !dbg !455
  %23 = addrspacecast {} addrspace(10)* addrspace(10)* %22 to {} addrspace(10)* addrspace(11)*, !dbg !455
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %23, align 8, !dbg !455, !tbaa !136, !alias.scope !140, !noalias !458
  %24 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %23, i64 1, !dbg !455
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %24, align 8, !dbg !455, !tbaa !136, !alias.scope !140, !noalias !458
  %25 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task19, i64 noundef 72, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737021366480 to {}*) to {} addrspace(10)*)) #27, !dbg !455
  %26 = bitcast {} addrspace(10)* %25 to { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)*, !dbg !455
  %.repack = bitcast {} addrspace(10)* %25 to {} addrspace(10)* addrspace(10)*, !dbg !455
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772240 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack95 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %26, i64 0, i32 1, !dbg !455
  store i64 %8, i64 addrspace(10)* %.repack95, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack97 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %26, i64 0, i32 2, !dbg !455
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772208 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack97, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack99 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %26, i64 0, i32 3, !dbg !455
  store i64 %11, i64 addrspace(10)* %.repack99, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack101 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %26, i64 0, i32 4, !dbg !455
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772160 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack101, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack103 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %26, i64 0, i32 5, !dbg !455
  store i64 %16, i64 addrspace(10)* %.repack103, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack105 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %26, i64 0, i32 6, !dbg !455
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772208 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack105, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack107 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %26, i64 0, i32 7, !dbg !455
  store i64 %19, i64 addrspace(10)* %.repack107, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack109 = getelementptr inbounds { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %26, i64 0, i32 8, !dbg !455
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772128 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack109, align 8, !dbg !455, !tbaa !144, !alias.scope !140, !noalias !458
  store atomic {} addrspace(10)* %25, {} addrspace(10)* addrspace(11)* %23 release, align 8, !dbg !455, !tbaa !136, !alias.scope !140, !noalias !458
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %21, {} addrspace(10)* nonnull %25) #28, !dbg !455
  %27 = bitcast {} addrspace(10)* %21 to i8 addrspace(10)*, !dbg !455
  %28 = addrspacecast i8 addrspace(10)* %27 to i8 addrspace(11)*, !dbg !455
  %29 = getelementptr inbounds i8, i8 addrspace(11)* %28, i64 8, !dbg !455
  %30 = bitcast i8 addrspace(11)* %29 to {} addrspace(10)* addrspace(11)*, !dbg !455
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(11)* %30 release, align 8, !dbg !455, !tbaa !136, !alias.scope !140, !noalias !458
  %31 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %30 acquire, align 8, !dbg !461, !tbaa !136, !alias.scope !140, !noalias !156, !nonnull !50
  %32 = addrspacecast {} addrspace(10)* %31 to {} addrspace(11)*, !dbg !469
  %.not111 = icmp eq {} addrspace(11)* %32, addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(11)*), !dbg !469
  br i1 %.not111, label %L43, label %L60, !dbg !469

L43:                                              ; preds = %L38
  %33 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task19, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737021929984 to {}*) to {} addrspace(10)*)) #27, !dbg !470
  %34 = bitcast {} addrspace(10)* %33 to {} addrspace(10)* addrspace(10)*, !dbg !470
  store {} addrspace(10)* %21, {} addrspace(10)* addrspace(10)* %34, align 8, !dbg !470, !tbaa !144, !alias.scope !140, !noalias !458
  %35 = call nonnull {} addrspace(10)* ({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* noundef nonnull @ijl_invoke, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737062426592 to {}*) to {} addrspace(10)*), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140736998151488 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226350624 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140736999550464 to {}*) to {} addrspace(10)*), {} addrspace(10)* nonnull %33) #29, !dbg !470
  %36 = cmpxchg {} addrspace(10)* addrspace(11)* %30, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* %35 acq_rel acquire, align 8, !dbg !473, !tbaa !136, !alias.scope !140, !noalias !156
  %37 = extractvalue { {} addrspace(10)*, i1 } %36, 0, !dbg !473
  %38 = extractvalue { {} addrspace(10)*, i1 } %36, 1, !dbg !473
  br i1 %38, label %xchg_wb, label %L53, !dbg !473

L53:                                              ; preds = %L43
  %39 = call {} addrspace(10)* @julia.typeof({} addrspace(10)* %37) #30, !dbg !476
  %40 = icmp eq {} addrspace(10)* %39, addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), !dbg !476
  br i1 %40, label %L60, label %fail24, !dbg !476

L60:                                              ; preds = %xchg_wb, %L53, %L38
  %value_phi22 = phi {} addrspace(10)* [ %35, %xchg_wb ], [ %31, %L38 ], [ %37, %L53 ]
  %41 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task19, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737046502288 to {}*) to {} addrspace(10)*)) #27, !dbg !457
  %42 = bitcast {} addrspace(10)* %41 to {} addrspace(10)* addrspace(10)*, !dbg !457
  store {} addrspace(10)* %value_phi22, {} addrspace(10)* addrspace(10)* %42, align 8, !dbg !457, !tbaa !144, !alias.scope !140, !noalias !458
  %43 = addrspacecast {} addrspace(10)* %41 to {} addrspace(12)*, !dbg !457
  call void @ijl_throw({} addrspace(12)* %43) #31, !dbg !457
  unreachable, !dbg !457

L64:                                              ; preds = %top
  %44 = addrspacecast {} addrspace(10)* %0 to {} addrspace(11)*, !dbg !477
  %45 = addrspacecast {} addrspace(10)* %1 to {} addrspace(11)*, !dbg !477
  %.not112 = icmp eq {} addrspace(11)* %44, %45, !dbg !477
  %46 = addrspacecast {} addrspace(10)* %2 to {} addrspace(11)*, !dbg !477
  %.not114 = icmp eq {} addrspace(11)* %46, %44, !dbg !477
  %or.cond = or i1 %.not112, %.not114, !dbg !477
  br i1 %or.cond, label %L244, label %L70, !dbg !477

L70:                                              ; preds = %L64
  %.not115 = icmp eq i64 %8, 0, !dbg !478
  %.not140 = icmp eq i64 %11, 0
  %or.cond149 = select i1 %.not115, i1 true, i1 %.not140, !dbg !479
  %.not141 = icmp eq i64 %19, 0
  %or.cond150 = select i1 %or.cond149, i1 true, i1 %.not141, !dbg !479
  br i1 %or.cond150, label %L127, label %L83, !dbg !479

L83:                                              ; preds = %L70
  %.not143 = icmp eq i64 %8, 2, !dbg !480
  %.not144 = icmp eq i64 %11, 2
  %or.cond153 = select i1 %.not143, i1 %.not144, i1 false, !dbg !481
  %.not145 = icmp eq i64 %19, 2
  %or.cond154 = select i1 %or.cond153, i1 %.not145, i1 false, !dbg !481
  br i1 %or.cond154, label %L89, label %L91, !dbg !481

common.ret:                                       ; preds = %L181.L193_crit_edge, %L174, %L99, %L97, %L89
  ret void, !dbg !440

L89:                                              ; preds = %L83
  call fastcc void @julia_matmul2x2__878({} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %0, {} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %1, {} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %2) #26, !dbg !482
  br label %common.ret

L91:                                              ; preds = %L83
  %.not146 = icmp eq i64 %8, 3, !dbg !483
  %.not147 = icmp eq i64 %11, 3
  %or.cond155 = select i1 %.not146, i1 %.not147, i1 false, !dbg !484
  %.not148 = icmp eq i64 %19, 3
  %or.cond156 = select i1 %or.cond155, i1 %.not148, i1 false, !dbg !484
  br i1 %or.cond156, label %L97, label %L99, !dbg !484

L97:                                              ; preds = %L91
  call fastcc void @julia_matmul3x3__876({} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %0, {} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %1, {} addrspace(10)* nocapture noundef nonnull readonly align 16 dereferenceable(40) %2) #26, !dbg !485
  br label %common.ret

L99:                                              ; preds = %L91
  call fastcc void @julia_gemm__880({} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %1, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %2, {} addrspace(10)* noundef nonnull align 16 dereferenceable(40) %0) #26, !dbg !486
  br label %common.ret

L127:                                             ; preds = %L70
  %47 = bitcast {} addrspace(10)* %0 to {} addrspace(10)* addrspace(10)*, !dbg !487
  %48 = addrspacecast {} addrspace(10)* addrspace(10)* %47 to {} addrspace(10)* addrspace(11)*, !dbg !487
  %49 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %48, i64 3, !dbg !487
  %50 = bitcast {} addrspace(10)* addrspace(11)* %49 to i64 addrspace(11)*, !dbg !487
  %51 = load i64, i64 addrspace(11)* %50, align 8, !dbg !487, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %52 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %48, i64 4, !dbg !487
  %53 = bitcast {} addrspace(10)* addrspace(11)* %52 to i64 addrspace(11)*, !dbg !487
  %54 = load i64, i64 addrspace(11)* %53, align 16, !dbg !487, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.not116 = icmp eq i64 %51, %8, !dbg !489
  %.not139 = icmp eq i64 %54, %19
  %spec.select = select i1 %.not116, i1 %.not139, i1 false, !dbg !495
  br i1 %spec.select, label %L174, label %L145, !dbg !488

L145:                                             ; preds = %L127
  %current_task30117 = getelementptr inbounds {}**, {}*** %3, i64 -13, !dbg !496
  %current_task30 = bitcast {}*** %current_task30117 to {}**, !dbg !496
  %55 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task30, i64 noundef 16, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737038955296 to {}*) to {} addrspace(10)*)) #27, !dbg !496
  %56 = bitcast {} addrspace(10)* %55 to {} addrspace(10)* addrspace(10)*, !dbg !496
  %57 = addrspacecast {} addrspace(10)* addrspace(10)* %56 to {} addrspace(10)* addrspace(11)*, !dbg !496
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %57, align 8, !dbg !496, !tbaa !136, !alias.scope !140, !noalias !458
  %58 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %57, i64 1, !dbg !496
  store {} addrspace(10)* null, {} addrspace(10)* addrspace(11)* %58, align 8, !dbg !496, !tbaa !136, !alias.scope !140, !noalias !458
  %59 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task30, i64 noundef 64, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140733403908112 to {}*) to {} addrspace(10)*)) #27, !dbg !496
  %60 = bitcast {} addrspace(10)* %59 to { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)*, !dbg !496
  %.repack118 = bitcast {} addrspace(10)* %59 to {} addrspace(10)* addrspace(10)*, !dbg !496
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772000 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack118, align 8, !dbg !496, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack119.repack = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %60, i64 0, i32 1, i64 0, !dbg !496
  store i64 %51, i64 addrspace(10)* %.repack119.repack, align 8, !dbg !496, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack119.repack131 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %60, i64 0, i32 1, i64 1, !dbg !496
  store i64 %54, i64 addrspace(10)* %.repack119.repack131, align 8, !dbg !496, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack121 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %60, i64 0, i32 2, !dbg !496
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048771968 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack121, align 8, !dbg !496, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack123 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %60, i64 0, i32 3, !dbg !496
  store i64 %8, i64 addrspace(10)* %.repack123, align 8, !dbg !496, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack125 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %60, i64 0, i32 4, !dbg !496
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772208 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack125, align 8, !dbg !496, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack127 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %60, i64 0, i32 5, !dbg !496
  store i64 %19, i64 addrspace(10)* %.repack127, align 8, !dbg !496, !tbaa !144, !alias.scope !140, !noalias !458
  %.repack129 = getelementptr inbounds { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* }, { {} addrspace(10)*, [2 x i64], {} addrspace(10)*, i64, {} addrspace(10)*, i64, {} addrspace(10)* } addrspace(10)* %60, i64 0, i32 6, !dbg !496
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772128 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %.repack129, align 8, !dbg !496, !tbaa !144, !alias.scope !140, !noalias !458
  store atomic {} addrspace(10)* %59, {} addrspace(10)* addrspace(11)* %57 release, align 8, !dbg !496, !tbaa !136, !alias.scope !140, !noalias !458
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %55, {} addrspace(10)* nonnull %59) #28, !dbg !496
  %61 = bitcast {} addrspace(10)* %55 to i8 addrspace(10)*, !dbg !496
  %62 = addrspacecast i8 addrspace(10)* %61 to i8 addrspace(11)*, !dbg !496
  %63 = getelementptr inbounds i8, i8 addrspace(11)* %62, i64 8, !dbg !496
  %64 = bitcast i8 addrspace(11)* %63 to {} addrspace(10)* addrspace(11)*, !dbg !496
  store atomic {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(11)* %64 release, align 8, !dbg !496, !tbaa !136, !alias.scope !140, !noalias !458
  %65 = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %64 acquire, align 8, !dbg !498, !tbaa !136, !alias.scope !140, !noalias !156, !nonnull !50
  %66 = addrspacecast {} addrspace(10)* %65 to {} addrspace(11)*, !dbg !502
  %.not133 = icmp eq {} addrspace(11)* %66, addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(11)*), !dbg !502
  br i1 %.not133, label %L153, label %L170, !dbg !502

L153:                                             ; preds = %L145
  %67 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task30, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737021929984 to {}*) to {} addrspace(10)*)) #27, !dbg !503
  %68 = bitcast {} addrspace(10)* %67 to {} addrspace(10)* addrspace(10)*, !dbg !503
  store {} addrspace(10)* %55, {} addrspace(10)* addrspace(10)* %68, align 8, !dbg !503, !tbaa !144, !alias.scope !140, !noalias !458
  %69 = call nonnull {} addrspace(10)* ({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)*, {} addrspace(10)*, {} addrspace(10)*, ...) @julia.call2({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* noundef nonnull @ijl_invoke, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737062426592 to {}*) to {} addrspace(10)*), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140736998151488 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226350624 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140736999550464 to {}*) to {} addrspace(10)*), {} addrspace(10)* nonnull %67) #29, !dbg !503
  %70 = cmpxchg {} addrspace(10)* addrspace(11)* %64, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737226317832 to {}*) to {} addrspace(10)*), {} addrspace(10)* %69 acq_rel acquire, align 8, !dbg !505, !tbaa !136, !alias.scope !140, !noalias !156
  %71 = extractvalue { {} addrspace(10)*, i1 } %70, 0, !dbg !505
  %72 = extractvalue { {} addrspace(10)*, i1 } %70, 1, !dbg !505
  br i1 %72, label %xchg_wb34, label %L163, !dbg !505

L163:                                             ; preds = %L153
  %73 = call {} addrspace(10)* @julia.typeof({} addrspace(10)* %71) #30, !dbg !507
  %74 = icmp eq {} addrspace(10)* %73, addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), !dbg !507
  br i1 %74, label %L170, label %fail40, !dbg !507

L170:                                             ; preds = %xchg_wb34, %L163, %L145
  %value_phi36 = phi {} addrspace(10)* [ %69, %xchg_wb34 ], [ %65, %L145 ], [ %71, %L163 ]
  %75 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task30, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737046502288 to {}*) to {} addrspace(10)*)) #27, !dbg !497
  %76 = bitcast {} addrspace(10)* %75 to {} addrspace(10)* addrspace(10)*, !dbg !497
  store {} addrspace(10)* %value_phi36, {} addrspace(10)* addrspace(10)* %76, align 8, !dbg !497, !tbaa !144, !alias.scope !140, !noalias !458
  %77 = addrspacecast {} addrspace(10)* %75 to {} addrspace(12)*, !dbg !497
  call void @ijl_throw({} addrspace(12)* %77) #31, !dbg !497
  unreachable, !dbg !497

L174:                                             ; preds = %L127
  %78 = bitcast {} addrspace(10)* %0 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)*, !dbg !508
  %79 = addrspacecast { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(10)* %78 to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)*, !dbg !508
  %80 = getelementptr inbounds { i8 addrspace(13)*, i64, i16, i16, i32 }, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)* %79, i64 0, i32 1, !dbg !508
  %81 = load i64, i64 addrspace(11)* %80, align 8, !dbg !508, !tbaa !63, !range !67, !invariant.load !50, !alias.scope !68, !noalias !71
  %.not134.not = icmp eq i64 %81, 0, !dbg !515
  br i1 %.not134.not, label %common.ret, label %L181.L193_crit_edge, !dbg !512

L181.L193_crit_edge:                              ; preds = %L174
  %82 = bitcast {} addrspace(10)* %0 to i8 addrspace(13)* addrspace(10)*, !dbg !516
  %83 = addrspacecast i8 addrspace(13)* addrspace(10)* %82 to i8 addrspace(13)* addrspace(11)*, !dbg !516
  %.pre136161 = load i8 addrspace(13)*, i8 addrspace(13)* addrspace(11)* %83, align 8, !dbg !516, !tbaa !63, !invariant.load !50, !alias.scope !521, !noalias !71
  %84 = shl nuw i64 %81, 3, !dbg !522
  call void @llvm.memset.p13i8.i64(i8 addrspace(13)* align 8 %.pre136161, i8 noundef 0, i64 %84, i1 noundef false), !dbg !516, !tbaa !172, !alias.scope !140, !noalias !458
  br label %common.ret, !dbg !440

L244:                                             ; preds = %L64
  %85 = call fastcc [1 x {} addrspace(10)*] @julia_ArgumentError_864() #26, !dbg !523
  %current_task26113 = getelementptr inbounds {}**, {}*** %3, i64 -13, !dbg !523
  %current_task26 = bitcast {}*** %current_task26113 to {}**, !dbg !523
  %86 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** nonnull %current_task26, i64 noundef 8, {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737081189696 to {}*) to {} addrspace(10)*)) #27, !dbg !523
  %87 = bitcast {} addrspace(10)* %86 to {} addrspace(10)* addrspace(10)*, !dbg !523
  store {} addrspace(10)* addrspacecast ({}* inttoptr (i64 140737048772048 to {}*) to {} addrspace(10)*), {} addrspace(10)* addrspace(10)* %87, align 8, !dbg !523, !tbaa !144, !alias.scope !140, !noalias !458
  %88 = addrspacecast {} addrspace(10)* %86 to {} addrspace(12)*, !dbg !523
  call void @ijl_throw({} addrspace(12)* %88) #31, !dbg !523
  unreachable, !dbg !523

xchg_wb:                                          ; preds = %L43
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %21, {} addrspace(10)* nonnull %35) #28, !dbg !473
  br label %L60, !dbg !476

fail24:                                           ; preds = %L53
  %89 = addrspacecast {} addrspace(10)* %37 to {} addrspace(12)*, !dbg !476
  call void @ijl_type_error(i8* noundef getelementptr inbounds ([11 x i8], [11 x i8]* @_j_str3, i64 0, i64 0), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), {} addrspace(12)* %89) #31, !dbg !476
  unreachable, !dbg !476

xchg_wb34:                                        ; preds = %L153
  call void ({} addrspace(10)*, ...) @julia.write_barrier({} addrspace(10)* noundef nonnull %55, {} addrspace(10)* nonnull %69) #28, !dbg !505
  br label %L170, !dbg !507

fail40:                                           ; preds = %L163
  %90 = addrspacecast {} addrspace(10)* %71 to {} addrspace(12)*, !dbg !507
  call void @ijl_type_error(i8* noundef getelementptr inbounds ([11 x i8], [11 x i8]* @_j_str3, i64 0, i64 0), {} addrspace(10)* noundef addrspacecast ({}* inttoptr (i64 140737082352976 to {}*) to {} addrspace(10)*), {} addrspace(12)* %90) #31, !dbg !507
  unreachable, !dbg !507
}

declare i8* @malloc(i8 %0)

declare void @dlacpy_64_(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6)

attributes #0 = { nofree readnone "enzyme_inactive" "enzyme_shouldrecompute" "enzymejl_world"="33474" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn "enzymejl_world"="33474" }
attributes #2 = { inaccessiblememonly allocsize(1) "enzymejl_world"="33474" }
attributes #3 = { inaccessiblememonly nofree norecurse nounwind "enzyme_inactive" "enzymejl_world"="33474" }
attributes #4 = { nofree "enzymejl_world"="33474" }
attributes #5 = { "enzymejl_world"="33474" }
attributes #6 = { noreturn "enzymejl_world"="33474" }
attributes #7 = { nofree norecurse nounwind readnone "enzyme_inactive" "enzyme_shouldrecompute" "enzymejl_world"="33474" }
attributes #8 = { argmemonly nofree nosync nounwind willreturn "enzymejl_world"="33474" }
attributes #9 = { nofree nounwind readnone "enzymejl_world"="33474" }
attributes #10 = { nofree "enzyme_inactive" "enzymejl_world"="33474" }
attributes #11 = { inaccessiblememonly }
attributes #12 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn }
attributes #13 = { "enzymejl_mi"="140733395731808" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "probe-stack"="inline-asm" }
attributes #14 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #15 = { argmemonly nofree nounwind willreturn writeonly }
attributes #16 = { nofree nosync readnone "enzyme_parmremove"="0" "enzymejl_mi"="140737064959920" "enzymejl_rt"="140737081189696" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #17 = { "enzyme_parmremove"="1,2,5" "enzyme_retremove" "enzymejl_mi"="140733325480016" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #18 = { "enzyme_parmremove"="1,2,5" "enzyme_retremove" "enzymejl_mi"="140733326854752" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #19 = { "enzyme_parmremove"="0,1,2,5" "enzyme_retremove" "enzymejl_mi"="140733428578384" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #20 = { "enzyme_parmremove"="1,2,5" "enzyme_retremove" "enzymejl_mi"="140733325192384" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #21 = { mustprogress willreturn "enzymejl_mi"="140733395731808" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "probe-stack"="inline-asm" }
attributes #22 = { mustprogress willreturn "enzyme_parmremove"="1,2,5" "enzyme_retremove" "enzymejl_mi"="140733325192384" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #23 = { mustprogress willreturn "enzyme_parmremove"="1,2,5" "enzyme_retremove" "enzymejl_mi"="140733325480016" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #24 = { mustprogress willreturn "enzyme_parmremove"="1,2,5" "enzyme_retremove" "enzymejl_mi"="140733326854752" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #25 = { mustprogress willreturn "enzyme_parmremove"="0,1,2,5" "enzyme_retremove" "enzymejl_mi"="140733428578384" "enzymejl_rt"="140733404252304" "enzymejl_world"="33474" "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #26 = { "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #27 = { inaccessiblememonly allocsize(1) }
attributes #28 = { nounwind }
attributes #29 = { nofree }
attributes #30 = { nounwind readnone }
attributes #31 = { noreturn }
attributes #32 = { mustprogress willreturn }
attributes #33 = { inaccessiblememonly mustprogress willreturn }
attributes #34 = { mustprogress willreturn "frame-pointer"="all" "probe-stack"="inline-asm" }
attributes #35 = { inaccessiblememonly mustprogress willreturn allocsize(1) }
attributes #36 = { mustprogress nounwind willreturn }
attributes #37 = { mustprogress nofree willreturn }
attributes #38 = { mustprogress nounwind readnone willreturn }
attributes #39 = { mustprogress noreturn willreturn }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2, !4, !6, !7, !8, !9, !11, !13, !14, !16, !18, !19, !20, !21, !23, !25, !26, !27, !28, !30, !31, !32, !34, !35, !36, !38, !39}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!3 = !DIFile(filename: "/h/292/drehwald/prog/julia/usr/share/julia/stdlib/v1.9/LinearAlgebra/src/matmul.jl", directory: ".")
!4 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!5 = !DIFile(filename: "abstractarray.jl", directory: ".")
!6 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!7 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!8 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!9 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !10, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!10 = !DIFile(filename: "/h/344/drehwald/prog/Enzyme.jl/min.jl", directory: ".")
!11 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !12, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!12 = !DIFile(filename: "boot.jl", directory: ".")
!13 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !12, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!14 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !15, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!15 = !DIFile(filename: "range.jl", directory: ".")
!16 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !17, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!17 = !DIFile(filename: "array.jl", directory: ".")
!18 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!19 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!20 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!21 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !22, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!22 = !DIFile(filename: "broadcast.jl", directory: ".")
!23 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !24, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!24 = !DIFile(filename: "multidimensional.jl", directory: ".")
!25 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!26 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!27 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!28 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !29, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!29 = !DIFile(filename: "subarray.jl", directory: ".")
!30 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!31 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!32 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !33, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!33 = !DIFile(filename: "char.jl", directory: ".")
!34 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !33, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!35 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !12, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!36 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !37, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!37 = !DIFile(filename: "/h/292/drehwald/prog/julia/usr/share/julia/stdlib/v1.9/LinearAlgebra/src/blas.jl", directory: ".")
!38 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !5, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!39 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !40, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!40 = !DIFile(filename: "/h/292/drehwald/prog/julia/usr/share/julia/stdlib/v1.9/LinearAlgebra/src/transpose.jl", directory: ".")
!41 = distinct !DISubprogram(name: "foo", linkageName: "julia_foo_801", scope: null, file: !10, line: 4, type: !42, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !51)
!42 = !DISubroutineType(types: !43)
!43 = !{!44, !49, !44, !44}
!44 = !DIDerivedType(tag: DW_TAG_typedef, name: "Array", baseType: !45)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46, size: 64, align: 64)
!46 = !DICompositeType(tag: DW_TAG_structure_type, name: "jl_value_t", file: !47, line: 71, align: 64, elements: !48)
!47 = !DIFile(filename: "julia.h", directory: "")
!48 = !{!45}
!49 = !DICompositeType(tag: DW_TAG_structure_type, name: "#foo", align: 8, elements: !50, runtimeLang: DW_LANG_Julia, identifier: "140733404209360")
!50 = !{}
!51 = !{!52, !53, !54}
!52 = !DILocalVariable(name: "#self#", arg: 1, scope: !41, file: !10, line: 4, type: !49)
!53 = !DILocalVariable(name: "A", arg: 2, scope: !41, file: !10, line: 4, type: !44)
!54 = !DILocalVariable(name: "B", arg: 3, scope: !41, file: !10, line: 4, type: !44)
!55 = !DILocation(line: 0, scope: !41, inlinedAt: !56)
!56 = distinct !DILocation(line: 0, scope: !41)
!57 = !DILocation(line: 148, scope: !58, inlinedAt: !60)
!58 = distinct !DISubprogram(name: "size;", linkageName: "size", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !50)
!59 = !DISubroutineType(types: !50)
!60 = distinct !DILocation(line: 148, scope: !61, inlinedAt: !62)
!61 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !3, file: !3, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !50)
!62 = distinct !DILocation(line: 4, scope: !41, inlinedAt: !56)
!63 = !{!64, !64, i64 0, i64 1}
!64 = !{!"jtbaa_const", !65, i64 0}
!65 = !{!"jtbaa", !66, i64 0}
!66 = !{!"jtbaa"}
!67 = !{i64 0, i64 9223372036854775807}
!68 = !{!69}
!69 = !{!"jnoalias_const", !70}
!70 = !{!"jnoalias"}
!71 = !{!72, !73, !74, !75}
!72 = !{!"jnoalias_gcframe", !70}
!73 = !{!"jnoalias_stack", !70}
!74 = !{!"jnoalias_data", !70}
!75 = !{!"jnoalias_typemd", !70}
!76 = !DILocation(line: 479, scope: !77, inlinedAt: !78)
!77 = distinct !DISubprogram(name: "Array;", linkageName: "Array", scope: !12, file: !12, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !50)
!78 = distinct !DILocation(line: 487, scope: !77, inlinedAt: !79)
!79 = distinct !DILocation(line: 374, scope: !80, inlinedAt: !60)
!80 = distinct !DISubprogram(name: "similar;", linkageName: "similar", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !50)
!81 = !DILocation(line: 161, scope: !82, inlinedAt: !83)
!82 = distinct !DISubprogram(name: "mul!;", linkageName: "mul!", scope: !3, file: !3, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !50)
!83 = distinct !DILocation(line: 276, scope: !82, inlinedAt: !60)
!84 = !DILocation(line: 0, scope: !41)
!85 = distinct !DISubprogram(name: "ArgumentError", linkageName: "julia_ArgumentError_864", scope: null, file: !12, line: 327, type: !86, scopeLine: 327, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !11, retainedNodes: !96)
!86 = !DISubroutineType(types: !87)
!87 = !{!88, !90, !92}
!88 = !DICompositeType(tag: DW_TAG_structure_type, name: "ArgumentError", size: 64, align: 64, elements: !89, runtimeLang: DW_LANG_Julia, identifier: "140737081189696")
!89 = !{!90}
!90 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !91, size: 64, align: 64)
!91 = !DICompositeType(tag: DW_TAG_structure_type, name: "jl_value_t", file: !47, line: 71, align: 64, elements: !89)
!92 = !DIDerivedType(tag: DW_TAG_typedef, name: "String", baseType: !93)
!93 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !94, size: 64, align: 64)
!94 = !DICompositeType(tag: DW_TAG_structure_type, name: "jl_value_t", file: !47, line: 71, align: 64, elements: !95)
!95 = !{!93}
!96 = !{!97, !98}
!97 = !DILocalVariable(name: "#ctor-self#", arg: 1, scope: !85, file: !12, line: 327, type: !90)
!98 = !DILocalVariable(name: "msg", arg: 2, scope: !85, file: !12, line: 327, type: !92)
!99 = !DILocation(line: 0, scope: !85)
!100 = !DILocation(line: 327, scope: !85)
!101 = distinct !DISubprogram(name: "matmul2x2!", linkageName: "julia_matmul2x2!_878", scope: null, file: !3, line: 1028, type: !102, scopeLine: 1028, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !109)
!102 = !DISubroutineType(types: !103)
!103 = !{!44, !104, !44, !105, !105, !44, !44, !106}
!104 = !DICompositeType(tag: DW_TAG_structure_type, name: "#matmul2x2!", align: 8, elements: !50, runtimeLang: DW_LANG_Julia, identifier: "140737048916480")
!105 = !DIBasicType(name: "Char", size: 32, encoding: DW_ATE_unsigned)
!106 = !DICompositeType(tag: DW_TAG_structure_type, name: "MulAddMul", size: 16, align: 8, elements: !107, runtimeLang: DW_LANG_Julia, identifier: "140737011489248")
!107 = !{!108, !108}
!108 = !DIBasicType(name: "Bool", size: 8, encoding: DW_ATE_unsigned)
!109 = !{!110, !111, !112, !113, !114, !115, !116}
!110 = !DILocalVariable(name: "#self#", arg: 1, scope: !101, file: !3, line: 1028, type: !104)
!111 = !DILocalVariable(name: "C", arg: 2, scope: !101, file: !3, line: 1028, type: !44)
!112 = !DILocalVariable(name: "tA", arg: 3, scope: !101, file: !3, line: 1028, type: !105)
!113 = !DILocalVariable(name: "tB", arg: 4, scope: !101, file: !3, line: 1028, type: !105)
!114 = !DILocalVariable(name: "A", arg: 5, scope: !101, file: !3, line: 1028, type: !44)
!115 = !DILocalVariable(name: "B", arg: 6, scope: !101, file: !3, line: 1028, type: !44)
!116 = !DILocalVariable(name: "_add", arg: 7, scope: !101, file: !3, line: 1028, type: !106)
!117 = !DILocation(line: 0, scope: !101)
!118 = !DILocation(line: 1028, scope: !101)
!119 = !DILocation(line: 150, scope: !120, inlinedAt: !121)
!120 = distinct !DISubprogram(name: "size;", linkageName: "size", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!121 = !DILocation(line: 1031, scope: !101)
!122 = !DILocation(line: 499, scope: !123, inlinedAt: !125)
!123 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !124, file: !124, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!124 = !DIFile(filename: "promotion.jl", directory: ".")
!125 = !DILocation(line: 462, scope: !126, inlinedAt: !128)
!126 = distinct !DISubprogram(name: "_eq;", linkageName: "_eq", scope: !127, file: !127, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!127 = !DIFile(filename: "tuple.jl", directory: ".")
!128 = !DILocation(line: 458, scope: !129, inlinedAt: !121)
!129 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !127, file: !127, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!130 = !DILocation(line: 463, scope: !126, inlinedAt: !128)
!131 = !DILocation(line: 150, scope: !120, inlinedAt: !132)
!132 = !DILocation(line: 1032, scope: !101)
!133 = !DILocation(line: 41, scope: !134, inlinedAt: !132)
!134 = distinct !DISubprogram(name: "LazyString;", linkageName: "LazyString", scope: !135, file: !135, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!135 = !DIFile(filename: "strings/lazy.jl", directory: ".")
!136 = !{!137, !137, i64 0}
!137 = !{!"jtbaa_mutab", !138, i64 0}
!138 = !{!"jtbaa_value", !139, i64 0}
!139 = !{!"jtbaa_data", !65, i64 0}
!140 = !{!74}
!141 = !{!142, !72, !73, !75, !69}
!142 = distinct !{!142, !143, !"na_addr13"}
!143 = distinct !{!143, !"addr13"}
!144 = !{!145, !145, i64 0}
!145 = !{!"jtbaa_immut", !138, i64 0}
!146 = !DILocation(line: 53, scope: !147, inlinedAt: !149)
!147 = distinct !DISubprogram(name: "getproperty;", linkageName: "getproperty", scope: !148, file: !148, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!148 = !DIFile(filename: "Base.jl", directory: ".")
!149 = !DILocation(line: 81, scope: !150, inlinedAt: !151)
!150 = distinct !DISubprogram(name: "String;", linkageName: "String", scope: !135, file: !135, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!151 = !DILocation(line: 232, scope: !152, inlinedAt: !154)
!152 = distinct !DISubprogram(name: "convert;", linkageName: "convert", scope: !153, file: !153, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!153 = !DIFile(filename: "strings/basic.jl", directory: ".")
!154 = !DILocation(line: 12, scope: !155, inlinedAt: !132)
!155 = distinct !DISubprogram(name: "DimensionMismatch;", linkageName: "DimensionMismatch", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!156 = !{!72, !73, !75, !69}
!157 = !DILocation(line: 82, scope: !150, inlinedAt: !151)
!158 = !DILocation(line: 107, scope: !159, inlinedAt: !161)
!159 = distinct !DISubprogram(name: "sprint;", linkageName: "sprint", scope: !160, file: !160, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!160 = !DIFile(filename: "strings/io.jl", directory: ".")
!161 = !DILocation(line: 83, scope: !150, inlinedAt: !151)
!162 = !DILocation(line: 61, scope: !163, inlinedAt: !164)
!163 = distinct !DISubprogram(name: "replaceproperty!;", linkageName: "replaceproperty!", scope: !148, file: !148, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!164 = !DILocation(line: 88, scope: !150, inlinedAt: !151)
!165 = !DILocation(line: 89, scope: !150, inlinedAt: !151)
!166 = !DILocation(line: 1035, scope: !101)
!167 = !DILocation(line: 14, scope: !168, inlinedAt: !170)
!168 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !169, file: !169, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!169 = !DIFile(filename: "essentials.jl", directory: ".")
!170 = !DILocation(line: 1044, scope: !101)
!171 = !{!142, !69}
!172 = !{!173, !173, i64 0}
!173 = !{!"jtbaa_arraybuf", !139, i64 0}
!174 = !DILocation(line: 1046, scope: !101)
!175 = !DILocation(line: 14, scope: !168, inlinedAt: !176)
!176 = !DILocation(line: 1055, scope: !101)
!177 = !DILocation(line: 14, scope: !168, inlinedAt: !178)
!178 = !DILocation(line: 1056, scope: !101)
!179 = !DILocation(line: 410, scope: !180, inlinedAt: !182)
!180 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !181, file: !181, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!181 = !DIFile(filename: "float.jl", directory: ".")
!182 = !DILocation(line: 1058, scope: !101)
!183 = !DILocation(line: 408, scope: !184, inlinedAt: !182)
!184 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !181, file: !181, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!185 = !DILocation(line: 971, scope: !186, inlinedAt: !187)
!186 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!187 = !DILocation(line: 670, scope: !188, inlinedAt: !189)
!188 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !24, file: !24, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!189 = !DILocation(line: 91, scope: !190, inlinedAt: !182)
!190 = distinct !DISubprogram(name: "_modify!;", linkageName: "_modify!", scope: !191, file: !191, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !50)
!191 = !DIFile(filename: "/h/292/drehwald/prog/julia/usr/share/julia/stdlib/v1.9/LinearAlgebra/src/generic.jl", directory: ".")
!192 = !DILocation(line: 410, scope: !180, inlinedAt: !193)
!193 = !DILocation(line: 1059, scope: !101)
!194 = !DILocation(line: 408, scope: !184, inlinedAt: !193)
!195 = !DILocation(line: 971, scope: !186, inlinedAt: !196)
!196 = !DILocation(line: 670, scope: !188, inlinedAt: !197)
!197 = !DILocation(line: 91, scope: !190, inlinedAt: !193)
!198 = !DILocation(line: 410, scope: !180, inlinedAt: !199)
!199 = !DILocation(line: 1060, scope: !101)
!200 = !DILocation(line: 408, scope: !184, inlinedAt: !199)
!201 = !DILocation(line: 971, scope: !186, inlinedAt: !202)
!202 = !DILocation(line: 670, scope: !188, inlinedAt: !203)
!203 = !DILocation(line: 91, scope: !190, inlinedAt: !199)
!204 = !DILocation(line: 410, scope: !180, inlinedAt: !205)
!205 = !DILocation(line: 1061, scope: !101)
!206 = !DILocation(line: 408, scope: !184, inlinedAt: !205)
!207 = !DILocation(line: 971, scope: !186, inlinedAt: !208)
!208 = !DILocation(line: 670, scope: !188, inlinedAt: !209)
!209 = !DILocation(line: 91, scope: !190, inlinedAt: !205)
!210 = !DILocation(line: 1063, scope: !101)
!211 = distinct !DISubprogram(name: "matmul3x3!", linkageName: "julia_matmul3x3!_876", scope: null, file: !3, line: 1071, type: !212, scopeLine: 1071, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !215)
!212 = !DISubroutineType(types: !213)
!213 = !{!44, !214, !44, !105, !105, !44, !44, !106}
!214 = !DICompositeType(tag: DW_TAG_structure_type, name: "#matmul3x3!", align: 8, elements: !50, runtimeLang: DW_LANG_Julia, identifier: "140737050799152")
!215 = !{!216, !217, !218, !219, !220, !221, !222}
!216 = !DILocalVariable(name: "#self#", arg: 1, scope: !211, file: !3, line: 1071, type: !214)
!217 = !DILocalVariable(name: "C", arg: 2, scope: !211, file: !3, line: 1071, type: !44)
!218 = !DILocalVariable(name: "tA", arg: 3, scope: !211, file: !3, line: 1071, type: !105)
!219 = !DILocalVariable(name: "tB", arg: 4, scope: !211, file: !3, line: 1071, type: !105)
!220 = !DILocalVariable(name: "A", arg: 5, scope: !211, file: !3, line: 1071, type: !44)
!221 = !DILocalVariable(name: "B", arg: 6, scope: !211, file: !3, line: 1071, type: !44)
!222 = !DILocalVariable(name: "_add", arg: 7, scope: !211, file: !3, line: 1071, type: !106)
!223 = !DILocation(line: 0, scope: !211)
!224 = !DILocation(line: 1071, scope: !211)
!225 = !DILocation(line: 150, scope: !226, inlinedAt: !227)
!226 = distinct !DISubprogram(name: "size;", linkageName: "size", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!227 = !DILocation(line: 1074, scope: !211)
!228 = !DILocation(line: 499, scope: !229, inlinedAt: !230)
!229 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !124, file: !124, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!230 = !DILocation(line: 462, scope: !231, inlinedAt: !232)
!231 = distinct !DISubprogram(name: "_eq;", linkageName: "_eq", scope: !127, file: !127, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!232 = !DILocation(line: 458, scope: !233, inlinedAt: !227)
!233 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !127, file: !127, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!234 = !DILocation(line: 463, scope: !231, inlinedAt: !232)
!235 = !DILocation(line: 150, scope: !226, inlinedAt: !236)
!236 = !DILocation(line: 1075, scope: !211)
!237 = !DILocation(line: 41, scope: !238, inlinedAt: !236)
!238 = distinct !DISubprogram(name: "LazyString;", linkageName: "LazyString", scope: !135, file: !135, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!239 = !{!240, !72, !73, !75, !69}
!240 = distinct !{!240, !241, !"na_addr13"}
!241 = distinct !{!241, !"addr13"}
!242 = !DILocation(line: 53, scope: !243, inlinedAt: !244)
!243 = distinct !DISubprogram(name: "getproperty;", linkageName: "getproperty", scope: !148, file: !148, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!244 = !DILocation(line: 81, scope: !245, inlinedAt: !246)
!245 = distinct !DISubprogram(name: "String;", linkageName: "String", scope: !135, file: !135, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!246 = !DILocation(line: 232, scope: !247, inlinedAt: !248)
!247 = distinct !DISubprogram(name: "convert;", linkageName: "convert", scope: !153, file: !153, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!248 = !DILocation(line: 12, scope: !249, inlinedAt: !236)
!249 = distinct !DISubprogram(name: "DimensionMismatch;", linkageName: "DimensionMismatch", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!250 = !DILocation(line: 82, scope: !245, inlinedAt: !246)
!251 = !DILocation(line: 107, scope: !252, inlinedAt: !253)
!252 = distinct !DISubprogram(name: "sprint;", linkageName: "sprint", scope: !160, file: !160, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!253 = !DILocation(line: 83, scope: !245, inlinedAt: !246)
!254 = !DILocation(line: 61, scope: !255, inlinedAt: !256)
!255 = distinct !DISubprogram(name: "replaceproperty!;", linkageName: "replaceproperty!", scope: !148, file: !148, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!256 = !DILocation(line: 88, scope: !245, inlinedAt: !246)
!257 = !DILocation(line: 89, scope: !245, inlinedAt: !246)
!258 = !DILocation(line: 1078, scope: !211)
!259 = !DILocation(line: 14, scope: !260, inlinedAt: !261)
!260 = distinct !DISubprogram(name: "getindex;", linkageName: "getindex", scope: !169, file: !169, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!261 = !DILocation(line: 1089, scope: !211)
!262 = !{!240, !69}
!263 = !DILocation(line: 14, scope: !260, inlinedAt: !264)
!264 = !DILocation(line: 1091, scope: !211)
!265 = !DILocation(line: 14, scope: !260, inlinedAt: !266)
!266 = !DILocation(line: 1090, scope: !211)
!267 = !DILocation(line: 1094, scope: !211)
!268 = !DILocation(line: 14, scope: !260, inlinedAt: !269)
!269 = !DILocation(line: 1105, scope: !211)
!270 = !DILocation(line: 14, scope: !260, inlinedAt: !271)
!271 = !DILocation(line: 1106, scope: !211)
!272 = !DILocation(line: 14, scope: !260, inlinedAt: !273)
!273 = !DILocation(line: 1107, scope: !211)
!274 = !DILocation(line: 410, scope: !275, inlinedAt: !276)
!275 = distinct !DISubprogram(name: "*;", linkageName: "*", scope: !181, file: !181, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!276 = !DILocation(line: 1110, scope: !211)
!277 = !DILocation(line: 408, scope: !278, inlinedAt: !279)
!278 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !181, file: !181, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!279 = !DILocation(line: 578, scope: !280, inlinedAt: !276)
!280 = distinct !DISubprogram(name: "+;", linkageName: "+", scope: !281, file: !281, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!281 = !DIFile(filename: "operators.jl", directory: ".")
!282 = !DILocation(line: 971, scope: !283, inlinedAt: !284)
!283 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!284 = !DILocation(line: 670, scope: !285, inlinedAt: !286)
!285 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !24, file: !24, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!286 = !DILocation(line: 91, scope: !287, inlinedAt: !276)
!287 = distinct !DISubprogram(name: "_modify!;", linkageName: "_modify!", scope: !191, file: !191, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !50)
!288 = !DILocation(line: 410, scope: !275, inlinedAt: !289)
!289 = !DILocation(line: 1111, scope: !211)
!290 = !DILocation(line: 408, scope: !278, inlinedAt: !291)
!291 = !DILocation(line: 578, scope: !280, inlinedAt: !289)
!292 = !DILocation(line: 971, scope: !283, inlinedAt: !293)
!293 = !DILocation(line: 670, scope: !285, inlinedAt: !294)
!294 = !DILocation(line: 91, scope: !287, inlinedAt: !289)
!295 = !DILocation(line: 410, scope: !275, inlinedAt: !296)
!296 = !DILocation(line: 1112, scope: !211)
!297 = !DILocation(line: 408, scope: !278, inlinedAt: !298)
!298 = !DILocation(line: 578, scope: !280, inlinedAt: !296)
!299 = !DILocation(line: 971, scope: !283, inlinedAt: !300)
!300 = !DILocation(line: 670, scope: !285, inlinedAt: !301)
!301 = !DILocation(line: 91, scope: !287, inlinedAt: !296)
!302 = !DILocation(line: 410, scope: !275, inlinedAt: !303)
!303 = !DILocation(line: 1114, scope: !211)
!304 = !DILocation(line: 408, scope: !278, inlinedAt: !305)
!305 = !DILocation(line: 578, scope: !280, inlinedAt: !303)
!306 = !DILocation(line: 971, scope: !283, inlinedAt: !307)
!307 = !DILocation(line: 670, scope: !285, inlinedAt: !308)
!308 = !DILocation(line: 91, scope: !287, inlinedAt: !303)
!309 = !DILocation(line: 410, scope: !275, inlinedAt: !310)
!310 = !DILocation(line: 1115, scope: !211)
!311 = !DILocation(line: 408, scope: !278, inlinedAt: !312)
!312 = !DILocation(line: 578, scope: !280, inlinedAt: !310)
!313 = !DILocation(line: 971, scope: !283, inlinedAt: !314)
!314 = !DILocation(line: 670, scope: !285, inlinedAt: !315)
!315 = !DILocation(line: 91, scope: !287, inlinedAt: !310)
!316 = !DILocation(line: 410, scope: !275, inlinedAt: !317)
!317 = !DILocation(line: 1116, scope: !211)
!318 = !DILocation(line: 408, scope: !278, inlinedAt: !319)
!319 = !DILocation(line: 578, scope: !280, inlinedAt: !317)
!320 = !DILocation(line: 971, scope: !283, inlinedAt: !321)
!321 = !DILocation(line: 670, scope: !285, inlinedAt: !322)
!322 = !DILocation(line: 91, scope: !287, inlinedAt: !317)
!323 = !DILocation(line: 410, scope: !275, inlinedAt: !324)
!324 = !DILocation(line: 1118, scope: !211)
!325 = !DILocation(line: 408, scope: !278, inlinedAt: !326)
!326 = !DILocation(line: 578, scope: !280, inlinedAt: !324)
!327 = !DILocation(line: 971, scope: !283, inlinedAt: !328)
!328 = !DILocation(line: 670, scope: !285, inlinedAt: !329)
!329 = !DILocation(line: 91, scope: !287, inlinedAt: !324)
!330 = !DILocation(line: 410, scope: !275, inlinedAt: !331)
!331 = !DILocation(line: 1119, scope: !211)
!332 = !DILocation(line: 408, scope: !278, inlinedAt: !333)
!333 = !DILocation(line: 578, scope: !280, inlinedAt: !331)
!334 = !DILocation(line: 971, scope: !283, inlinedAt: !335)
!335 = !DILocation(line: 670, scope: !285, inlinedAt: !336)
!336 = !DILocation(line: 91, scope: !287, inlinedAt: !331)
!337 = !DILocation(line: 410, scope: !275, inlinedAt: !338)
!338 = !DILocation(line: 1120, scope: !211)
!339 = !DILocation(line: 408, scope: !278, inlinedAt: !340)
!340 = !DILocation(line: 578, scope: !280, inlinedAt: !338)
!341 = !DILocation(line: 971, scope: !283, inlinedAt: !342)
!342 = !DILocation(line: 670, scope: !285, inlinedAt: !343)
!343 = !DILocation(line: 91, scope: !287, inlinedAt: !338)
!344 = !DILocation(line: 1122, scope: !211)
!345 = distinct !DISubprogram(name: "gemm!", linkageName: "julia_gemm!_880", scope: null, file: !37, line: 1505, type: !346, scopeLine: 1505, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !350)
!346 = !DISubroutineType(types: !347)
!347 = !{!44, !348, !105, !105, !349, !44, !44, !349, !44}
!348 = !DICompositeType(tag: DW_TAG_structure_type, name: "#gemm!", align: 8, elements: !50, runtimeLang: DW_LANG_Julia, identifier: "140737065507440")
!349 = !DIBasicType(name: "Float64", size: 64, encoding: DW_ATE_unsigned)
!350 = !{!351, !352, !353, !354, !355, !356, !357, !358}
!351 = !DILocalVariable(name: "#self#", arg: 1, scope: !345, file: !37, line: 1505, type: !348)
!352 = !DILocalVariable(name: "transA", arg: 2, scope: !345, file: !37, line: 1505, type: !105)
!353 = !DILocalVariable(name: "transB", arg: 3, scope: !345, file: !37, line: 1505, type: !105)
!354 = !DILocalVariable(name: "alpha", arg: 4, scope: !345, file: !37, line: 1505, type: !349)
!355 = !DILocalVariable(name: "A", arg: 5, scope: !345, file: !37, line: 1505, type: !44)
!356 = !DILocalVariable(name: "B", arg: 6, scope: !345, file: !37, line: 1505, type: !44)
!357 = !DILocalVariable(name: "beta", arg: 7, scope: !345, file: !37, line: 1505, type: !349)
!358 = !DILocalVariable(name: "C", arg: 8, scope: !345, file: !37, line: 1505, type: !44)
!359 = !DILocation(line: 0, scope: !345)
!360 = !DILocation(line: 148, scope: !361, inlinedAt: !362)
!361 = distinct !DISubprogram(name: "size;", linkageName: "size", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!362 = !DILocation(line: 1514, scope: !345)
!363 = !DILocation(line: 148, scope: !361, inlinedAt: !364)
!364 = !DILocation(line: 1515, scope: !345)
!365 = !DILocation(line: 148, scope: !361, inlinedAt: !366)
!366 = !DILocation(line: 1516, scope: !345)
!367 = !DILocation(line: 148, scope: !361, inlinedAt: !368)
!368 = !DILocation(line: 1517, scope: !345)
!369 = !DILocation(line: 499, scope: !370, inlinedAt: !371)
!370 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !124, file: !124, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!371 = !DILocation(line: 269, scope: !372, inlinedAt: !373)
!372 = distinct !DISubprogram(name: "!=;", linkageName: "!=", scope: !281, file: !281, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!373 = !DILocation(line: 1518, scope: !345)
!374 = !DILocation(line: 148, scope: !361, inlinedAt: !373)
!375 = !DILocation(line: 175, scope: !376, inlinedAt: !377)
!376 = distinct !DISubprogram(name: "Int8;", linkageName: "Int8", scope: !33, file: !33, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!377 = !DILocation(line: 185, scope: !378, inlinedAt: !379)
!378 = distinct !DISubprogram(name: "convert;", linkageName: "convert", scope: !33, file: !33, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!379 = !DILocation(line: 8, scope: !380, inlinedAt: !382)
!380 = distinct !DISubprogram(name: "RefValue;", linkageName: "RefValue", scope: !381, file: !381, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!381 = !DIFile(filename: "refvalue.jl", directory: ".")
!382 = !DILocation(line: 104, scope: !383, inlinedAt: !385)
!383 = distinct !DISubprogram(name: "convert;", linkageName: "convert", scope: !384, file: !384, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!384 = !DIFile(filename: "refpointer.jl", directory: ".")
!385 = !DILocation(line: 492, scope: !386, inlinedAt: !387)
!386 = distinct !DISubprogram(name: "cconvert;", linkageName: "cconvert", scope: !169, file: !169, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!387 = !DILocation(line: 1524, scope: !345)
!388 = !{!389, !72, !73, !75, !69}
!389 = distinct !{!389, !390, !"na_addr13"}
!390 = distinct !{!390, !"addr13"}
!391 = !DILocation(line: 150, scope: !361, inlinedAt: !392)
!392 = !DILocation(line: 173, scope: !393, inlinedAt: !395)
!393 = distinct !DISubprogram(name: "strides;", linkageName: "strides", scope: !394, file: !394, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!394 = !DIFile(filename: "reinterpretarray.jl", directory: ".")
!395 = !DILocation(line: 174, scope: !396, inlinedAt: !387)
!396 = distinct !DISubprogram(name: "stride;", linkageName: "stride", scope: !394, file: !394, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!397 = !DILocation(line: 83, scope: !398, inlinedAt: !400)
!398 = distinct !DISubprogram(name: "<;", linkageName: "<", scope: !399, file: !399, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!399 = !DIFile(filename: "int.jl", directory: ".")
!400 = !DILocation(line: 510, scope: !401, inlinedAt: !387)
!401 = distinct !DISubprogram(name: "max;", linkageName: "max", scope: !124, file: !124, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!402 = !DILocation(line: 575, scope: !403, inlinedAt: !400)
!403 = distinct !DISubprogram(name: "ifelse;", linkageName: "ifelse", scope: !169, file: !169, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!404 = !DILocation(line: 65, scope: !405, inlinedAt: !387)
!405 = distinct !DISubprogram(name: "unsafe_convert;", linkageName: "unsafe_convert", scope: !406, file: !406, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!406 = !DIFile(filename: "pointer.jl", directory: ".")
!407 = !DILocation(line: 1533, scope: !345)
!408 = !DILocation(line: 150, scope: !361, inlinedAt: !409)
!409 = !DILocation(line: 1519, scope: !345)
!410 = !DILocation(line: 41, scope: !411, inlinedAt: !409)
!411 = distinct !DISubprogram(name: "LazyString;", linkageName: "LazyString", scope: !135, file: !135, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!412 = !DILocation(line: 53, scope: !413, inlinedAt: !414)
!413 = distinct !DISubprogram(name: "getproperty;", linkageName: "getproperty", scope: !148, file: !148, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!414 = !DILocation(line: 81, scope: !415, inlinedAt: !416)
!415 = distinct !DISubprogram(name: "String;", linkageName: "String", scope: !135, file: !135, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!416 = !DILocation(line: 232, scope: !417, inlinedAt: !418)
!417 = distinct !DISubprogram(name: "convert;", linkageName: "convert", scope: !153, file: !153, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!418 = !DILocation(line: 12, scope: !419, inlinedAt: !409)
!419 = distinct !DISubprogram(name: "DimensionMismatch;", linkageName: "DimensionMismatch", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!420 = !DILocation(line: 82, scope: !415, inlinedAt: !416)
!421 = !DILocation(line: 107, scope: !422, inlinedAt: !423)
!422 = distinct !DISubprogram(name: "sprint;", linkageName: "sprint", scope: !160, file: !160, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!423 = !DILocation(line: 83, scope: !415, inlinedAt: !416)
!424 = !DILocation(line: 61, scope: !425, inlinedAt: !426)
!425 = distinct !DISubprogram(name: "replaceproperty!;", linkageName: "replaceproperty!", scope: !148, file: !148, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !50)
!426 = !DILocation(line: 88, scope: !415, inlinedAt: !416)
!427 = !DILocation(line: 89, scope: !415, inlinedAt: !416)
!428 = distinct !DISubprogram(name: "gemm_wrapper!", linkageName: "julia_gemm_wrapper!_804", scope: null, file: !3, line: 639, type: !429, scopeLine: 639, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !432)
!429 = !DISubroutineType(types: !430)
!430 = !{!44, !431, !44, !105, !105, !44, !44, !106}
!431 = !DICompositeType(tag: DW_TAG_structure_type, name: "#gemm_wrapper!", align: 8, elements: !50, runtimeLang: DW_LANG_Julia, identifier: "140737048754400")
!432 = !{!433, !434, !435, !436, !437, !438, !439}
!433 = !DILocalVariable(name: "#self#", arg: 1, scope: !428, file: !3, line: 639, type: !431)
!434 = !DILocalVariable(name: "C", arg: 2, scope: !428, file: !3, line: 639, type: !44)
!435 = !DILocalVariable(name: "tA", arg: 3, scope: !428, file: !3, line: 639, type: !105)
!436 = !DILocalVariable(name: "tB", arg: 4, scope: !428, file: !3, line: 639, type: !105)
!437 = !DILocalVariable(name: "A", arg: 5, scope: !428, file: !3, line: 639, type: !44)
!438 = !DILocalVariable(name: "B", arg: 6, scope: !428, file: !3, line: 639, type: !44)
!439 = !DILocalVariable(name: "_add", arg: 7, scope: !428, file: !3, line: 639, type: !106)
!440 = !DILocation(line: 0, scope: !428)
!441 = !DILocation(line: 639, scope: !428)
!442 = !DILocation(line: 148, scope: !443, inlinedAt: !444)
!443 = distinct !DISubprogram(name: "size;", linkageName: "size", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!444 = !DILocation(line: 725, scope: !445, inlinedAt: !446)
!445 = distinct !DISubprogram(name: "lapack_size;", linkageName: "lapack_size", scope: !3, file: !3, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!446 = !DILocation(line: 642, scope: !428)
!447 = !DILocation(line: 148, scope: !443, inlinedAt: !448)
!448 = !DILocation(line: 725, scope: !445, inlinedAt: !449)
!449 = !DILocation(line: 643, scope: !428)
!450 = !DILocation(line: 499, scope: !451, inlinedAt: !452)
!451 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !124, file: !124, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!452 = !DILocation(line: 269, scope: !453, inlinedAt: !454)
!453 = distinct !DISubprogram(name: "!=;", linkageName: "!=", scope: !281, file: !281, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!454 = !DILocation(line: 645, scope: !428)
!455 = !DILocation(line: 41, scope: !456, inlinedAt: !457)
!456 = distinct !DISubprogram(name: "LazyString;", linkageName: "LazyString", scope: !135, file: !135, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!457 = !DILocation(line: 646, scope: !428)
!458 = !{!459, !72, !73, !75, !69}
!459 = distinct !{!459, !460, !"na_addr13"}
!460 = distinct !{!460, !"addr13"}
!461 = !DILocation(line: 53, scope: !462, inlinedAt: !463)
!462 = distinct !DISubprogram(name: "getproperty;", linkageName: "getproperty", scope: !148, file: !148, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!463 = !DILocation(line: 81, scope: !464, inlinedAt: !465)
!464 = distinct !DISubprogram(name: "String;", linkageName: "String", scope: !135, file: !135, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!465 = !DILocation(line: 232, scope: !466, inlinedAt: !467)
!466 = distinct !DISubprogram(name: "convert;", linkageName: "convert", scope: !153, file: !153, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!467 = !DILocation(line: 12, scope: !468, inlinedAt: !457)
!468 = distinct !DISubprogram(name: "DimensionMismatch;", linkageName: "DimensionMismatch", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!469 = !DILocation(line: 82, scope: !464, inlinedAt: !465)
!470 = !DILocation(line: 107, scope: !471, inlinedAt: !472)
!471 = distinct !DISubprogram(name: "sprint;", linkageName: "sprint", scope: !160, file: !160, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!472 = !DILocation(line: 83, scope: !464, inlinedAt: !465)
!473 = !DILocation(line: 61, scope: !474, inlinedAt: !475)
!474 = distinct !DISubprogram(name: "replaceproperty!;", linkageName: "replaceproperty!", scope: !148, file: !148, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!475 = !DILocation(line: 88, scope: !464, inlinedAt: !465)
!476 = !DILocation(line: 89, scope: !464, inlinedAt: !465)
!477 = !DILocation(line: 649, scope: !428)
!478 = !DILocation(line: 499, scope: !451, inlinedAt: !479)
!479 = !DILocation(line: 653, scope: !428)
!480 = !DILocation(line: 499, scope: !451, inlinedAt: !481)
!481 = !DILocation(line: 660, scope: !428)
!482 = !DILocation(line: 661, scope: !428)
!483 = !DILocation(line: 499, scope: !451, inlinedAt: !484)
!484 = !DILocation(line: 663, scope: !428)
!485 = !DILocation(line: 664, scope: !428)
!486 = !DILocation(line: 674, scope: !428)
!487 = !DILocation(line: 150, scope: !443, inlinedAt: !488)
!488 = !DILocation(line: 654, scope: !428)
!489 = !DILocation(line: 499, scope: !451, inlinedAt: !490)
!490 = !DILocation(line: 462, scope: !491, inlinedAt: !492)
!491 = distinct !DISubprogram(name: "_eq;", linkageName: "_eq", scope: !127, file: !127, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!492 = !DILocation(line: 458, scope: !493, inlinedAt: !494)
!493 = distinct !DISubprogram(name: "==;", linkageName: "==", scope: !127, file: !127, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!494 = !DILocation(line: 269, scope: !453, inlinedAt: !488)
!495 = !DILocation(line: 463, scope: !491, inlinedAt: !492)
!496 = !DILocation(line: 41, scope: !456, inlinedAt: !497)
!497 = !DILocation(line: 655, scope: !428)
!498 = !DILocation(line: 53, scope: !462, inlinedAt: !499)
!499 = !DILocation(line: 81, scope: !464, inlinedAt: !500)
!500 = !DILocation(line: 232, scope: !466, inlinedAt: !501)
!501 = !DILocation(line: 12, scope: !468, inlinedAt: !497)
!502 = !DILocation(line: 82, scope: !464, inlinedAt: !500)
!503 = !DILocation(line: 107, scope: !471, inlinedAt: !504)
!504 = !DILocation(line: 83, scope: !464, inlinedAt: !500)
!505 = !DILocation(line: 61, scope: !474, inlinedAt: !506)
!506 = !DILocation(line: 88, scope: !464, inlinedAt: !500)
!507 = !DILocation(line: 89, scope: !464, inlinedAt: !500)
!508 = !DILocation(line: 10, scope: !509, inlinedAt: !510)
!509 = distinct !DISubprogram(name: "length;", linkageName: "length", scope: !169, file: !169, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!510 = !DILocation(line: 1223, scope: !511, inlinedAt: !512)
!511 = distinct !DISubprogram(name: "isempty;", linkageName: "isempty", scope: !5, file: !5, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!512 = !DILocation(line: 99, scope: !513, inlinedAt: !514)
!513 = distinct !DISubprogram(name: "_rmul_or_fill!;", linkageName: "_rmul_or_fill!", scope: !191, file: !191, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!514 = !DILocation(line: 657, scope: !428)
!515 = !DILocation(line: 499, scope: !451, inlinedAt: !510)
!516 = !DILocation(line: 969, scope: !517, inlinedAt: !518)
!517 = distinct !DISubprogram(name: "setindex!;", linkageName: "setindex!", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!518 = !DILocation(line: 349, scope: !519, inlinedAt: !520)
!519 = distinct !DISubprogram(name: "fill!;", linkageName: "fill!", scope: !17, file: !17, type: !59, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !50)
!520 = !DILocation(line: 103, scope: !513, inlinedAt: !514)
!521 = !{!459, !69}
!522 = !DILocation(line: 348, scope: !519, inlinedAt: !520)
!523 = !DILocation(line: 650, scope: !428)
!524 = distinct !DISubprogram(name: "foo", linkageName: "julia_foo_801", scope: null, file: !10, line: 4, type: !42, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !525)
!525 = !{!526, !527, !528}
!526 = !DILocalVariable(name: "#self#", arg: 1, scope: !524, file: !10, line: 4, type: !49)
!527 = !DILocalVariable(name: "A", arg: 2, scope: !524, file: !10, line: 4, type: !44)
!528 = !DILocalVariable(name: "B", arg: 3, scope: !524, file: !10, line: 4, type: !44)
!529 = !DILocation(line: 0, scope: !524, inlinedAt: !530)
!530 = distinct !DILocation(line: 0, scope: !524)
!531 = !DILocation(line: 148, scope: !58, inlinedAt: !532)
!532 = distinct !DILocation(line: 148, scope: !61, inlinedAt: !533)
!533 = distinct !DILocation(line: 4, scope: !524, inlinedAt: !530)
!534 = !DILocation(line: 479, scope: !77, inlinedAt: !535)
!535 = distinct !DILocation(line: 487, scope: !77, inlinedAt: !536)
!536 = distinct !DILocation(line: 374, scope: !80, inlinedAt: !532)
!537 = !DILocation(line: 161, scope: !82, inlinedAt: !538)
!538 = distinct !DILocation(line: 276, scope: !82, inlinedAt: !532)
!539 = !DILocation(line: 0, scope: !524)
!540 = distinct !DISubprogram(name: "foo", linkageName: "julia_foo_801", scope: null, file: !10, line: 4, type: !42, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !541)
!541 = !{!542, !543, !544}
!542 = !DILocalVariable(name: "#self#", arg: 1, scope: !540, file: !10, line: 4, type: !49)
!543 = !DILocalVariable(name: "A", arg: 2, scope: !540, file: !10, line: 4, type: !44)
!544 = !DILocalVariable(name: "B", arg: 3, scope: !540, file: !10, line: 4, type: !44)
!545 = !DILocation(line: 0, scope: !540, inlinedAt: !546)
!546 = distinct !DILocation(line: 0, scope: !540)
!547 = !DILocation(line: 148, scope: !58, inlinedAt: !548)
!548 = distinct !DILocation(line: 148, scope: !61, inlinedAt: !549)
!549 = distinct !DILocation(line: 4, scope: !540, inlinedAt: !546)
!550 = !DILocation(line: 479, scope: !77, inlinedAt: !551)
!551 = distinct !DILocation(line: 487, scope: !77, inlinedAt: !552)
!552 = distinct !DILocation(line: 374, scope: !80, inlinedAt: !548)
!553 = !DILocation(line: 161, scope: !82, inlinedAt: !554)
!554 = distinct !DILocation(line: 276, scope: !82, inlinedAt: !548)
!555 = !DILocation(line: 0, scope: !540)
!556 = distinct !DISubprogram(name: "gemm_wrapper!", linkageName: "julia_gemm_wrapper!_804", scope: null, file: !3, line: 639, type: !429, scopeLine: 639, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !557)
!557 = !{!558, !559, !560, !561, !562, !563, !564}
!558 = !DILocalVariable(name: "#self#", arg: 1, scope: !556, file: !3, line: 639, type: !431)
!559 = !DILocalVariable(name: "C", arg: 2, scope: !556, file: !3, line: 639, type: !44)
!560 = !DILocalVariable(name: "tA", arg: 3, scope: !556, file: !3, line: 639, type: !105)
!561 = !DILocalVariable(name: "tB", arg: 4, scope: !556, file: !3, line: 639, type: !105)
!562 = !DILocalVariable(name: "A", arg: 5, scope: !556, file: !3, line: 639, type: !44)
!563 = !DILocalVariable(name: "B", arg: 6, scope: !556, file: !3, line: 639, type: !44)
!564 = !DILocalVariable(name: "_add", arg: 7, scope: !556, file: !3, line: 639, type: !106)
!565 = !DILocation(line: 0, scope: !556)
!566 = !DILocation(line: 639, scope: !556)
!567 = !DILocation(line: 148, scope: !443, inlinedAt: !568)
!568 = !DILocation(line: 725, scope: !445, inlinedAt: !569)
!569 = !DILocation(line: 642, scope: !556)
!570 = !DILocation(line: 148, scope: !443, inlinedAt: !571)
!571 = !DILocation(line: 725, scope: !445, inlinedAt: !572)
!572 = !DILocation(line: 643, scope: !556)
!573 = !DILocation(line: 499, scope: !451, inlinedAt: !574)
!574 = !DILocation(line: 269, scope: !453, inlinedAt: !575)
!575 = !DILocation(line: 645, scope: !556)
!576 = !DILocation(line: 41, scope: !456, inlinedAt: !577)
!577 = !DILocation(line: 646, scope: !556)
!578 = !{!579, !72, !73, !75, !69}
!579 = distinct !{!579, !580, !"na_addr13"}
!580 = distinct !{!580, !"addr13"}
!581 = !DILocation(line: 53, scope: !462, inlinedAt: !582)
!582 = !DILocation(line: 81, scope: !464, inlinedAt: !583)
!583 = !DILocation(line: 232, scope: !466, inlinedAt: !584)
!584 = !DILocation(line: 12, scope: !468, inlinedAt: !577)
!585 = !DILocation(line: 82, scope: !464, inlinedAt: !583)
!586 = !DILocation(line: 107, scope: !471, inlinedAt: !587)
!587 = !DILocation(line: 83, scope: !464, inlinedAt: !583)
!588 = !DILocation(line: 61, scope: !474, inlinedAt: !589)
!589 = !DILocation(line: 88, scope: !464, inlinedAt: !583)
!590 = !DILocation(line: 89, scope: !464, inlinedAt: !583)
!591 = !DILocation(line: 649, scope: !556)
!592 = !DILocation(line: 499, scope: !451, inlinedAt: !593)
!593 = !DILocation(line: 653, scope: !556)
!594 = !DILocation(line: 499, scope: !451, inlinedAt: !595)
!595 = !DILocation(line: 660, scope: !556)
!596 = !DILocation(line: 661, scope: !556)
!597 = !DILocation(line: 499, scope: !451, inlinedAt: !598)
!598 = !DILocation(line: 663, scope: !556)
!599 = !DILocation(line: 664, scope: !556)
!600 = !DILocation(line: 674, scope: !556)
!601 = !DILocation(line: 150, scope: !443, inlinedAt: !602)
!602 = !DILocation(line: 654, scope: !556)
!603 = !DILocation(line: 499, scope: !451, inlinedAt: !604)
!604 = !DILocation(line: 462, scope: !491, inlinedAt: !605)
!605 = !DILocation(line: 458, scope: !493, inlinedAt: !606)
!606 = !DILocation(line: 269, scope: !453, inlinedAt: !602)
!607 = !DILocation(line: 463, scope: !491, inlinedAt: !605)
!608 = !DILocation(line: 41, scope: !456, inlinedAt: !609)
!609 = !DILocation(line: 655, scope: !556)
!610 = !DILocation(line: 53, scope: !462, inlinedAt: !611)
!611 = !DILocation(line: 81, scope: !464, inlinedAt: !612)
!612 = !DILocation(line: 232, scope: !466, inlinedAt: !613)
!613 = !DILocation(line: 12, scope: !468, inlinedAt: !609)
!614 = !DILocation(line: 82, scope: !464, inlinedAt: !612)
!615 = !DILocation(line: 107, scope: !471, inlinedAt: !616)
!616 = !DILocation(line: 83, scope: !464, inlinedAt: !612)
!617 = !DILocation(line: 61, scope: !474, inlinedAt: !618)
!618 = !DILocation(line: 88, scope: !464, inlinedAt: !612)
!619 = !DILocation(line: 89, scope: !464, inlinedAt: !612)
!620 = !DILocation(line: 10, scope: !509, inlinedAt: !621)
!621 = !DILocation(line: 1223, scope: !511, inlinedAt: !622)
!622 = !DILocation(line: 99, scope: !513, inlinedAt: !623)
!623 = !DILocation(line: 657, scope: !556)
!624 = !DILocation(line: 499, scope: !451, inlinedAt: !621)
!625 = !DILocation(line: 969, scope: !517, inlinedAt: !626)
!626 = !DILocation(line: 349, scope: !519, inlinedAt: !627)
!627 = !DILocation(line: 103, scope: !513, inlinedAt: !623)
!628 = !{!579, !69}
!629 = !DILocation(line: 348, scope: !519, inlinedAt: !627)
!630 = !DILocation(line: 650, scope: !556)
!631 = distinct !DISubprogram(name: "gemm_wrapper!", linkageName: "julia_gemm_wrapper!_804", scope: null, file: !3, line: 639, type: !429, scopeLine: 639, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !632)
!632 = !{!633, !634, !635, !636, !637, !638, !639}
!633 = !DILocalVariable(name: "#self#", arg: 1, scope: !631, file: !3, line: 639, type: !431)
!634 = !DILocalVariable(name: "C", arg: 2, scope: !631, file: !3, line: 639, type: !44)
!635 = !DILocalVariable(name: "tA", arg: 3, scope: !631, file: !3, line: 639, type: !105)
!636 = !DILocalVariable(name: "tB", arg: 4, scope: !631, file: !3, line: 639, type: !105)
!637 = !DILocalVariable(name: "A", arg: 5, scope: !631, file: !3, line: 639, type: !44)
!638 = !DILocalVariable(name: "B", arg: 6, scope: !631, file: !3, line: 639, type: !44)
!639 = !DILocalVariable(name: "_add", arg: 7, scope: !631, file: !3, line: 639, type: !106)
!640 = !DILocation(line: 148, scope: !443, inlinedAt: !641)
!641 = !DILocation(line: 725, scope: !445, inlinedAt: !642)
!642 = !DILocation(line: 642, scope: !631)
!643 = !{!644, !69}
!644 = distinct !{!644, !645, !"primal"}
!645 = distinct !{!645, !" diff: %"}
!646 = !{!647, !72, !73, !74, !75}
!647 = distinct !{!647, !645, !"shadow_0"}
!648 = !DILocation(line: 148, scope: !443, inlinedAt: !649)
!649 = !DILocation(line: 725, scope: !445, inlinedAt: !650)
!650 = !DILocation(line: 643, scope: !631)
!651 = !{!652, !69}
!652 = distinct !{!652, !653, !"primal"}
!653 = distinct !{!653, !" diff: %"}
!654 = !{!655, !72, !73, !74, !75}
!655 = distinct !{!655, !653, !"shadow_0"}
!656 = !DILocation(line: 499, scope: !451, inlinedAt: !657)
!657 = !DILocation(line: 269, scope: !453, inlinedAt: !658)
!658 = !DILocation(line: 645, scope: !631)
!659 = !DILocation(line: 41, scope: !456, inlinedAt: !660)
!660 = !DILocation(line: 646, scope: !631)
!661 = !{!662, !72, !73, !75, !69}
!662 = distinct !{!662, !663, !"na_addr13"}
!663 = distinct !{!663, !"addr13"}
!664 = !DILocation(line: 53, scope: !462, inlinedAt: !665)
!665 = !DILocation(line: 81, scope: !464, inlinedAt: !666)
!666 = !DILocation(line: 232, scope: !466, inlinedAt: !667)
!667 = !DILocation(line: 12, scope: !468, inlinedAt: !660)
!668 = !DILocation(line: 82, scope: !464, inlinedAt: !666)
!669 = !DILocation(line: 107, scope: !471, inlinedAt: !670)
!670 = !DILocation(line: 83, scope: !464, inlinedAt: !666)
!671 = !DILocation(line: 61, scope: !474, inlinedAt: !672)
!672 = !DILocation(line: 88, scope: !464, inlinedAt: !666)
!673 = !DILocation(line: 89, scope: !464, inlinedAt: !666)
!674 = !DILocation(line: 649, scope: !631)
!675 = !DILocation(line: 499, scope: !451, inlinedAt: !676)
!676 = !DILocation(line: 653, scope: !631)
!677 = !DILocation(line: 499, scope: !451, inlinedAt: !678)
!678 = !DILocation(line: 660, scope: !631)
!679 = !DILocation(line: 0, scope: !631)
!680 = distinct !{}
!681 = !DILocation(line: 499, scope: !451, inlinedAt: !682)
!682 = !DILocation(line: 663, scope: !631)
!683 = !DILocation(line: 674, scope: !631)
!684 = !DILocation(line: 150, scope: !443, inlinedAt: !685)
!685 = !DILocation(line: 654, scope: !631)
!686 = !DILocation(line: 499, scope: !451, inlinedAt: !687)
!687 = !DILocation(line: 462, scope: !491, inlinedAt: !688)
!688 = !DILocation(line: 458, scope: !493, inlinedAt: !689)
!689 = !DILocation(line: 269, scope: !453, inlinedAt: !685)
!690 = !DILocation(line: 463, scope: !491, inlinedAt: !688)
!691 = !DILocation(line: 41, scope: !456, inlinedAt: !692)
!692 = !DILocation(line: 655, scope: !631)
!693 = !DILocation(line: 53, scope: !462, inlinedAt: !694)
!694 = !DILocation(line: 81, scope: !464, inlinedAt: !695)
!695 = !DILocation(line: 232, scope: !466, inlinedAt: !696)
!696 = !DILocation(line: 12, scope: !468, inlinedAt: !692)
!697 = !DILocation(line: 82, scope: !464, inlinedAt: !695)
!698 = !DILocation(line: 107, scope: !471, inlinedAt: !699)
!699 = !DILocation(line: 83, scope: !464, inlinedAt: !695)
!700 = !DILocation(line: 61, scope: !474, inlinedAt: !701)
!701 = !DILocation(line: 88, scope: !464, inlinedAt: !695)
!702 = !DILocation(line: 89, scope: !464, inlinedAt: !695)
!703 = !DILocation(line: 10, scope: !509, inlinedAt: !704)
!704 = !DILocation(line: 1223, scope: !511, inlinedAt: !705)
!705 = !DILocation(line: 99, scope: !513, inlinedAt: !706)
!706 = !DILocation(line: 657, scope: !631)
!707 = !DILocation(line: 499, scope: !451, inlinedAt: !704)
!708 = !DILocation(line: 969, scope: !517, inlinedAt: !709)
!709 = !DILocation(line: 349, scope: !519, inlinedAt: !710)
!710 = !DILocation(line: 103, scope: !513, inlinedAt: !706)
!711 = !{!662, !69}
!712 = !DILocation(line: 348, scope: !519, inlinedAt: !710)
!713 = !DILocation(line: 650, scope: !631)
!714 = !DILocation(line: 661, scope: !631)
!715 = !DILocation(line: 664, scope: !631)
!716 = distinct !DISubprogram(name: "matmul2x2!", linkageName: "julia_matmul2x2!_878", scope: null, file: !3, line: 1028, type: !102, scopeLine: 1028, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !717)
!717 = !{!718, !719, !720, !721, !722, !723, !724}
!718 = !DILocalVariable(name: "#self#", arg: 1, scope: !716, file: !3, line: 1028, type: !104)
!719 = !DILocalVariable(name: "C", arg: 2, scope: !716, file: !3, line: 1028, type: !44)
!720 = !DILocalVariable(name: "tA", arg: 3, scope: !716, file: !3, line: 1028, type: !105)
!721 = !DILocalVariable(name: "tB", arg: 4, scope: !716, file: !3, line: 1028, type: !105)
!722 = !DILocalVariable(name: "A", arg: 5, scope: !716, file: !3, line: 1028, type: !44)
!723 = !DILocalVariable(name: "B", arg: 6, scope: !716, file: !3, line: 1028, type: !44)
!724 = !DILocalVariable(name: "_add", arg: 7, scope: !716, file: !3, line: 1028, type: !106)
!725 = !DILocation(line: 0, scope: !716)
!726 = !DILocation(line: 1028, scope: !716)
!727 = !DILocation(line: 150, scope: !120, inlinedAt: !728)
!728 = !DILocation(line: 1031, scope: !716)
!729 = !DILocation(line: 499, scope: !123, inlinedAt: !730)
!730 = !DILocation(line: 462, scope: !126, inlinedAt: !731)
!731 = !DILocation(line: 458, scope: !129, inlinedAt: !728)
!732 = !DILocation(line: 463, scope: !126, inlinedAt: !731)
!733 = !DILocation(line: 150, scope: !120, inlinedAt: !734)
!734 = !DILocation(line: 1032, scope: !716)
!735 = !DILocation(line: 41, scope: !134, inlinedAt: !734)
!736 = !{!737, !72, !73, !75, !69}
!737 = distinct !{!737, !738, !"na_addr13"}
!738 = distinct !{!738, !"addr13"}
!739 = !DILocation(line: 53, scope: !147, inlinedAt: !740)
!740 = !DILocation(line: 81, scope: !150, inlinedAt: !741)
!741 = !DILocation(line: 232, scope: !152, inlinedAt: !742)
!742 = !DILocation(line: 12, scope: !155, inlinedAt: !734)
!743 = !DILocation(line: 82, scope: !150, inlinedAt: !741)
!744 = !DILocation(line: 107, scope: !159, inlinedAt: !745)
!745 = !DILocation(line: 83, scope: !150, inlinedAt: !741)
!746 = !DILocation(line: 61, scope: !163, inlinedAt: !747)
!747 = !DILocation(line: 88, scope: !150, inlinedAt: !741)
!748 = !DILocation(line: 89, scope: !150, inlinedAt: !741)
!749 = !DILocation(line: 14, scope: !168, inlinedAt: !750)
!750 = !DILocation(line: 1044, scope: !716)
!751 = !{!737, !69}
!752 = !DILocation(line: 14, scope: !168, inlinedAt: !753)
!753 = !DILocation(line: 1055, scope: !716)
!754 = !DILocation(line: 14, scope: !168, inlinedAt: !755)
!755 = !DILocation(line: 1056, scope: !716)
!756 = !DILocation(line: 410, scope: !180, inlinedAt: !757)
!757 = !DILocation(line: 1058, scope: !716)
!758 = !DILocation(line: 408, scope: !184, inlinedAt: !757)
!759 = !DILocation(line: 971, scope: !186, inlinedAt: !760)
!760 = !DILocation(line: 670, scope: !188, inlinedAt: !761)
!761 = !DILocation(line: 91, scope: !190, inlinedAt: !757)
!762 = !DILocation(line: 410, scope: !180, inlinedAt: !763)
!763 = !DILocation(line: 1059, scope: !716)
!764 = !DILocation(line: 408, scope: !184, inlinedAt: !763)
!765 = !DILocation(line: 971, scope: !186, inlinedAt: !766)
!766 = !DILocation(line: 670, scope: !188, inlinedAt: !767)
!767 = !DILocation(line: 91, scope: !190, inlinedAt: !763)
!768 = !DILocation(line: 410, scope: !180, inlinedAt: !769)
!769 = !DILocation(line: 1060, scope: !716)
!770 = !DILocation(line: 408, scope: !184, inlinedAt: !769)
!771 = !DILocation(line: 971, scope: !186, inlinedAt: !772)
!772 = !DILocation(line: 670, scope: !188, inlinedAt: !773)
!773 = !DILocation(line: 91, scope: !190, inlinedAt: !769)
!774 = !DILocation(line: 410, scope: !180, inlinedAt: !775)
!775 = !DILocation(line: 1061, scope: !716)
!776 = !DILocation(line: 408, scope: !184, inlinedAt: !775)
!777 = !DILocation(line: 971, scope: !186, inlinedAt: !778)
!778 = !DILocation(line: 670, scope: !188, inlinedAt: !779)
!779 = !DILocation(line: 91, scope: !190, inlinedAt: !775)
!780 = !DILocation(line: 1063, scope: !716)
!781 = distinct !DISubprogram(name: "matmul2x2!", linkageName: "julia_matmul2x2!_878", scope: null, file: !3, line: 1028, type: !102, scopeLine: 1028, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !18, retainedNodes: !782)
!782 = !{!783, !784, !785, !786, !787, !788, !789}
!783 = !DILocalVariable(name: "#self#", arg: 1, scope: !781, file: !3, line: 1028, type: !104)
!784 = !DILocalVariable(name: "C", arg: 2, scope: !781, file: !3, line: 1028, type: !44)
!785 = !DILocalVariable(name: "tA", arg: 3, scope: !781, file: !3, line: 1028, type: !105)
!786 = !DILocalVariable(name: "tB", arg: 4, scope: !781, file: !3, line: 1028, type: !105)
!787 = !DILocalVariable(name: "A", arg: 5, scope: !781, file: !3, line: 1028, type: !44)
!788 = !DILocalVariable(name: "B", arg: 6, scope: !781, file: !3, line: 1028, type: !44)
!789 = !DILocalVariable(name: "_add", arg: 7, scope: !781, file: !3, line: 1028, type: !106)
!790 = !DILocation(line: 150, scope: !120, inlinedAt: !791)
!791 = !DILocation(line: 1031, scope: !781)
!792 = !{!793, !69}
!793 = distinct !{!793, !794, !"primal"}
!794 = distinct !{!794, !" diff: %"}
!795 = !{!796, !72, !73, !74, !75}
!796 = distinct !{!796, !794, !"shadow_0"}
!797 = !{!798, !69}
!798 = distinct !{!798, !799, !"primal"}
!799 = distinct !{!799, !" diff: %"}
!800 = !{!801, !72, !73, !74, !75}
!801 = distinct !{!801, !799, !"shadow_0"}
!802 = !DILocation(line: 499, scope: !123, inlinedAt: !803)
!803 = !DILocation(line: 462, scope: !126, inlinedAt: !804)
!804 = !DILocation(line: 458, scope: !129, inlinedAt: !791)
!805 = !DILocation(line: 463, scope: !126, inlinedAt: !804)
!806 = !DILocation(line: 150, scope: !120, inlinedAt: !807)
!807 = !DILocation(line: 1032, scope: !781)
!808 = !{!809, !69}
!809 = distinct !{!809, !810, !"primal"}
!810 = distinct !{!810, !" diff: %"}
!811 = !{!812, !72, !73, !74, !75}
!812 = distinct !{!812, !810, !"shadow_0"}
!813 = !DILocation(line: 41, scope: !134, inlinedAt: !807)
!814 = !{!815, !72, !73, !75, !69}
!815 = distinct !{!815, !816, !"na_addr13"}
!816 = distinct !{!816, !"addr13"}
!817 = !DILocation(line: 53, scope: !147, inlinedAt: !818)
!818 = !DILocation(line: 81, scope: !150, inlinedAt: !819)
!819 = !DILocation(line: 232, scope: !152, inlinedAt: !820)
!820 = !DILocation(line: 12, scope: !155, inlinedAt: !807)
!821 = !DILocation(line: 82, scope: !150, inlinedAt: !819)
!822 = !DILocation(line: 107, scope: !159, inlinedAt: !823)
!823 = !DILocation(line: 83, scope: !150, inlinedAt: !819)
!824 = !DILocation(line: 61, scope: !163, inlinedAt: !825)
!825 = !DILocation(line: 88, scope: !150, inlinedAt: !819)
!826 = !DILocation(line: 89, scope: !150, inlinedAt: !819)
!827 = !DILocation(line: 14, scope: !168, inlinedAt: !828)
!828 = !DILocation(line: 1044, scope: !781)
!829 = !{!801, !737, !69}
!830 = !{!798, !72, !73, !74, !75}
!831 = !{!798, !737, !69}
!832 = !DILocation(line: 0, scope: !781)
!833 = !{!834, !74}
!834 = distinct !{!834, !835, !"primal"}
!835 = distinct !{!835, !" diff: %"}
!836 = !{!837, !72, !73, !75, !69}
!837 = distinct !{!837, !835, !"shadow_0"}
!838 = !DILocation(line: 14, scope: !168, inlinedAt: !839)
!839 = !DILocation(line: 1055, scope: !781)
!840 = !{!796, !737, !69}
!841 = !{!793, !72, !73, !74, !75}
!842 = !{!793, !737, !69}
!843 = !DILocation(line: 14, scope: !168, inlinedAt: !844)
!844 = !DILocation(line: 1056, scope: !781)
!845 = !{!846, !74}
!846 = distinct !{!846, !847, !"primal"}
!847 = distinct !{!847, !" diff: %"}
!848 = !{!849, !72, !73, !75, !69}
!849 = distinct !{!849, !847, !"shadow_0"}
!850 = !DILocation(line: 410, scope: !180, inlinedAt: !851)
!851 = !DILocation(line: 1058, scope: !781)
!852 = !DILocation(line: 408, scope: !184, inlinedAt: !851)
!853 = !DILocation(line: 971, scope: !186, inlinedAt: !854)
!854 = !DILocation(line: 670, scope: !188, inlinedAt: !855)
!855 = !DILocation(line: 91, scope: !190, inlinedAt: !851)
!856 = !{!812, !737, !69}
!857 = !{!809, !72, !73, !74, !75}
!858 = !{!809, !737, !69}
!859 = !{!860, !74}
!860 = distinct !{!860, !861, !"primal"}
!861 = distinct !{!861, !" diff: %"}
!862 = !{!863, !737, !72, !73, !75, !69}
!863 = distinct !{!863, !861, !"shadow_0"}
!864 = !DILocation(line: 410, scope: !180, inlinedAt: !865)
!865 = !DILocation(line: 1059, scope: !781)
!866 = !DILocation(line: 408, scope: !184, inlinedAt: !865)
!867 = !DILocation(line: 971, scope: !186, inlinedAt: !868)
!868 = !DILocation(line: 670, scope: !188, inlinedAt: !869)
!869 = !DILocation(line: 91, scope: !190, inlinedAt: !865)
!870 = !DILocation(line: 410, scope: !180, inlinedAt: !871)
!871 = !DILocation(line: 1060, scope: !781)
!872 = !DILocation(line: 408, scope: !184, inlinedAt: !871)
!873 = !DILocation(line: 971, scope: !186, inlinedAt: !874)
!874 = !DILocation(line: 670, scope: !188, inlinedAt: !875)
!875 = !DILocation(line: 91, scope: !190, inlinedAt: !871)
!876 = !DILocation(line: 410, scope: !180, inlinedAt: !877)
!877 = !DILocation(line: 1061, scope: !781)
!878 = !DILocation(line: 408, scope: !184, inlinedAt: !877)
!879 = !DILocation(line: 971, scope: !186, inlinedAt: !880)
!880 = !DILocation(line: 670, scope: !188, inlinedAt: !881)
!881 = !DILocation(line: 91, scope: !190, inlinedAt: !877)
!882 = !DILocation(line: 1063, scope: !781)
!883 = !{!863, !74}
!884 = !{!860, !737, !72, !73, !75, !69}
!885 = !{!849, !74}
!886 = !{!846, !72, !73, !75, !69}
!887 = !{!837, !74}
!888 = !{!834, !72, !73, !75, !69}
!889 = distinct !DISubprogram(name: "matmul3x3!", linkageName: "julia_matmul3x3!_876", scope: null, file: !3, line: 1071, type: !212, scopeLine: 1071, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !890)
!890 = !{!891, !892, !893, !894, !895, !896, !897}
!891 = !DILocalVariable(name: "#self#", arg: 1, scope: !889, file: !3, line: 1071, type: !214)
!892 = !DILocalVariable(name: "C", arg: 2, scope: !889, file: !3, line: 1071, type: !44)
!893 = !DILocalVariable(name: "tA", arg: 3, scope: !889, file: !3, line: 1071, type: !105)
!894 = !DILocalVariable(name: "tB", arg: 4, scope: !889, file: !3, line: 1071, type: !105)
!895 = !DILocalVariable(name: "A", arg: 5, scope: !889, file: !3, line: 1071, type: !44)
!896 = !DILocalVariable(name: "B", arg: 6, scope: !889, file: !3, line: 1071, type: !44)
!897 = !DILocalVariable(name: "_add", arg: 7, scope: !889, file: !3, line: 1071, type: !106)
!898 = !DILocation(line: 0, scope: !889)
!899 = !DILocation(line: 1071, scope: !889)
!900 = !DILocation(line: 150, scope: !226, inlinedAt: !901)
!901 = !DILocation(line: 1074, scope: !889)
!902 = !DILocation(line: 499, scope: !229, inlinedAt: !903)
!903 = !DILocation(line: 462, scope: !231, inlinedAt: !904)
!904 = !DILocation(line: 458, scope: !233, inlinedAt: !901)
!905 = !DILocation(line: 463, scope: !231, inlinedAt: !904)
!906 = !DILocation(line: 150, scope: !226, inlinedAt: !907)
!907 = !DILocation(line: 1075, scope: !889)
!908 = !DILocation(line: 41, scope: !238, inlinedAt: !907)
!909 = !{!910, !72, !73, !75, !69}
!910 = distinct !{!910, !911, !"na_addr13"}
!911 = distinct !{!911, !"addr13"}
!912 = !DILocation(line: 53, scope: !243, inlinedAt: !913)
!913 = !DILocation(line: 81, scope: !245, inlinedAt: !914)
!914 = !DILocation(line: 232, scope: !247, inlinedAt: !915)
!915 = !DILocation(line: 12, scope: !249, inlinedAt: !907)
!916 = !DILocation(line: 82, scope: !245, inlinedAt: !914)
!917 = !DILocation(line: 107, scope: !252, inlinedAt: !918)
!918 = !DILocation(line: 83, scope: !245, inlinedAt: !914)
!919 = !DILocation(line: 61, scope: !255, inlinedAt: !920)
!920 = !DILocation(line: 88, scope: !245, inlinedAt: !914)
!921 = !DILocation(line: 89, scope: !245, inlinedAt: !914)
!922 = !DILocation(line: 14, scope: !260, inlinedAt: !923)
!923 = !DILocation(line: 1089, scope: !889)
!924 = !{!910, !69}
!925 = !DILocation(line: 14, scope: !260, inlinedAt: !926)
!926 = !DILocation(line: 1091, scope: !889)
!927 = !DILocation(line: 14, scope: !260, inlinedAt: !928)
!928 = !DILocation(line: 1090, scope: !889)
!929 = !DILocation(line: 14, scope: !260, inlinedAt: !930)
!930 = !DILocation(line: 1105, scope: !889)
!931 = !DILocation(line: 14, scope: !260, inlinedAt: !932)
!932 = !DILocation(line: 1106, scope: !889)
!933 = !DILocation(line: 14, scope: !260, inlinedAt: !934)
!934 = !DILocation(line: 1107, scope: !889)
!935 = !DILocation(line: 410, scope: !275, inlinedAt: !936)
!936 = !DILocation(line: 1110, scope: !889)
!937 = !DILocation(line: 408, scope: !278, inlinedAt: !938)
!938 = !DILocation(line: 578, scope: !280, inlinedAt: !936)
!939 = !DILocation(line: 971, scope: !283, inlinedAt: !940)
!940 = !DILocation(line: 670, scope: !285, inlinedAt: !941)
!941 = !DILocation(line: 91, scope: !287, inlinedAt: !936)
!942 = !DILocation(line: 410, scope: !275, inlinedAt: !943)
!943 = !DILocation(line: 1111, scope: !889)
!944 = !DILocation(line: 408, scope: !278, inlinedAt: !945)
!945 = !DILocation(line: 578, scope: !280, inlinedAt: !943)
!946 = !DILocation(line: 971, scope: !283, inlinedAt: !947)
!947 = !DILocation(line: 670, scope: !285, inlinedAt: !948)
!948 = !DILocation(line: 91, scope: !287, inlinedAt: !943)
!949 = !DILocation(line: 410, scope: !275, inlinedAt: !950)
!950 = !DILocation(line: 1112, scope: !889)
!951 = !DILocation(line: 408, scope: !278, inlinedAt: !952)
!952 = !DILocation(line: 578, scope: !280, inlinedAt: !950)
!953 = !DILocation(line: 971, scope: !283, inlinedAt: !954)
!954 = !DILocation(line: 670, scope: !285, inlinedAt: !955)
!955 = !DILocation(line: 91, scope: !287, inlinedAt: !950)
!956 = !DILocation(line: 410, scope: !275, inlinedAt: !957)
!957 = !DILocation(line: 1114, scope: !889)
!958 = !DILocation(line: 408, scope: !278, inlinedAt: !959)
!959 = !DILocation(line: 578, scope: !280, inlinedAt: !957)
!960 = !DILocation(line: 971, scope: !283, inlinedAt: !961)
!961 = !DILocation(line: 670, scope: !285, inlinedAt: !962)
!962 = !DILocation(line: 91, scope: !287, inlinedAt: !957)
!963 = !DILocation(line: 410, scope: !275, inlinedAt: !964)
!964 = !DILocation(line: 1115, scope: !889)
!965 = !DILocation(line: 408, scope: !278, inlinedAt: !966)
!966 = !DILocation(line: 578, scope: !280, inlinedAt: !964)
!967 = !DILocation(line: 971, scope: !283, inlinedAt: !968)
!968 = !DILocation(line: 670, scope: !285, inlinedAt: !969)
!969 = !DILocation(line: 91, scope: !287, inlinedAt: !964)
!970 = !DILocation(line: 410, scope: !275, inlinedAt: !971)
!971 = !DILocation(line: 1116, scope: !889)
!972 = !DILocation(line: 408, scope: !278, inlinedAt: !973)
!973 = !DILocation(line: 578, scope: !280, inlinedAt: !971)
!974 = !DILocation(line: 971, scope: !283, inlinedAt: !975)
!975 = !DILocation(line: 670, scope: !285, inlinedAt: !976)
!976 = !DILocation(line: 91, scope: !287, inlinedAt: !971)
!977 = !DILocation(line: 410, scope: !275, inlinedAt: !978)
!978 = !DILocation(line: 1118, scope: !889)
!979 = !DILocation(line: 408, scope: !278, inlinedAt: !980)
!980 = !DILocation(line: 578, scope: !280, inlinedAt: !978)
!981 = !DILocation(line: 971, scope: !283, inlinedAt: !982)
!982 = !DILocation(line: 670, scope: !285, inlinedAt: !983)
!983 = !DILocation(line: 91, scope: !287, inlinedAt: !978)
!984 = !DILocation(line: 410, scope: !275, inlinedAt: !985)
!985 = !DILocation(line: 1119, scope: !889)
!986 = !DILocation(line: 408, scope: !278, inlinedAt: !987)
!987 = !DILocation(line: 578, scope: !280, inlinedAt: !985)
!988 = !DILocation(line: 971, scope: !283, inlinedAt: !989)
!989 = !DILocation(line: 670, scope: !285, inlinedAt: !990)
!990 = !DILocation(line: 91, scope: !287, inlinedAt: !985)
!991 = !DILocation(line: 410, scope: !275, inlinedAt: !992)
!992 = !DILocation(line: 1120, scope: !889)
!993 = !DILocation(line: 408, scope: !278, inlinedAt: !994)
!994 = !DILocation(line: 578, scope: !280, inlinedAt: !992)
!995 = !DILocation(line: 971, scope: !283, inlinedAt: !996)
!996 = !DILocation(line: 670, scope: !285, inlinedAt: !997)
!997 = !DILocation(line: 91, scope: !287, inlinedAt: !992)
!998 = !DILocation(line: 1122, scope: !889)
!999 = distinct !DISubprogram(name: "matmul3x3!", linkageName: "julia_matmul3x3!_876", scope: null, file: !3, line: 1071, type: !212, scopeLine: 1071, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !19, retainedNodes: !1000)
!1000 = !{!1001, !1002, !1003, !1004, !1005, !1006, !1007}
!1001 = !DILocalVariable(name: "#self#", arg: 1, scope: !999, file: !3, line: 1071, type: !214)
!1002 = !DILocalVariable(name: "C", arg: 2, scope: !999, file: !3, line: 1071, type: !44)
!1003 = !DILocalVariable(name: "tA", arg: 3, scope: !999, file: !3, line: 1071, type: !105)
!1004 = !DILocalVariable(name: "tB", arg: 4, scope: !999, file: !3, line: 1071, type: !105)
!1005 = !DILocalVariable(name: "A", arg: 5, scope: !999, file: !3, line: 1071, type: !44)
!1006 = !DILocalVariable(name: "B", arg: 6, scope: !999, file: !3, line: 1071, type: !44)
!1007 = !DILocalVariable(name: "_add", arg: 7, scope: !999, file: !3, line: 1071, type: !106)
!1008 = !DILocation(line: 150, scope: !226, inlinedAt: !1009)
!1009 = !DILocation(line: 1074, scope: !999)
!1010 = !{!1011, !69}
!1011 = distinct !{!1011, !1012, !"primal"}
!1012 = distinct !{!1012, !" diff: %"}
!1013 = !{!1014, !72, !73, !74, !75}
!1014 = distinct !{!1014, !1012, !"shadow_0"}
!1015 = !{!1016, !69}
!1016 = distinct !{!1016, !1017, !"primal"}
!1017 = distinct !{!1017, !" diff: %"}
!1018 = !{!1019, !72, !73, !74, !75}
!1019 = distinct !{!1019, !1017, !"shadow_0"}
!1020 = !DILocation(line: 499, scope: !229, inlinedAt: !1021)
!1021 = !DILocation(line: 462, scope: !231, inlinedAt: !1022)
!1022 = !DILocation(line: 458, scope: !233, inlinedAt: !1009)
!1023 = !DILocation(line: 463, scope: !231, inlinedAt: !1022)
!1024 = !DILocation(line: 150, scope: !226, inlinedAt: !1025)
!1025 = !DILocation(line: 1075, scope: !999)
!1026 = !{!1027, !69}
!1027 = distinct !{!1027, !1028, !"primal"}
!1028 = distinct !{!1028, !" diff: %"}
!1029 = !{!1030, !72, !73, !74, !75}
!1030 = distinct !{!1030, !1028, !"shadow_0"}
!1031 = !DILocation(line: 41, scope: !238, inlinedAt: !1025)
!1032 = !{!1033, !72, !73, !75, !69}
!1033 = distinct !{!1033, !1034, !"na_addr13"}
!1034 = distinct !{!1034, !"addr13"}
!1035 = !DILocation(line: 53, scope: !243, inlinedAt: !1036)
!1036 = !DILocation(line: 81, scope: !245, inlinedAt: !1037)
!1037 = !DILocation(line: 232, scope: !247, inlinedAt: !1038)
!1038 = !DILocation(line: 12, scope: !249, inlinedAt: !1025)
!1039 = !DILocation(line: 82, scope: !245, inlinedAt: !1037)
!1040 = !DILocation(line: 107, scope: !252, inlinedAt: !1041)
!1041 = !DILocation(line: 83, scope: !245, inlinedAt: !1037)
!1042 = !DILocation(line: 61, scope: !255, inlinedAt: !1043)
!1043 = !DILocation(line: 88, scope: !245, inlinedAt: !1037)
!1044 = !DILocation(line: 89, scope: !245, inlinedAt: !1037)
!1045 = !DILocation(line: 14, scope: !260, inlinedAt: !1046)
!1046 = !DILocation(line: 1089, scope: !999)
!1047 = !{!1019, !910, !69}
!1048 = !{!1016, !72, !73, !74, !75}
!1049 = !{!1016, !910, !69}
!1050 = !DILocation(line: 14, scope: !260, inlinedAt: !1051)
!1051 = !DILocation(line: 1091, scope: !999)
!1052 = !DILocation(line: 14, scope: !260, inlinedAt: !1053)
!1053 = !DILocation(line: 1090, scope: !999)
!1054 = !DILocation(line: 0, scope: !999)
!1055 = !{!1056, !74}
!1056 = distinct !{!1056, !1057, !"primal"}
!1057 = distinct !{!1057, !" diff: %"}
!1058 = !{!1059, !72, !73, !75, !69}
!1059 = distinct !{!1059, !1057, !"shadow_0"}
!1060 = !DILocation(line: 14, scope: !260, inlinedAt: !1061)
!1061 = !DILocation(line: 1105, scope: !999)
!1062 = !{!1014, !910, !69}
!1063 = !{!1011, !72, !73, !74, !75}
!1064 = !{!1011, !910, !69}
!1065 = !DILocation(line: 14, scope: !260, inlinedAt: !1066)
!1066 = !DILocation(line: 1106, scope: !999)
!1067 = !DILocation(line: 14, scope: !260, inlinedAt: !1068)
!1068 = !DILocation(line: 1107, scope: !999)
!1069 = !{!1070, !74}
!1070 = distinct !{!1070, !1071, !"primal"}
!1071 = distinct !{!1071, !" diff: %"}
!1072 = !{!1073, !72, !73, !75, !69}
!1073 = distinct !{!1073, !1071, !"shadow_0"}
!1074 = !DILocation(line: 410, scope: !275, inlinedAt: !1075)
!1075 = !DILocation(line: 1110, scope: !999)
!1076 = !DILocation(line: 408, scope: !278, inlinedAt: !1077)
!1077 = !DILocation(line: 578, scope: !280, inlinedAt: !1075)
!1078 = !DILocation(line: 971, scope: !283, inlinedAt: !1079)
!1079 = !DILocation(line: 670, scope: !285, inlinedAt: !1080)
!1080 = !DILocation(line: 91, scope: !287, inlinedAt: !1075)
!1081 = !{!1030, !910, !69}
!1082 = !{!1027, !72, !73, !74, !75}
!1083 = !{!1027, !910, !69}
!1084 = !{!1085, !74}
!1085 = distinct !{!1085, !1086, !"primal"}
!1086 = distinct !{!1086, !" diff: %"}
!1087 = !{!1088, !910, !72, !73, !75, !69}
!1088 = distinct !{!1088, !1086, !"shadow_0"}
!1089 = !DILocation(line: 410, scope: !275, inlinedAt: !1090)
!1090 = !DILocation(line: 1111, scope: !999)
!1091 = !DILocation(line: 408, scope: !278, inlinedAt: !1092)
!1092 = !DILocation(line: 578, scope: !280, inlinedAt: !1090)
!1093 = !DILocation(line: 971, scope: !283, inlinedAt: !1094)
!1094 = !DILocation(line: 670, scope: !285, inlinedAt: !1095)
!1095 = !DILocation(line: 91, scope: !287, inlinedAt: !1090)
!1096 = !DILocation(line: 410, scope: !275, inlinedAt: !1097)
!1097 = !DILocation(line: 1112, scope: !999)
!1098 = !DILocation(line: 408, scope: !278, inlinedAt: !1099)
!1099 = !DILocation(line: 578, scope: !280, inlinedAt: !1097)
!1100 = !DILocation(line: 971, scope: !283, inlinedAt: !1101)
!1101 = !DILocation(line: 670, scope: !285, inlinedAt: !1102)
!1102 = !DILocation(line: 91, scope: !287, inlinedAt: !1097)
!1103 = !DILocation(line: 410, scope: !275, inlinedAt: !1104)
!1104 = !DILocation(line: 1114, scope: !999)
!1105 = !DILocation(line: 408, scope: !278, inlinedAt: !1106)
!1106 = !DILocation(line: 578, scope: !280, inlinedAt: !1104)
!1107 = !DILocation(line: 971, scope: !283, inlinedAt: !1108)
!1108 = !DILocation(line: 670, scope: !285, inlinedAt: !1109)
!1109 = !DILocation(line: 91, scope: !287, inlinedAt: !1104)
!1110 = !DILocation(line: 410, scope: !275, inlinedAt: !1111)
!1111 = !DILocation(line: 1115, scope: !999)
!1112 = !DILocation(line: 408, scope: !278, inlinedAt: !1113)
!1113 = !DILocation(line: 578, scope: !280, inlinedAt: !1111)
!1114 = !DILocation(line: 971, scope: !283, inlinedAt: !1115)
!1115 = !DILocation(line: 670, scope: !285, inlinedAt: !1116)
!1116 = !DILocation(line: 91, scope: !287, inlinedAt: !1111)
!1117 = !DILocation(line: 410, scope: !275, inlinedAt: !1118)
!1118 = !DILocation(line: 1116, scope: !999)
!1119 = !DILocation(line: 408, scope: !278, inlinedAt: !1120)
!1120 = !DILocation(line: 578, scope: !280, inlinedAt: !1118)
!1121 = !DILocation(line: 971, scope: !283, inlinedAt: !1122)
!1122 = !DILocation(line: 670, scope: !285, inlinedAt: !1123)
!1123 = !DILocation(line: 91, scope: !287, inlinedAt: !1118)
!1124 = !DILocation(line: 410, scope: !275, inlinedAt: !1125)
!1125 = !DILocation(line: 1118, scope: !999)
!1126 = !DILocation(line: 408, scope: !278, inlinedAt: !1127)
!1127 = !DILocation(line: 578, scope: !280, inlinedAt: !1125)
!1128 = !DILocation(line: 971, scope: !283, inlinedAt: !1129)
!1129 = !DILocation(line: 670, scope: !285, inlinedAt: !1130)
!1130 = !DILocation(line: 91, scope: !287, inlinedAt: !1125)
!1131 = !DILocation(line: 410, scope: !275, inlinedAt: !1132)
!1132 = !DILocation(line: 1119, scope: !999)
!1133 = !DILocation(line: 408, scope: !278, inlinedAt: !1134)
!1134 = !DILocation(line: 578, scope: !280, inlinedAt: !1132)
!1135 = !DILocation(line: 971, scope: !283, inlinedAt: !1136)
!1136 = !DILocation(line: 670, scope: !285, inlinedAt: !1137)
!1137 = !DILocation(line: 91, scope: !287, inlinedAt: !1132)
!1138 = !DILocation(line: 410, scope: !275, inlinedAt: !1139)
!1139 = !DILocation(line: 1120, scope: !999)
!1140 = !DILocation(line: 408, scope: !278, inlinedAt: !1141)
!1141 = !DILocation(line: 578, scope: !280, inlinedAt: !1139)
!1142 = !DILocation(line: 971, scope: !283, inlinedAt: !1143)
!1143 = !DILocation(line: 670, scope: !285, inlinedAt: !1144)
!1144 = !DILocation(line: 91, scope: !287, inlinedAt: !1139)
!1145 = !DILocation(line: 1122, scope: !999)
!1146 = !{!1088, !74}
!1147 = !{!1085, !910, !72, !73, !75, !69}
!1148 = !{!1073, !74}
!1149 = !{!1070, !72, !73, !75, !69}
!1150 = !{!1059, !74}
!1151 = !{!1056, !72, !73, !75, !69}
!1152 = distinct !DISubprogram(name: "gemm!", linkageName: "julia_gemm!_880", scope: null, file: !37, line: 1505, type: !346, scopeLine: 1505, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !1153)
!1153 = !{!1154, !1155, !1156, !1157, !1158, !1159, !1160, !1161}
!1154 = !DILocalVariable(name: "#self#", arg: 1, scope: !1152, file: !37, line: 1505, type: !348)
!1155 = !DILocalVariable(name: "transA", arg: 2, scope: !1152, file: !37, line: 1505, type: !105)
!1156 = !DILocalVariable(name: "transB", arg: 3, scope: !1152, file: !37, line: 1505, type: !105)
!1157 = !DILocalVariable(name: "alpha", arg: 4, scope: !1152, file: !37, line: 1505, type: !349)
!1158 = !DILocalVariable(name: "A", arg: 5, scope: !1152, file: !37, line: 1505, type: !44)
!1159 = !DILocalVariable(name: "B", arg: 6, scope: !1152, file: !37, line: 1505, type: !44)
!1160 = !DILocalVariable(name: "beta", arg: 7, scope: !1152, file: !37, line: 1505, type: !349)
!1161 = !DILocalVariable(name: "C", arg: 8, scope: !1152, file: !37, line: 1505, type: !44)
!1162 = !DILocation(line: 0, scope: !1152)
!1163 = !DILocation(line: 148, scope: !361, inlinedAt: !1164)
!1164 = !DILocation(line: 1514, scope: !1152)
!1165 = !DILocation(line: 148, scope: !361, inlinedAt: !1166)
!1166 = !DILocation(line: 1515, scope: !1152)
!1167 = !DILocation(line: 148, scope: !361, inlinedAt: !1168)
!1168 = !DILocation(line: 1516, scope: !1152)
!1169 = !DILocation(line: 148, scope: !361, inlinedAt: !1170)
!1170 = !DILocation(line: 1517, scope: !1152)
!1171 = !DILocation(line: 499, scope: !370, inlinedAt: !1172)
!1172 = !DILocation(line: 269, scope: !372, inlinedAt: !1173)
!1173 = !DILocation(line: 1518, scope: !1152)
!1174 = !DILocation(line: 148, scope: !361, inlinedAt: !1173)
!1175 = !DILocation(line: 8, scope: !380, inlinedAt: !1176)
!1176 = !DILocation(line: 104, scope: !383, inlinedAt: !1177)
!1177 = !DILocation(line: 492, scope: !386, inlinedAt: !1178)
!1178 = !DILocation(line: 1524, scope: !1152)
!1179 = !{!1180, !72, !73, !75, !69}
!1180 = distinct !{!1180, !1181, !"na_addr13"}
!1181 = distinct !{!1181, !"addr13"}
!1182 = !DILocation(line: 150, scope: !361, inlinedAt: !1183)
!1183 = !DILocation(line: 173, scope: !393, inlinedAt: !1184)
!1184 = !DILocation(line: 174, scope: !396, inlinedAt: !1178)
!1185 = !DILocation(line: 83, scope: !398, inlinedAt: !1186)
!1186 = !DILocation(line: 510, scope: !401, inlinedAt: !1178)
!1187 = !DILocation(line: 575, scope: !403, inlinedAt: !1186)
!1188 = !DILocation(line: 65, scope: !405, inlinedAt: !1178)
!1189 = !DILocation(line: 1533, scope: !1152)
!1190 = !DILocation(line: 150, scope: !361, inlinedAt: !1191)
!1191 = !DILocation(line: 1519, scope: !1152)
!1192 = !DILocation(line: 41, scope: !411, inlinedAt: !1191)
!1193 = !DILocation(line: 53, scope: !413, inlinedAt: !1194)
!1194 = !DILocation(line: 81, scope: !415, inlinedAt: !1195)
!1195 = !DILocation(line: 232, scope: !417, inlinedAt: !1196)
!1196 = !DILocation(line: 12, scope: !419, inlinedAt: !1191)
!1197 = !DILocation(line: 82, scope: !415, inlinedAt: !1195)
!1198 = !DILocation(line: 107, scope: !422, inlinedAt: !1199)
!1199 = !DILocation(line: 83, scope: !415, inlinedAt: !1195)
!1200 = !DILocation(line: 61, scope: !425, inlinedAt: !1201)
!1201 = !DILocation(line: 88, scope: !415, inlinedAt: !1195)
!1202 = !DILocation(line: 89, scope: !415, inlinedAt: !1195)
!1203 = distinct !DISubprogram(name: "gemm!", linkageName: "julia_gemm!_880", scope: null, file: !37, line: 1505, type: !346, scopeLine: 1505, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !36, retainedNodes: !1204)
!1204 = !{!1205, !1206, !1207, !1208, !1209, !1210, !1211, !1212}
!1205 = !DILocalVariable(name: "#self#", arg: 1, scope: !1203, file: !37, line: 1505, type: !348)
!1206 = !DILocalVariable(name: "transA", arg: 2, scope: !1203, file: !37, line: 1505, type: !105)
!1207 = !DILocalVariable(name: "transB", arg: 3, scope: !1203, file: !37, line: 1505, type: !105)
!1208 = !DILocalVariable(name: "alpha", arg: 4, scope: !1203, file: !37, line: 1505, type: !349)
!1209 = !DILocalVariable(name: "A", arg: 5, scope: !1203, file: !37, line: 1505, type: !44)
!1210 = !DILocalVariable(name: "B", arg: 6, scope: !1203, file: !37, line: 1505, type: !44)
!1211 = !DILocalVariable(name: "beta", arg: 7, scope: !1203, file: !37, line: 1505, type: !349)
!1212 = !DILocalVariable(name: "C", arg: 8, scope: !1203, file: !37, line: 1505, type: !44)
!1213 = !DILocation(line: 148, scope: !361, inlinedAt: !1214)
!1214 = !DILocation(line: 1514, scope: !1203)
!1215 = !{!1216, !69}
!1216 = distinct !{!1216, !1217, !"primal"}
!1217 = distinct !{!1217, !" diff: %"}
!1218 = !{!1219, !72, !73, !74, !75}
!1219 = distinct !{!1219, !1217, !"shadow_0"}
!1220 = !DILocation(line: 148, scope: !361, inlinedAt: !1221)
!1221 = !DILocation(line: 1515, scope: !1203)
!1222 = !DILocation(line: 148, scope: !361, inlinedAt: !1223)
!1223 = !DILocation(line: 1516, scope: !1203)
!1224 = !{!1225, !69}
!1225 = distinct !{!1225, !1226, !"primal"}
!1226 = distinct !{!1226, !" diff: %"}
!1227 = !{!1228, !72, !73, !74, !75}
!1228 = distinct !{!1228, !1226, !"shadow_0"}
!1229 = !DILocation(line: 148, scope: !361, inlinedAt: !1230)
!1230 = !DILocation(line: 1517, scope: !1203)
!1231 = !DILocation(line: 499, scope: !370, inlinedAt: !1232)
!1232 = !DILocation(line: 269, scope: !372, inlinedAt: !1233)
!1233 = !DILocation(line: 1518, scope: !1203)
!1234 = !DILocation(line: 148, scope: !361, inlinedAt: !1233)
!1235 = !{!1236, !69}
!1236 = distinct !{!1236, !1237, !"primal"}
!1237 = distinct !{!1237, !" diff: %"}
!1238 = !{!1239, !72, !73, !74, !75}
!1239 = distinct !{!1239, !1237, !"shadow_0"}
!1240 = !DILocation(line: 8, scope: !380, inlinedAt: !1241)
!1241 = !DILocation(line: 104, scope: !383, inlinedAt: !1242)
!1242 = !DILocation(line: 492, scope: !386, inlinedAt: !1243)
!1243 = !DILocation(line: 1524, scope: !1203)
!1244 = !{!1245, !72, !73, !75, !69}
!1245 = distinct !{!1245, !1246, !"na_addr13"}
!1246 = distinct !{!1246, !"addr13"}
!1247 = !DILocation(line: 150, scope: !361, inlinedAt: !1248)
!1248 = !DILocation(line: 173, scope: !393, inlinedAt: !1249)
!1249 = !DILocation(line: 174, scope: !396, inlinedAt: !1243)
!1250 = !DILocation(line: 83, scope: !398, inlinedAt: !1251)
!1251 = !DILocation(line: 510, scope: !401, inlinedAt: !1243)
!1252 = !DILocation(line: 575, scope: !403, inlinedAt: !1251)
!1253 = !DILocation(line: 65, scope: !405, inlinedAt: !1243)
!1254 = !DILocation(line: 1533, scope: !1203)
!1255 = !DILocation(line: 150, scope: !361, inlinedAt: !1256)
!1256 = !DILocation(line: 1519, scope: !1203)
!1257 = !DILocation(line: 41, scope: !411, inlinedAt: !1256)
!1258 = !DILocation(line: 53, scope: !413, inlinedAt: !1259)
!1259 = !DILocation(line: 81, scope: !415, inlinedAt: !1260)
!1260 = !DILocation(line: 232, scope: !417, inlinedAt: !1261)
!1261 = !DILocation(line: 12, scope: !419, inlinedAt: !1256)
!1262 = !DILocation(line: 82, scope: !415, inlinedAt: !1260)
!1263 = !DILocation(line: 107, scope: !422, inlinedAt: !1264)
!1264 = !DILocation(line: 83, scope: !415, inlinedAt: !1260)
!1265 = !DILocation(line: 61, scope: !425, inlinedAt: !1266)
!1266 = !DILocation(line: 88, scope: !415, inlinedAt: !1260)
!1267 = !DILocation(line: 89, scope: !415, inlinedAt: !1260)
