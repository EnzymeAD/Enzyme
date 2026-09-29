; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -ge 14 ] ; then %opt < %s %loadEnzyme -opaque-pointers -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -jump-threading -adce -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 14 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -passes="enzyme,function(mem2reg,early-cse,instsimplify,jump-threading,adce)" -enzyme-preopt=false -S | FileCheck %s ; fi

; ModuleID = '../examples/big/big_inlined_correctness.cpp'
source_filename = "../examples/big/big_inlined_correctness.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Prod = type { ptr, double }

declare i32 @dgemm_(ptr nocapture noundef readonly %transa_t, ptr nocapture noundef readonly %transb_t, ptr nocapture noundef readonly %m, ptr nocapture noundef readonly %n, ptr nocapture noundef readonly %k, ptr nocapture noundef readonly %alpha, ptr nocapture noundef readonly %a, ptr nocapture noundef readonly %lda, ptr nocapture noundef readonly %b, ptr nocapture noundef readonly %ldb, ptr nocapture noundef readonly %beta, ptr nocapture noundef %c, ptr nocapture noundef readonly %ldc, i32, i32)

; Function Attrs: mustprogress noinline nounwind uwtable
define dso_local void @_Z3mulR4ProdPd(ptr nocapture noundef nonnull align 8 dereferenceable(16) %P, ptr noalias nocapture noundef readonly %rhs) {
entry:
  %N = alloca i8, align 1
  %ten = alloca i32, align 4
  %one = alloca double, align 8
  %zero = alloca double, align 8
  %calloc = call dereferenceable_or_null(32) ptr @calloc(i64 1, i64 32)
  store i8 78, ptr %N, align 1
  store i32 2, ptr %ten, align 4
  store double 1.000000e+00, ptr %one, align 8
  store double 0.000000e+00, ptr %zero, align 8
  %call1 = call i32 @dgemm_(ptr noundef nonnull %N, ptr noundef nonnull %N, ptr noundef nonnull %ten, ptr noundef nonnull %ten, ptr noundef nonnull %ten, ptr noundef nonnull %one, ptr noundef %rhs, ptr noundef nonnull %ten, ptr noundef %rhs, ptr noundef nonnull %ten, ptr noundef nonnull %one, ptr noundef %calloc, ptr noundef nonnull %ten, i32 1, i32 1)
  %0 = load ptr, ptr %P, align 8
  %call2 = call i32 @dgemm_(ptr noundef nonnull %N, ptr noundef nonnull %N, ptr noundef nonnull %ten, ptr noundef nonnull %ten, ptr noundef nonnull %ten, ptr noundef nonnull %one, ptr noundef %calloc, ptr noundef nonnull %ten, ptr noundef %rhs, ptr noundef nonnull %ten, ptr noundef nonnull %zero, ptr noundef %0, ptr noundef nonnull %ten, i32 1, i32 1)
  %alpha = getelementptr inbounds %struct.Prod, ptr %P, i64 0, i32 1
  store double 0.000000e+00, ptr %alpha, align 8
  ret void
}

declare noalias ptr @malloc(i64)

; Function Attrs: mustprogress nounwind uwtable
define dso_local double @_Z8simulatePd(ptr nocapture noundef readonly %P) {
entry:
  %M = alloca %struct.Prod, align 8
  %call = tail call noalias dereferenceable_or_null(32) ptr @malloc(i64 noundef 32) 
  store ptr %call, ptr %M, align 8
  %alpha = getelementptr inbounds %struct.Prod, ptr %M, i64 0, i32 1
  store double 1.000000e+00, ptr %alpha, align 8
  call void @_Z3mulR4ProdPd(ptr noundef nonnull align 8 dereferenceable(16) %M, ptr noundef %P)
  %0 = load ptr, ptr %M, align 8
  %1 = load double, ptr %0, align 8
  ret double %1
}

define void @caller(ptr %A, ptr %Adup) {
entry:
  call void (...) @_Z17__enzyme_autodiffz(ptr noundef nonnull @_Z8simulatePd, metadata !"enzyme_dup", ptr noundef nonnull %A, ptr noundef nonnull %Adup)
  ret void
}

declare void @_Z17__enzyme_autodiffz(...) 

declare noalias noundef ptr @calloc(i64 noundef, i64 noundef)

; we must actually save or set the matmul
; CHECK: define internal void @diffe_Z3mulR4ProdPd(ptr nocapture align 8 %P, ptr nocapture align 8 %"P'", ptr noalias nocapture readonly %rhs, ptr nocapture %"rhs'", { ptr, ptr, ptr } %tapeArg)
; CHECK-NEXT: invertentry:
; CHECK-NEXT:   %byref.transpose.transb = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1.04 = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.0 = alloca i32, align 4
; CHECK-NEXT:   %byref.constant.int.05 = alloca i32, align 4
; CHECK-NEXT:   %byref.constant.fp.1.06 = alloca double, align 8
; CHECK-NEXT:   %[[i0:.+]] = alloca i32, align 4
; CHECK-NEXT:   %byref.transpose.transb10 = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1.013 = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa15 = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1.018 = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G19 = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.020 = alloca i32, align 4
; CHECK-NEXT:   %byref.constant.int.021 = alloca i32, align 4
; CHECK-NEXT:   %byref.constant.fp.1.022 = alloca double, align 8
; CHECK-NEXT:   %[[i1:.+]] = alloca i32, align 4
; CHECK-NEXT:   %malloccall3 = alloca double, i64 1, align 8
; CHECK-NEXT:   %malloccall = alloca i8, i64 1, align 1
; CHECK-NEXT:   %malloccall2 = alloca double, i64 1, align 8
; CHECK-NEXT:   %malloccall1 = alloca i32, i64 1, align 4
; CHECK-NEXT:   %"calloc'mi" = extractvalue { ptr, ptr, ptr } %tapeArg, 1
; CHECK-NEXT:   %calloc = extractvalue { ptr, ptr, ptr } %tapeArg, 2
; CHECK-NEXT:   store i8 78, ptr %malloccall, align 1
; CHECK-NEXT:   store i32 2, ptr %malloccall1, align 4
; CHECK-NEXT:   store double 1.000000e+00, ptr %malloccall2, align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %malloccall3, align 8
; CHECK-NEXT:   %"'il_phi" = extractvalue { ptr, ptr, ptr } %tapeArg, 0
; CHECK-NEXT:   %"alpha'ipg" = getelementptr inbounds %struct.Prod, ptr %"P'", i64 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, ptr %"alpha'ipg", align 8
; CHECK-NEXT:   %ld.transb = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i2:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-NEXT:   %[[i3:.+]] = select i1 %[[i2]], i8 116, i8 78
; CHECK-NEXT:   %[[i4:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-NEXT:   %[[i5:.+]] = select i1 %[[i4]], i8 84, i8 %[[i3]]
; CHECK-NEXT:   %[[i6:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-NEXT:   %[[i7:.+]] = select i1 %[[i6]], i8 110, i8 %[[i5]]
; CHECK-NEXT:   %[[i8:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-NEXT:   %[[i9:.+]] = select i1 %[[i8]], i8 78, i8 %[[i7]]
; CHECK-NEXT:   store i8 %[[i9]], ptr %byref.transpose.transb, align 1
; CHECK-NEXT:   %ld.row.trans = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i10:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[i11:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[i12:.+]] = or i1 %[[i11]], %[[i10]]
; CHECK-NEXT:   %[[i13:.+]] = select i1 %[[i12]], ptr %byref.transpose.transb, ptr %malloccall
; CHECK-NEXT:   %[[i14:.+]] = select i1 %[[i12]], ptr %"'il_phi", ptr %rhs
; CHECK-NEXT:   %[[i15:.+]] = select i1 %[[i12]], ptr %rhs, ptr %"'il_phi"
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   call void @dgemm_(ptr %malloccall, ptr %[[i13]], ptr %malloccall1, ptr %malloccall1, ptr %malloccall1, ptr %malloccall2, ptr %[[i14]], ptr %malloccall1, ptr %[[i15]], ptr %malloccall1, ptr %byref.constant.fp.1.0, ptr %"calloc'mi", ptr %malloccall1, i32 1, i32 1)
; CHECK-NEXT:   %ld.transa = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i16:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-NEXT:   %[[i17:.+]] = select i1 %[[i16]], i8 116, i8 78
; CHECK-NEXT:   %[[i18:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-NEXT:   %[[i19:.+]] = select i1 %[[i18]], i8 84, i8 %[[i17]]
; CHECK-NEXT:   %[[i20:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-NEXT:   %[[i21:.+]] = select i1 %[[i20]], i8 110, i8 %[[i19]]
; CHECK-NEXT:   %[[i22:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-NEXT:   %[[i23:.+]] = select i1 %[[i22]], i8 78, i8 %[[i21]]
; CHECK-NEXT:   store i8 %[[i23]], ptr %byref.transpose.transa, align 1
; CHECK-NEXT:   %ld.row.trans2 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i24:.+]] = icmp eq i8 %ld.row.trans2, 110
; CHECK-NEXT:   %[[i25:.+]] = icmp eq i8 %ld.row.trans2, 78
; CHECK-NEXT:   %[[i26:.+]] = or i1 %[[i25]], %[[i24]]
; CHECK-NEXT:   %[[i27:.+]] = select i1 %[[i26]], ptr %byref.transpose.transa, ptr %malloccall
; CHECK-NEXT:   %[[i28:.+]] = select i1 %[[i26]], ptr %calloc, ptr %"'il_phi"
; CHECK-NEXT:   %[[i29:.+]] = select i1 %[[i26]], ptr %"'il_phi", ptr %calloc
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.04, align 8
; CHECK-NEXT:   call void @dgemm_(ptr %[[i27]], ptr %malloccall, ptr %malloccall1, ptr %malloccall1, ptr %malloccall1, ptr %malloccall2, ptr %[[i28]], ptr %malloccall1, ptr %[[i29]], ptr %malloccall1, ptr %byref.constant.fp.1.04, ptr %"rhs'", ptr %malloccall1, i32 1, i32 1)
; CHECK-NEXT:   store i8 71, ptr %byref.constant.char.G, align 1
; CHECK-NEXT:   store i32 0, ptr %byref.constant.int.0, align 4
; CHECK-NEXT:   store i32 0, ptr %byref.constant.int.05, align 4
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.06, align 8
; CHECK-NEXT:   call void @dlascl_(ptr %byref.constant.char.G, ptr %byref.constant.int.0, ptr %byref.constant.int.05, ptr %byref.constant.fp.1.06, ptr %malloccall3, ptr %malloccall1, ptr %malloccall1, ptr %"'il_phi", ptr %malloccall1, ptr %[[i0]], i32 1)
; CHECK-NEXT:   %ld.transb9 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i30:.+]] = icmp eq i8 %ld.transb9, 110
; CHECK-NEXT:   %[[i31:.+]] = select i1 %[[i30]], i8 116, i8 78
; CHECK-NEXT:   %[[i32:.+]] = icmp eq i8 %ld.transb9, 78
; CHECK-NEXT:   %[[i33:.+]] = select i1 %[[i32]], i8 84, i8 %[[i31]]
; CHECK-NEXT:   %[[i34:.+]] = icmp eq i8 %ld.transb9, 116
; CHECK-NEXT:   %[[i35:.+]] = select i1 %[[i34]], i8 110, i8 %[[i33]]
; CHECK-NEXT:   %[[i36:.+]] = icmp eq i8 %ld.transb9, 84
; CHECK-NEXT:   %[[i37:.+]] = select i1 %[[i36]], i8 78, i8 %[[i35]]
; CHECK-NEXT:   store i8 %[[i37]], ptr %byref.transpose.transb10, align 1
; CHECK-NEXT:   %ld.row.trans11 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i38:.+]] = icmp eq i8 %ld.row.trans11, 110
; CHECK-NEXT:   %[[i39:.+]] = icmp eq i8 %ld.row.trans11, 78
; CHECK-NEXT:   %[[i40:.+]] = or i1 %[[i39]], %[[i38]]
; CHECK-NEXT:   %[[i41:.+]] = select i1 %[[i40]], ptr %byref.transpose.transb10, ptr %malloccall
; CHECK-NEXT:   %[[i42:.+]] = select i1 %[[i40]], ptr %"calloc'mi", ptr %rhs
; CHECK-NEXT:   %[[i43:.+]] = select i1 %[[i40]], ptr %rhs, ptr %"calloc'mi"
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.013, align 8
; CHECK-NEXT:   call void @dgemm_(ptr %malloccall, ptr %[[i41]], ptr %malloccall1, ptr %malloccall1, ptr %malloccall1, ptr %malloccall2, ptr %[[i42]], ptr %malloccall1, ptr %[[i43]], ptr %malloccall1, ptr %byref.constant.fp.1.013, ptr %"rhs'", ptr %malloccall1, i32 1, i32 1)
; CHECK-NEXT:   %ld.transa14 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i44:.+]] = icmp eq i8 %ld.transa14, 110
; CHECK-NEXT:   %[[i45:.+]] = select i1 %[[i44]], i8 116, i8 78
; CHECK-NEXT:   %[[i46:.+]] = icmp eq i8 %ld.transa14, 78
; CHECK-NEXT:   %[[i47:.+]] = select i1 %[[i46]], i8 84, i8 %[[i45]]
; CHECK-NEXT:   %[[i48:.+]] = icmp eq i8 %ld.transa14, 116
; CHECK-NEXT:   %[[i49:.+]] = select i1 %[[i48]], i8 110, i8 %[[i47]]
; CHECK-NEXT:   %[[i50:.+]] = icmp eq i8 %ld.transa14, 84
; CHECK-NEXT:   %[[i51:.+]] = select i1 %[[i50]], i8 78, i8 %[[i49]]
; CHECK-NEXT:   store i8 %[[i51]], ptr %byref.transpose.transa15, align 1
; CHECK-NEXT:   %ld.row.trans16 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i52:.+]] = icmp eq i8 %ld.row.trans16, 110
; CHECK-NEXT:   %[[i53:.+]] = icmp eq i8 %ld.row.trans16, 78
; CHECK-NEXT:   %[[i54:.+]] = or i1 %[[i53]], %[[i52]]
; CHECK-NEXT:   %[[i55:.+]] = select i1 %[[i54]], ptr %byref.transpose.transa15, ptr %malloccall
; CHECK-NEXT:   %[[i56:.+]] = select i1 %[[i54]], ptr %rhs, ptr %"calloc'mi"
; CHECK-NEXT:   %[[i57:.+]] = select i1 %[[i54]], ptr %"calloc'mi", ptr %rhs
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.018, align 8
; CHECK-NEXT:   call void @dgemm_(ptr %[[i55]], ptr %malloccall, ptr %malloccall1, ptr %malloccall1, ptr %malloccall1, ptr %malloccall2, ptr %[[i56]], ptr %malloccall1, ptr %[[i57]], ptr %malloccall1, ptr %byref.constant.fp.1.018, ptr %"rhs'", ptr %malloccall1, i32 1, i32 1)
; CHECK-NEXT:   store i8 71, ptr %byref.constant.char.G19, align 1
; CHECK-NEXT:   store i32 0, ptr %byref.constant.int.020, align 4
; CHECK-NEXT:   store i32 0, ptr %byref.constant.int.021, align 4
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.022, align 8
; CHECK-NEXT:   call void @dlascl_(ptr %byref.constant.char.G19, ptr %byref.constant.int.020, ptr %byref.constant.int.021, ptr %byref.constant.fp.1.022, ptr %malloccall2, ptr %malloccall1, ptr %malloccall1, ptr %"calloc'mi", ptr %malloccall1, ptr %[[i1]], i32 1)
; CHECK-NEXT:   call void @free(ptr nonnull %"calloc'mi")
; CHECK-NEXT:   call void @free(ptr %calloc)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
