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
; CHECK: define internal void @diffe_Z3mulR4ProdPd(ptr nocapture align 8 dereferenceable(16) %P, ptr nocapture align 8 %"P'", ptr noalias nocapture readonly %rhs, ptr nocapture %"rhs'", { ptr, ptr, ptr, ptr } %tapeArg)
; CHECK-NEXT: invertentry:
; CHECK-NEXT:   %byref.transpose.transb = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1.05 = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.0 = alloca i32, align 4
; CHECK-NEXT:   %byref.constant.int.06 = alloca i32, align 4
; CHECK-NEXT:   %byref.constant.fp.1.07 = alloca double, align 8
; CHECK-NEXT:   %[[i0:.+]] = alloca i32, align 4
; CHECK-NEXT:   %byref.transpose.transb11 = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1.014 = alloca double, align 8
; CHECK-NEXT:   %byref.transpose.transa16 = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.fp.1.019 = alloca double, align 8
; CHECK-NEXT:   %byref.constant.char.G20 = alloca i8, align 1
; CHECK-NEXT:   %byref.constant.int.021 = alloca i32, align 4
; CHECK-NEXT:   %byref.constant.int.022 = alloca i32, align 4
; CHECK-NEXT:   %byref.constant.fp.1.023 = alloca double, align 8
; CHECK-NEXT:   %[[i1:.+]] = alloca i32, align 4
; CHECK-NEXT:   %malloccall3 = alloca double, i64 1, align 8
; CHECK-NEXT:   %malloccall = alloca i8, i64 1, align 1
; CHECK-NEXT:   %malloccall2 = alloca double, i64 1, align 8
; CHECK-NEXT:   %malloccall1 = alloca i32, i64 1, align 4
; CHECK-NEXT:   %"calloc'mi" = extractvalue { ptr, ptr, ptr, ptr } %tapeArg, 2
; CHECK-NEXT:   %calloc = extractvalue { ptr, ptr, ptr, ptr } %tapeArg, 3
; CHECK-NEXT:   store i8 78, ptr %malloccall, align 1
; CHECK-NEXT:   store i32 2, ptr %malloccall1, align 4
; CHECK-NEXT:   store double 1.000000e+00, ptr %malloccall2, align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %malloccall3, align 8
; CHECK-NEXT:   %"'il_phi" = extractvalue { ptr, ptr, ptr, ptr } %tapeArg, 1
; CHECK-NEXT:   %[[i2:.+]] = extractvalue { ptr, ptr, ptr, ptr } %tapeArg, 0
; CHECK-NEXT:   %"alpha'ipg" = getelementptr inbounds %struct.Prod, ptr %"P'", i64 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, ptr %"alpha'ipg", align 8
; CHECK-NEXT:   %ld.transb = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i3:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-NEXT:   %[[i4:.+]] = select i1 %[[i3]], i8 116, i8 78
; CHECK-NEXT:   %[[i5:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-NEXT:   %[[i6:.+]] = select i1 %[[i5]], i8 84, i8 %[[i4]]
; CHECK-NEXT:   %[[i7:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-NEXT:   %[[i8:.+]] = select i1 %[[i7]], i8 110, i8 %[[i6]]
; CHECK-NEXT:   %[[i9:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-NEXT:   %[[i10:.+]] = select i1 %[[i9]], i8 78, i8 %[[i8]]
; CHECK-NEXT:   store i8 %[[i10]], ptr %byref.transpose.transb, align 1
; CHECK-NEXT:   %ld.row.trans = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i11:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-NEXT:   %[[i12:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[i13:.+]] = or i1 %[[i12]], %[[i11]]
; CHECK-NEXT:   %[[i14:.+]] = select i1 %[[i13]], ptr %byref.transpose.transb, ptr %malloccall
; CHECK-NEXT:   %[[i15:.+]] = select i1 %[[i13]], ptr %"'il_phi", ptr %rhs
; CHECK-NEXT:   %[[i16:.+]] = select i1 %[[i13]], ptr %rhs, ptr %"'il_phi"
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.0, align 8
; CHECK-NEXT:   call void @dgemm_(ptr %malloccall, ptr %[[i14]], ptr %malloccall1, ptr %malloccall1, ptr %malloccall1, ptr %malloccall2, ptr %[[i15]], ptr %malloccall1, ptr %[[i16]], ptr %malloccall1, ptr %byref.constant.fp.1.0, ptr %"calloc'mi", ptr %malloccall1, i32 1, i32 1)
; CHECK-NEXT:   %ld.transa = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i17:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-NEXT:   %[[i18:.+]] = select i1 %[[i17]], i8 116, i8 78
; CHECK-NEXT:   %[[i19:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-NEXT:   %[[i20:.+]] = select i1 %[[i19]], i8 84, i8 %[[i18]]
; CHECK-NEXT:   %[[i21:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-NEXT:   %[[i22:.+]] = select i1 %[[i21]], i8 110, i8 %[[i20]]
; CHECK-NEXT:   %[[i23:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-NEXT:   %[[i24:.+]] = select i1 %[[i23]], i8 78, i8 %[[i22]]
; CHECK-NEXT:   store i8 %[[i24]], ptr %byref.transpose.transa, align 1
; CHECK-NEXT:   %ld.row.trans2 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i25:.+]] = icmp eq i8 %ld.row.trans2, 110
; CHECK-NEXT:   %[[i26:.+]] = icmp eq i8 %ld.row.trans2, 78
; CHECK-NEXT:   %[[i27:.+]] = or i1 %[[i26]], %[[i25]]
; CHECK-NEXT:   %[[i28:.+]] = select i1 %[[i27]], ptr %byref.transpose.transa, ptr %malloccall
; CHECK-NEXT:   %[[i29:.+]] = select i1 %[[i27]], ptr %[[i2]], ptr %"'il_phi"
; CHECK-NEXT:   %[[i30:.+]] = select i1 %[[i27]], ptr %"'il_phi", ptr %[[i2]]
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.05, align 8
; CHECK-NEXT:   call void @dgemm_(ptr %[[i28]], ptr %malloccall, ptr %malloccall1, ptr %malloccall1, ptr %malloccall1, ptr %malloccall2, ptr %[[i29]], ptr %malloccall1, ptr %[[i30]], ptr %malloccall1, ptr %byref.constant.fp.1.05, ptr %"rhs'", ptr %malloccall1, i32 1, i32 1)
; CHECK-NEXT:   store i8 71, ptr %byref.constant.char.G, align 1
; CHECK-NEXT:   store i32 0, ptr %byref.constant.int.0, align 4
; CHECK-NEXT:   store i32 0, ptr %byref.constant.int.06, align 4
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.07, align 8
; CHECK-NEXT:   call void @dlascl_(ptr %byref.constant.char.G, ptr %byref.constant.int.0, ptr %byref.constant.int.06, ptr %byref.constant.fp.1.07, ptr %malloccall3, ptr %malloccall1, ptr %malloccall1, ptr %"'il_phi", ptr %malloccall1, ptr %[[i0]], i32 1)
; CHECK-NEXT:   tail call void @free(ptr nonnull %[[i2]])
; CHECK-NEXT:   %ld.transb10 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i31:.+]] = icmp eq i8 %ld.transb10, 110
; CHECK-NEXT:   %[[i32:.+]] = select i1 %[[i31]], i8 116, i8 78
; CHECK-NEXT:   %[[i33:.+]] = icmp eq i8 %ld.transb10, 78
; CHECK-NEXT:   %[[i34:.+]] = select i1 %[[i33]], i8 84, i8 %[[i32]]
; CHECK-NEXT:   %[[i35:.+]] = icmp eq i8 %ld.transb10, 116
; CHECK-NEXT:   %[[i36:.+]] = select i1 %[[i35]], i8 110, i8 %[[i34]]
; CHECK-NEXT:   %[[i37:.+]] = icmp eq i8 %ld.transb10, 84
; CHECK-NEXT:   %[[i38:.+]] = select i1 %[[i37]], i8 78, i8 %[[i36]]
; CHECK-NEXT:   store i8 %[[i38]], ptr %byref.transpose.transb11, align 1
; CHECK-NEXT:   %ld.row.trans12 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i39:.+]] = icmp eq i8 %ld.row.trans12, 110
; CHECK-NEXT:   %[[i40:.+]] = icmp eq i8 %ld.row.trans12, 78
; CHECK-NEXT:   %[[i41:.+]] = or i1 %[[i40]], %[[i39]]
; CHECK-NEXT:   %[[i42:.+]] = select i1 %[[i41]], ptr %byref.transpose.transb11, ptr %malloccall
; CHECK-NEXT:   %[[i43:.+]] = select i1 %[[i41]], ptr %"calloc'mi", ptr %rhs
; CHECK-NEXT:   %[[i44:.+]] = select i1 %[[i41]], ptr %rhs, ptr %"calloc'mi"
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.014, align 8
; CHECK-NEXT:   call void @dgemm_(ptr %malloccall, ptr %[[i42]], ptr %malloccall1, ptr %malloccall1, ptr %malloccall1, ptr %malloccall2, ptr %[[i43]], ptr %malloccall1, ptr %[[i44]], ptr %malloccall1, ptr %byref.constant.fp.1.014, ptr %"rhs'", ptr %malloccall1, i32 1, i32 1)
; CHECK-NEXT:   %ld.transa15 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i45:.+]] = icmp eq i8 %ld.transa15, 110
; CHECK-NEXT:   %[[i46:.+]] = select i1 %[[i45]], i8 116, i8 78
; CHECK-NEXT:   %[[i47:.+]] = icmp eq i8 %ld.transa15, 78
; CHECK-NEXT:   %[[i48:.+]] = select i1 %[[i47]], i8 84, i8 %[[i46]]
; CHECK-NEXT:   %[[i49:.+]] = icmp eq i8 %ld.transa15, 116
; CHECK-NEXT:   %[[i50:.+]] = select i1 %[[i49]], i8 110, i8 %[[i48]]
; CHECK-NEXT:   %[[i51:.+]] = icmp eq i8 %ld.transa15, 84
; CHECK-NEXT:   %[[i52:.+]] = select i1 %[[i51]], i8 78, i8 %[[i50]]
; CHECK-NEXT:   store i8 %[[i52]], ptr %byref.transpose.transa16, align 1
; CHECK-NEXT:   %ld.row.trans17 = load i8, ptr %malloccall, align 1
; CHECK-NEXT:   %[[i53:.+]] = icmp eq i8 %ld.row.trans17, 110
; CHECK-NEXT:   %[[i54:.+]] = icmp eq i8 %ld.row.trans17, 78
; CHECK-NEXT:   %[[i55:.+]] = or i1 %[[i54]], %[[i53]]
; CHECK-NEXT:   %[[i56:.+]] = select i1 %[[i55:.+]], ptr %byref.transpose.transa16, ptr %malloccall
; CHECK-NEXT:   %[[i57:.+]] = select i1 %[[i55:.+]], ptr %rhs, ptr %"calloc'mi"
; CHECK-NEXT:   %[[i58:.+]] = select i1 %[[i55:.+]], ptr %"calloc'mi", ptr %rhs
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.019, align 8
; CHECK-NEXT:   call void @dgemm_(ptr %[[i56]], ptr %malloccall, ptr %malloccall1, ptr %malloccall1, ptr %malloccall1, ptr %malloccall2, ptr %57, ptr %malloccall1, ptr %58, ptr %malloccall1, ptr %byref.constant.fp.1.019, ptr %"rhs'", ptr %malloccall1, i32 1, i32 1)
; CHECK-NEXT:   store i8 71, ptr %byref.constant.char.G20, align 1
; CHECK-NEXT:   store i32 0, ptr %byref.constant.int.021, align 4
; CHECK-NEXT:   store i32 0, ptr %byref.constant.int.022, align 4
; CHECK-NEXT:   store double 1.000000e+00, ptr %byref.constant.fp.1.023, align 8
; CHECK-NEXT:   call void @dlascl_(ptr %byref.constant.char.G20, ptr %byref.constant.int.021, ptr %byref.constant.int.022, ptr %byref.constant.fp.1.023, ptr %malloccall2, ptr %malloccall1, ptr %malloccall1, ptr %"calloc'mi", ptr %malloccall1, ptr %[[i1]], i32 1)
; CHECK-NEXT:   call void @free(ptr nonnull %"calloc'mi")
; CHECK-NEXT:   call void @free(ptr %calloc)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
