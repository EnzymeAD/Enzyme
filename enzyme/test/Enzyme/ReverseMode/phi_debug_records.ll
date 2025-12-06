; Test for fixing assertion failure when inserting PHI nodes after debug records
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-strict-aliasing=1 -S | FileCheck %s; fi

; ModuleID = 'safe.ll.txt'
source_filename = "std.c2be051742d3c8de-cgu.00"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@0 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@1 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@enzyme_const = external global ptr
@enzyme_dup = external global ptr
@2 = external hidden unnamed_addr constant [186 x i8], align 1
@3 = external hidden unnamed_addr constant <{ [8 x i8], [8 x i8] }>, align 8
@anon.d221b5a1ca7d011dae3ce51fb2611f53.15 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.d221b5a1ca7d011dae3ce51fb2611f53.17 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.677bc98fa4f847c2113641a77bae611c.3 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.e8ad111fcdba8f51c5e77cb80a764505.26 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.080693e073b7ed74ca255f40e933ae70.59 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.080693e073b7ed74ca255f40e933ae70.61 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.080693e073b7ed74ca255f40e933ae70.64 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.8f5ef5ec36c943a0d1df6b6363c9934d.6 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.8f5ef5ec36c943a0d1df6b6363c9934d.9 = external hidden unnamed_addr constant [36 x i8], align 1
@anon.8f5ef5ec36c943a0d1df6b6363c9934d.12 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.8f5ef5ec36c943a0d1df6b6363c9934d.23 = external hidden unnamed_addr constant [38 x i8], align 1
@anon.8f5ef5ec36c943a0d1df6b6363c9934d.24 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.8afe5fb67ae3f1afa5e6dc92f8599610.14 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.8afe5fb67ae3f1afa5e6dc92f8599610.15 = external hidden unnamed_addr constant [43 x i8], align 1
@4 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@_ZN12panic_unwind3imp6CANARY17hc59233b2ced56dcdE = external hidden constant [1 x i8], align 1
@anon.babb0264256341099826814eb4f3574c.2 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.babb0264256341099826814eb4f3574c.51 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.babb0264256341099826814eb4f3574c.53 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.babb0264256341099826814eb4f3574c.54 = external hidden unnamed_addr constant [36 x i8], align 1
@anon.babb0264256341099826814eb4f3574c.55 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.babb0264256341099826814eb4f3574c.60 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@_ZN3std3sys9backtrace4lock4LOCK17hf34eb78a76761272E = external hidden global <{ [5 x i8], [3 x i8] }>, align 4
@anon.babb0264256341099826814eb4f3574c.67 = external hidden unnamed_addr constant <{ ptr, [16 x i8], ptr, ptr, ptr, ptr, ptr, ptr, ptr }>, align 8
@anon.babb0264256341099826814eb4f3574c.68 = external hidden unnamed_addr constant <{ [24 x i8], ptr, ptr, ptr, ptr, ptr, ptr, ptr }>, align 8
@"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$11FIRST_PANIC17h1c9f85ba2097d41cE" = external hidden global [1 x i8], align 1
@anon.babb0264256341099826814eb4f3574c.70 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.babb0264256341099826814eb4f3574c.72 = external hidden unnamed_addr constant <{ [24 x i8], ptr, ptr, ptr, ptr }>, align 8
@anon.babb0264256341099826814eb4f3574c.73 = external hidden unnamed_addr constant <{ ptr, [16 x i8], ptr, ptr, ptr, ptr }>, align 8
@anon.babb0264256341099826814eb4f3574c.75 = external hidden unnamed_addr constant [12 x i8], align 1
@anon.babb0264256341099826814eb4f3574c.79 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.babb0264256341099826814eb4f3574c.82 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.babb0264256341099826814eb4f3574c.84 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@_ZN3std9panicking4HOOK17hf5ae89eefcc7069aE = external hidden global <{ [9 x i8], [7 x i8], [8 x i8], [8 x i8] }>, align 8
@anon.bc75b2dae71f64fa323d019d8562410b.3 = external hidden unnamed_addr constant <{ ptr, [9 x i8], [7 x i8] }>, align 8
@anon.bc75b2dae71f64fa323d019d8562410b.5 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@_ZN3std5panic14SHOULD_CAPTURE17h6ec7b3e7f01a7b14E = external hidden global [1 x i8], align 1
@anon.bc75b2dae71f64fa323d019d8562410b.79 = external hidden unnamed_addr constant [14 x i8], align 1
@anon.bc75b2dae71f64fa323d019d8562410b.80 = external hidden unnamed_addr constant [4 x i8], align 1
@anon.bc75b2dae71f64fa323d019d8562410b.83 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.bc75b2dae71f64fa323d019d8562410b.85 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@_ZN3std6thread8ThreadId3new7COUNTER17h101f9f62173dcc92E = external hidden global [8 x i8], align 8
@anon.bc75b2dae71f64fa323d019d8562410b.90 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.bc75b2dae71f64fa323d019d8562410b.91 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@_ZN3std5alloc4HOOK17h7130ea174ab09dc3E = external hidden local_unnamed_addr global [8 x i8], align 8
@anon.bb9016958efe0580f15baffae1ffbb38.6 = external hidden unnamed_addr constant ptr, align 8
@"_ZN3std2io5stdio14OUTPUT_CAPTURE29_$u7b$$u7b$constant$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$3VAL17h8917e6542b635b84E" = external hidden thread_local global <{ [9 x i8], [7 x i8] }>, align 8
@_ZN3std2io5stdio19OUTPUT_CAPTURE_USED17h71b5b109888cde8aE.0 = external hidden unnamed_addr global i8, align 1
@anon.7b33aeb34b5389b82a5a31bdbf8976e8.13 = external hidden unnamed_addr constant <{ [24 x i8], ptr, ptr, ptr }>, align 8
@_ZN3std6thread11main_thread4MAIN17he08dbfe964edeb58E.0 = external hidden unnamed_addr global i64, align 8
@anon.a3b51bfd7273ae0b2428f9302b4a6a52.3 = external hidden unnamed_addr constant <{ ptr, [9 x i8], [7 x i8] }>, align 8
@anon.a3b51bfd7273ae0b2428f9302b4a6a52.5 = external hidden unnamed_addr constant <{ ptr, [9 x i8], [7 x i8] }>, align 8
@_ZN3std3sys12thread_local5guard3key6enable5DTORS17h7430097ae0e77bf4E = external hidden global <{ [8 x i8], ptr }>, align 8
@anon.0bc126ce095015e3131941c1eb180bc1.18 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.0bc126ce095015e3131941c1eb180bc1.20 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@_ZN3std9panicking11panic_count18GLOBAL_PANIC_COUNT17hae1b51fffade13abE = external hidden global [8 x i8], align 8
@_rust_extern_with_linkage_c2be051742d3c8de___dso_handle = external hidden global ptr
@anon.20a33724a9b558c90ab53ce9f9d617b9.46 = external hidden unnamed_addr constant <{ ptr, [9 x i8], [7 x i8] }>, align 8
@anon.20a33724a9b558c90ab53ce9f9d617b9.48 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.20a33724a9b558c90ab53ce9f9d617b9.49 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.20a33724a9b558c90ab53ce9f9d617b9.72 = external hidden unnamed_addr constant <{ ptr, [16 x i8], ptr }>, align 8
@anon.20a33724a9b558c90ab53ce9f9d617b9.73 = external hidden unnamed_addr constant [43 x i8], align 1
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.13 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.14 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.17 = external hidden unnamed_addr constant <{ ptr, [16 x i8], ptr, ptr, ptr }>, align 8
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.18 = external hidden unnamed_addr constant <{ ptr, [16 x i8], ptr, ptr, ptr }>, align 8
@_ZN3std6thread7current2id2ID17hf976129d244513b0E = external hidden thread_local local_unnamed_addr global [8 x i8], align 8
@_ZN3std3sys12thread_local11destructors4list5DTORS17he82e25e5cbceeca8E = external hidden thread_local global [32 x i8], align 8
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.39 = external hidden unnamed_addr constant <{ ptr, [8 x i8] }>, align 8
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.40 = external hidden unnamed_addr constant [8 x i8], align 1
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.50 = external hidden unnamed_addr constant [4 x i8], align 1
@_ZN3std6thread7current7CURRENT17h856398c56bd7c7a7E = external hidden thread_local local_unnamed_addr global [8 x i8], align 8
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.61 = external hidden unnamed_addr constant [9 x i8], align 1
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.63 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.69 = external hidden unnamed_addr constant <{ ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8], ptr, [8 x i8] }>, align 8
@anon.fa0b702a440d3cf6c6d0f877f47e0aa3.93 = external hidden unnamed_addr constant <{ ptr, [16 x i8] }>, align 8

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #5

; Function Attrs: inlinehint nonlazybind uwtable
define hidden { i64, i64 } @"_ZN4core4iter5range101_$LT$impl$u20$core..iter..traits..iterator..Iterator$u20$for$u20$core..ops..range..Range$LT$A$GT$$GT$4next17h566d9e10f4c48efbE"(ptr align 8 %0) unnamed_addr #6 !dbg !13 {
  %2 = alloca [8 x i8], align 8
  store ptr %0, ptr %2, align 8
    #dbg_declare(ptr %2, !49, !DIExpression(), !52)
  %3 = call { i64, i64 } @"_ZN89_$LT$core..ops..range..Range$LT$T$GT$$u20$as$u20$core..iter..range..RangeIteratorImpl$GT$9spec_next17h4dfe42ce4e69327cE"(ptr align 8 %0), !dbg !53
  %4 = extractvalue { i64, i64 } %3, 0, !dbg !53
  %5 = extractvalue { i64, i64 } %3, 1, !dbg !53
  %6 = insertvalue { i64, i64 } poison, i64 %4, 0, !dbg !54
  %7 = insertvalue { i64, i64 } %6, i64 %5, 1, !dbg !54
  ret { i64, i64 } %7, !dbg !54
}

; Function Attrs: inlinehint nonlazybind uwtable
define hidden { i64, i64 } @"_ZN89_$LT$core..ops..range..Range$LT$T$GT$$u20$as$u20$core..iter..range..RangeIteratorImpl$GT$9spec_next17h4dfe42ce4e69327cE"(ptr align 8 %0) unnamed_addr #6 !dbg !55 {
  %2 = alloca [8 x i8], align 8
  %3 = alloca [8 x i8], align 8
  %4 = alloca [8 x i8], align 8
  %5 = alloca [8 x i8], align 8
  %6 = alloca [16 x i8], align 8
  store ptr %0, ptr %5, align 8
    #dbg_declare(ptr %5, !58, !DIExpression(), !61)
  %7 = getelementptr inbounds i8, ptr %0, i64 8, !dbg !62
  store ptr %0, ptr %3, align 8
    #dbg_declare(ptr %3, !63, !DIExpression(), !75)
  store ptr %7, ptr %2, align 8
    #dbg_declare(ptr %2, !74, !DIExpression(), !77)
  %8 = load i64, ptr %0, align 8, !dbg !78
  %9 = load i64, ptr %7, align 8, !dbg !79
  %10 = icmp ult i64 %8, %9, !dbg !78
  br i1 %10, label %12, label %11, !dbg !80

11:                                               ; preds = %1
  store i64 0, ptr %6, align 8, !dbg !81
  br label %16, !dbg !82

12:                                               ; preds = %1
  %13 = load i64, ptr %0, align 8, !dbg !83
  store i64 %13, ptr %4, align 8, !dbg !83
    #dbg_declare(ptr %4, !59, !DIExpression(), !84)
  %14 = call i64 @"_ZN49_$LT$usize$u20$as$u20$core..iter..range..Step$GT$17forward_unchecked17h4a46ddd3bf136cddE"(i64 %13, i64 1), !dbg !85
  store i64 %14, ptr %0, align 8, !dbg !86
  %15 = getelementptr inbounds i8, ptr %6, i64 8, !dbg !87
  store i64 %13, ptr %15, align 8, !dbg !87
  store i64 1, ptr %6, align 8, !dbg !87
  br label %16, !dbg !82

16:                                               ; preds = %12, %11
  %17 = load i64, ptr %6, align 8, !dbg !88
  %18 = getelementptr inbounds i8, ptr %6, i64 8, !dbg !88
  %19 = load i64, ptr %18, align 8, !dbg !88
  %20 = insertvalue { i64, i64 } poison, i64 %17, 0, !dbg !88
  %21 = insertvalue { i64, i64 } %20, i64 %19, 1, !dbg !88
  ret { i64, i64 } %21, !dbg !88
}

; Function Attrs: inlinehint nonlazybind uwtable
define hidden { i64, i64 } @"_ZN63_$LT$I$u20$as$u20$core..iter..traits..collect..IntoIterator$GT$9into_iter17h35b862536a052749E"(i64 %0, i64 %1) unnamed_addr #6 !dbg !89 {
  %3 = alloca [16 x i8], align 8
  store i64 %0, ptr %3, align 8
  %4 = getelementptr inbounds i8, ptr %3, i64 8
  store i64 %1, ptr %4, align 8
    #dbg_declare(ptr %3, !97, !DIExpression(), !100)
  %5 = insertvalue { i64, i64 } poison, i64 %0, 0, !dbg !101
  %6 = insertvalue { i64, i64 } %5, i64 %1, 1, !dbg !101
  ret { i64, i64 } %6, !dbg !101
}

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN5gmmrs4safe14dgmm_objective17hd5def2d68203fb76E(ptr align 8 %0, i64 %1, ptr align 8 %2, i64 %3) unnamed_addr #7 !dbg !102 {
  %5 = alloca [16 x i8], align 8
  %6 = alloca [16 x i8], align 8
  %7 = alloca [32 x i8], align 8
  store ptr %0, ptr %6, align 8
  %8 = getelementptr inbounds i8, ptr %6, i64 8
  store i64 %1, ptr %8, align 8
    #dbg_declare(ptr %6, !115, !DIExpression(), !117)
  store ptr %2, ptr %5, align 8
  %9 = getelementptr inbounds i8, ptr %5, i64 8
  store i64 %3, ptr %9, align 8
    #dbg_declare(ptr %5, !116, !DIExpression(), !117)
  store ptr %0, ptr %7, align 8, !dbg !118
  %10 = getelementptr inbounds i8, ptr %7, i64 8, !dbg !118
  store i64 %1, ptr %10, align 8, !dbg !118
  %11 = getelementptr inbounds i8, ptr %7, i64 16, !dbg !118
  store ptr %2, ptr %11, align 8, !dbg !118
  %12 = getelementptr inbounds i8, ptr %11, i64 8, !dbg !118
  store i64 %3, ptr %12, align 8, !dbg !118
  %13 = load { ptr, i64 }, ptr %7, align 8, !dbg !118
  %14 = extractvalue { ptr, i64 } %13, 0, !dbg !118
  %15 = extractvalue { ptr, i64 } %13, 1, !dbg !118
  %16 = getelementptr inbounds i8, ptr %7, i64 16, !dbg !118
  %17 = load { ptr, i64 }, ptr %16, align 8, !dbg !118
  %18 = extractvalue { ptr, i64 } %17, 0, !dbg !118
  %19 = extractvalue { ptr, i64 } %17, 1, !dbg !118
  %20 = call {} (...) @__enzyme_autodiff_ZN5gmmrs4safe14dgmm_objective17hd5def2d68203fb76E(ptr @_ZN5gmmrs4safe13gmm_objective17h849a4e1a48dbbe36E, ptr @enzyme_dup, ptr %14, ptr %18, ptr @enzyme_const, i64 %15), !dbg !118
  ret void, !dbg !119
}

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN5gmmrs4safe13gmm_objective17h849a4e1a48dbbe36E(ptr align 8 %0, i64 %1) unnamed_addr #7 !dbg !120 {
  %3 = alloca [8 x i8], align 8
  %4 = alloca [16 x i8], align 8
  %5 = alloca [16 x i8], align 8
  %6 = alloca [16 x i8], align 8
  %7 = alloca [8 x i8], align 8
  store ptr %0, ptr %4, align 8
  %8 = getelementptr inbounds i8, ptr %4, i64 8
  store i64 %1, ptr %8, align 8
    #dbg_declare(ptr %4, !124, !DIExpression(), !131)
    #dbg_declare(ptr %7, !125, !DIExpression(), !132)
    #dbg_declare(ptr %6, !127, !DIExpression(), !133)
  store double 0.000000e+00, ptr %7, align 8, !dbg !134
  %9 = call { i64, i64 } @"_ZN63_$LT$I$u20$as$u20$core..iter..traits..collect..IntoIterator$GT$9into_iter17h35b862536a052749E"(i64 0, i64 10), !dbg !135
  %10 = extractvalue { i64, i64 } %9, 0, !dbg !135
  %11 = extractvalue { i64, i64 } %9, 1, !dbg !135
  store i64 %10, ptr %6, align 8, !dbg !135
  %12 = getelementptr inbounds i8, ptr %6, i64 8, !dbg !135
  store i64 %11, ptr %12, align 8, !dbg !135
  br label %13, !dbg !136

13:                                               ; preds = %33, %2
  %14 = call { i64, i64 } @"_ZN4core4iter5range101_$LT$impl$u20$core..iter..traits..iterator..Iterator$u20$for$u20$core..ops..range..Range$LT$A$GT$$GT$4next17h566d9e10f4c48efbE"(ptr align 8 %6), !dbg !133
  %15 = extractvalue { i64, i64 } %14, 0, !dbg !133
  %16 = extractvalue { i64, i64 } %14, 1, !dbg !133
  store i64 %15, ptr %5, align 8, !dbg !133
  %17 = getelementptr inbounds i8, ptr %5, i64 8, !dbg !133
  store i64 %16, ptr %17, align 8, !dbg !133
  %18 = load i64, ptr %5, align 8, !dbg !133
  %19 = getelementptr inbounds i8, ptr %5, i64 8, !dbg !133
  %20 = load i64, ptr %19, align 8, !dbg !133
  %21 = trunc nuw i64 %18 to i1, !dbg !133
  br i1 %21, label %22, label %26, !dbg !133

22:                                               ; preds = %13
  %23 = getelementptr inbounds i8, ptr %5, i64 8, !dbg !137
  %24 = load i64, ptr %23, align 8, !dbg !137
  store i64 %24, ptr %3, align 8, !dbg !137
    #dbg_declare(ptr %3, !129, !DIExpression(), !138)
  %25 = icmp ult i64 %24, %1, !dbg !139
  br i1 %25, label %28, label %32, !dbg !139

26:                                               ; preds = %13
  %27 = load double, ptr %7, align 8, !dbg !140
  store double %27, ptr %7, align 8, !dbg !141
  ret void, !dbg !142

28:                                               ; preds = %22
  %29 = getelementptr inbounds nuw double, ptr %0, i64 %24, !dbg !139
  %30 = load double, ptr %29, align 8, !dbg !139
  %31 = icmp ult i64 %24, %1, !dbg !143
  br i1 %31, label %33, label %35, !dbg !143

32:                                               ; preds = %22
  call void @_ZN4core9panicking18panic_bounds_check17h74d7c20cf9e1a9efE(i64 %24, i64 %1, ptr align 8 @0) #53, !dbg !139
  unreachable, !dbg !139

33:                                               ; preds = %28
  %34 = getelementptr inbounds nuw double, ptr %0, i64 %24, !dbg !143
  store double %30, ptr %34, align 8, !dbg !143
  br label %13, !dbg !136

35:                                               ; preds = %28
  call void @_ZN4core9panicking18panic_bounds_check17h74d7c20cf9e1a9efE(i64 %24, i64 %1, ptr align 8 @1) #53, !dbg !143
  unreachable, !dbg !143

36:                                               ; No predecessors!
  unreachable, !dbg !133
}

declare {} @__enzyme_autodiff_ZN5gmmrs4safe14dgmm_objective17hd5def2d68203fb76E(...)

; Function Attrs: allockind("alloc,uninitialized,aligned") allocsize(0) uwtable
define hidden noalias ptr @_RNvCs1QLEhZ2QfLZ_7___rustc12___rust_alloc(i64 %0, i64 allocalign %1) unnamed_addr #8 {
  %3 = tail call ptr @_RNvCs1QLEhZ2QfLZ_7___rustc11___rdl_alloc(i64 %0, i64 %1)
  ret ptr %3
}

; Function Attrs: allockind("free") uwtable
define hidden void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr allocptr %0, i64 %1, i64 %2) unnamed_addr #9 {
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc13___rdl_dealloc(ptr %0, i64 %1, i64 %2)
  ret void
}

; Function Attrs: allockind("realloc,aligned") allocsize(3) uwtable
define hidden noalias ptr @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_realloc(ptr allocptr %0, i64 %1, i64 allocalign %2, i64 %3) unnamed_addr #10 {
  %5 = tail call ptr @_RNvCs1QLEhZ2QfLZ_7___rustc13___rdl_realloc(ptr %0, i64 %1, i64 %2, i64 %3)
  ret ptr %5
}

; Function Attrs: allockind("alloc,zeroed,aligned") allocsize(0) uwtable
define hidden noalias ptr @_RNvCs1QLEhZ2QfLZ_7___rustc19___rust_alloc_zeroed(i64 %0, i64 allocalign %1) unnamed_addr #11 {
  %3 = tail call ptr @_RNvCs1QLEhZ2QfLZ_7___rustc18___rdl_alloc_zeroed(i64 %0, i64 %1)
  ret ptr %3
}

; Function Attrs: noreturn uwtable
define hidden void @_RNvCs1QLEhZ2QfLZ_7___rustc26___rust_alloc_error_handler(i64 %0, i64 %1) unnamed_addr #12 {
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc8___rg_oom(i64 %0, i64 %1)
  ret void
}

; Function Attrs: uwtable
define hidden void @_RNvCs1QLEhZ2QfLZ_7___rustc35___rust_no_alloc_shim_is_unstable_v2() unnamed_addr #13 {
  ret void
}

; Function Attrs: nofree nounwind nonlazybind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #14

; Function Attrs: inlinehint nounwind nonlazybind uwtable
define hidden void @"_ZN4core3num23_$LT$impl$u20$usize$GT$13unchecked_add18precondition_check17h5021e0cd12831d11E"(i64 %0, i64 %1, ptr align 8 %2) unnamed_addr #15 !dbg !144 {
  %4 = alloca [8 x i8], align 8
  %5 = alloca [16 x i8], align 8
  %6 = alloca [8 x i8], align 8
  %7 = alloca [8 x i8], align 8
  %8 = alloca [16 x i8], align 8
  %9 = alloca [48 x i8], align 8
  store i64 %0, ptr %7, align 8
    #dbg_declare(ptr %7, !183, !DIExpression(), !187)
    #dbg_declare(ptr %7, !188, !DIExpression(), !199)
  store i64 %1, ptr %6, align 8
    #dbg_declare(ptr %6, !184, !DIExpression(), !187)
    #dbg_declare(ptr %6, !198, !DIExpression(), !202)
  store ptr @2, ptr %5, align 8, !dbg !203
  %10 = getelementptr inbounds i8, ptr %5, i64 8, !dbg !203
  store i64 186, ptr %10, align 8, !dbg !203
    #dbg_declare(ptr %5, !185, !DIExpression(), !203)
  %11 = add i64 %0, %1, !dbg !204
  %12 = icmp ult i64 %11, %0, !dbg !204
  br i1 %12, label %14, label %13, !dbg !205

13:                                               ; preds = %3
  ret void, !dbg !206

14:                                               ; preds = %3
  %15 = getelementptr inbounds nuw { ptr, i64 }, ptr %8, i64 0, !dbg !207
  store ptr @2, ptr %15, align 8, !dbg !207
  %16 = getelementptr inbounds i8, ptr %15, i64 8, !dbg !207
  store i64 186, ptr %16, align 8, !dbg !207
  store ptr %8, ptr %4, align 8, !dbg !208
    #dbg_declare(ptr %4, !209, !DIExpression(), !346)
  store ptr %8, ptr %9, align 8, !dbg !348
  %17 = getelementptr inbounds i8, ptr %9, i64 8, !dbg !348
  store i64 1, ptr %17, align 8, !dbg !348
  %18 = load ptr, ptr @3, align 8, !dbg !348
  %19 = load i64, ptr getelementptr inbounds (i8, ptr @3, i64 8), align 8, !dbg !348
  %20 = getelementptr inbounds i8, ptr %9, i64 32, !dbg !348
  store ptr %18, ptr %20, align 8, !dbg !348
  %21 = getelementptr inbounds i8, ptr %20, i64 8, !dbg !348
  store i64 %19, ptr %21, align 8, !dbg !348
  %22 = getelementptr inbounds i8, ptr %9, i64 16, !dbg !348
  store ptr inttoptr (i64 8 to ptr), ptr %22, align 8, !dbg !348
  %23 = getelementptr inbounds i8, ptr %22, i64 8, !dbg !348
  store i64 0, ptr %23, align 8, !dbg !348
  call void @_ZN4core9panicking18panic_nounwind_fmt17hc0b47d79028b5977E(ptr align 8 %9, i1 zeroext false, ptr align 8 %2) #54, !dbg !349
  unreachable, !dbg !349
}

; Function Attrs: cold nonlazybind uwtable
define hidden fastcc void @"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h742096eab92de7ccE"(ptr noalias noundef nonnull align 8 captures(none) dereferenceable(16) %0, i64 noundef %1, i64 noundef %2) unnamed_addr #16 personality ptr @rust_eh_personality {
  %4 = alloca [24 x i8], align 8
  %5 = alloca [24 x i8], align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !350)
  %6 = add i64 %2, %1
  %7 = icmp ult i64 %6, %1
  br i1 %7, label %30, label %8, !prof !353

8:                                                ; preds = %3
  %9 = load i64, ptr %0, align 8, !range !354, !alias.scope !350, !noundef !29
  %10 = shl nuw i64 %9, 1
  %11 = tail call noundef i64 @llvm.umax.i64(i64 %6, i64 range(i64 0, -1) %10)
  %12 = tail call noundef i64 @llvm.umax.i64(i64 %11, i64 range(i64 0, -1) 8)
  %13 = icmp slt i64 %12, 0
  br i1 %13, label %30, label %14, !prof !355

14:                                               ; preds = %8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %5), !noalias !350
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4), !noalias !350
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %16 = icmp eq i64 %9, 0
  br i1 %16, label %20, label %17

17:                                               ; preds = %14
  %18 = load ptr, ptr %15, align 8, !alias.scope !350, !nonnull !29, !noundef !29
  store ptr %18, ptr %4, align 8, !alias.scope !356, !noalias !350
  %19 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i64 %9, ptr %19, align 8, !alias.scope !356, !noalias !350
  br label %20

20:                                               ; preds = %17, %14
  %21 = phi i64 [ 1, %17 ], [ 0, %14 ]
  %22 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %21, ptr %22, align 8, !alias.scope !356, !noalias !350
  call fastcc void @_ZN5alloc7raw_vec11finish_grow17hd50c6a4d78cdbf94E(ptr noalias noundef align 8 captures(address) dereferenceable(24) %5, i64 noundef 1, i64 noundef %12, ptr noalias noundef readonly align 8 captures(address) dereferenceable(24) %4), !noalias !350
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4), !noalias !350
  %23 = load i64, ptr %5, align 8, !range !359, !noalias !350, !noundef !29
  %24 = trunc nuw i64 %23 to i1
  %25 = getelementptr inbounds nuw i8, ptr %5, i64 8
  br i1 %24, label %26, label %33

26:                                               ; preds = %20
  %27 = load i64, ptr %25, align 8, !range !360, !noalias !350, !noundef !29
  %28 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %29 = load i64, ptr %28, align 8, !noalias !350
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %5), !noalias !350
  br label %30

30:                                               ; preds = %26, %8, %3
  %31 = phi i64 [ undef, %8 ], [ undef, %3 ], [ %29, %26 ]
  %32 = phi i64 [ 0, %8 ], [ 0, %3 ], [ %27, %26 ]
  tail call void @_ZN5alloc7raw_vec12handle_error17h2546acf93648cb86E(i64 noundef %32, i64 %31) #53
  unreachable

33:                                               ; preds = %20
  %34 = load ptr, ptr %25, align 8, !noalias !350, !nonnull !29, !noundef !29
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %5), !noalias !350
  store ptr %34, ptr %15, align 8, !alias.scope !350
  store i64 %12, ptr %0, align 8, !alias.scope !350
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #3

; Function Attrs: cold nounwind nonlazybind uwtable
define hidden fastcc void @_ZN5alloc7raw_vec11finish_grow17hd50c6a4d78cdbf94E(ptr dead_on_unwind noalias noundef nonnull writable writeonly align 8 captures(none) dereferenceable(24) initializes((0, 24)) %0, i64 noundef range(i64 1, -9223372036854775807) %1, i64 noundef %2, ptr dead_on_return noalias noundef nonnull readonly align 8 captures(none) dereferenceable(24) %3) unnamed_addr #17 {
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %6 = load i64, ptr %5, align 8, !range !360, !noundef !29
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %23, label %8

8:                                                ; preds = %4
  %9 = load ptr, ptr %3, align 8, !nonnull !29, !noundef !29
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %11 = load i64, ptr %10, align 8, !noundef !29
  %12 = icmp eq i64 %6, %1
  tail call void @llvm.assume(i1 %12)
  %13 = icmp eq i64 %11, 0
  br i1 %13, label %14, label %20

14:                                               ; preds = %8
  %15 = icmp eq i64 %2, 0
  br i1 %15, label %16, label %18

16:                                               ; preds = %14
  %17 = getelementptr i8, ptr null, i64 %1
  br label %29

18:                                               ; preds = %14
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #37
  %19 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc12___rust_alloc(i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) %1) #37
  br label %29

20:                                               ; preds = %8
  %21 = icmp uge i64 %2, %11
  tail call void @llvm.assume(i1 %21)
  %22 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_realloc(ptr noundef nonnull %9, i64 noundef %11, i64 noundef range(i64 1, -9223372036854775807) %1, i64 noundef %2) #37
  br label %29

23:                                               ; preds = %4
  %24 = icmp eq i64 %2, 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %23
  %26 = getelementptr i8, ptr null, i64 %1
  br label %29

27:                                               ; preds = %23
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #37
  %28 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc12___rust_alloc(i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) %1) #37
  br label %29

29:                                               ; preds = %27, %25, %20, %18, %16
  %30 = phi ptr [ %22, %20 ], [ %17, %16 ], [ %19, %18 ], [ %26, %25 ], [ %28, %27 ]
  %31 = icmp eq ptr %30, null
  %32 = inttoptr i64 %1 to ptr
  %33 = select i1 %31, ptr %32, ptr %30
  %34 = zext i1 %31 to i64
  %35 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %33, ptr %35, align 8
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %2, ptr %36, align 8
  store i64 %34, ptr %0, align 8
  ret void
}

; Function Attrs: cold minsize noreturn nonlazybind optsize uwtable
define hidden void @_ZN5alloc7raw_vec12handle_error17h2546acf93648cb86E(i64 noundef range(i64 0, -9223372036854775807) %0, i64 %1) unnamed_addr #18 {
  %3 = icmp eq i64 %0, 0
  br i1 %3, label %5, label %4, !prof !361

4:                                                ; preds = %2
  tail call void @_ZN5alloc5alloc18handle_alloc_error17h1bbcba5314f57599E(i64 noundef %0, i64 noundef %1) #53
  unreachable

5:                                                ; preds = %2
  tail call fastcc void @_ZN5alloc7raw_vec17capacity_overflow17he436d9f6450bc036E() #53
  unreachable
}

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden fastcc void @_ZN5alloc7raw_vec17capacity_overflow17he436d9f6450bc036E() unnamed_addr #19 {
  %1 = alloca [48 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %1)
  store ptr @anon.d221b5a1ca7d011dae3ce51fb2611f53.15, ptr %1, align 8
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i64 1, ptr %2, align 8
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store ptr null, ptr %3, align 8
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %4, align 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 0, ptr %5, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.d221b5a1ca7d011dae3ce51fb2611f53.17) #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
define hidden void @"_ZN132_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$alloc..vec..spec_extend..SpecExtend$LT$$RF$T$C$core..slice..iter..Iter$LT$T$GT$$GT$$GT$11spec_extend17h12d2578eb1732a16E"(ptr noalias noundef align 8 captures(none) dereferenceable(24) %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #20 {
  %4 = icmp ne ptr %2, null
  tail call void @llvm.assume(i1 %4)
  %5 = ptrtoint ptr %2 to i64
  %6 = ptrtoint ptr %1 to i64
  %7 = sub nuw i64 %5, %6
  tail call void @llvm.experimental.noalias.scope.decl(metadata !362)
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = load i64, ptr %8, align 8, !alias.scope !365, !noundef !29
  %10 = load i64, ptr %0, align 8, !range !354, !alias.scope !365, !noundef !29
  %11 = sub i64 %10, %9
  %12 = icmp ugt i64 %7, %11
  br i1 %12, label %13, label %15, !prof !353

13:                                               ; preds = %3
  tail call fastcc void @"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h742096eab92de7ccE"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0, i64 noundef %9, i64 noundef %7)
  %14 = load i64, ptr %8, align 8, !alias.scope !362
  br label %15

15:                                               ; preds = %13, %3
  %16 = phi i64 [ %9, %3 ], [ %14, %13 ]
  %17 = icmp sgt i64 %16, -1
  tail call void @llvm.assume(i1 %17)
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %19 = load ptr, ptr %18, align 8, !alias.scope !362, !nonnull !29, !noundef !29
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 %16
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %20, ptr nonnull readonly align 1 %1, i64 %7, i1 false), !noalias !362
  %21 = add i64 %16, %7
  store i64 %21, ptr %8, align 8, !alias.scope !362
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden { ptr, i64 } @"_ZN5alloc3vec16Vec$LT$T$C$A$GT$16into_boxed_slice17hd30854b51e9f897aE"(ptr dead_on_return noalias noundef readonly align 8 captures(none) dereferenceable(24) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = load i64, ptr %0, align 8, !range !354, !noundef !29
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %4 = load i64, ptr %3, align 8, !noundef !29
  %5 = icmp ugt i64 %2, %4
  br i1 %5, label %9, label %6

6:                                                ; preds = %1
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %8 = load ptr, ptr %7, align 8
  br label %19

9:                                                ; preds = %1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !368)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !371)
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %11 = load ptr, ptr %10, align 8, !alias.scope !374, !nonnull !29, !noundef !29
  %12 = icmp eq i64 %4, 0
  br i1 %12, label %13, label %14

13:                                               ; preds = %9
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %11, i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) 1) #37, !noalias !374
  br label %19

14:                                               ; preds = %9
  %15 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_realloc(ptr noundef nonnull %11, i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) 1, i64 noundef range(i64 1, 9223372036854775807) %4) #37, !noalias !374
  %16 = icmp eq ptr %15, null
  br i1 %16, label %17, label %19

17:                                               ; preds = %14
  invoke void @_ZN5alloc7raw_vec12handle_error17h2546acf93648cb86E(i64 noundef 1, i64 range(i64 0, 9223372036854775807) %4) #53
          to label %18 unwind label %24

18:                                               ; preds = %17
  unreachable

19:                                               ; preds = %14, %13, %6
  %20 = phi ptr [ %8, %6 ], [ inttoptr (i64 1 to ptr), %13 ], [ %15, %14 ]
  %21 = icmp sgt i64 %4, -1
  tail call void @llvm.assume(i1 %21)
  %22 = insertvalue { ptr, i64 } poison, ptr %20, 0
  %23 = insertvalue { ptr, i64 } %22, i64 %4, 1
  ret { ptr, i64 } %23

24:                                               ; preds = %17
  %25 = landingpad { ptr, i32 }
          cleanup
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %11, i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) 1) #37, !noalias !375
  resume { ptr, i32 } %25
}

; Function Attrs: noinline nonlazybind uwtable
define hidden void @"_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h34bd7e99b107a31bE"(ptr noalias noundef align 8 captures(none) dereferenceable(16) %0) unnamed_addr #21 personality ptr @rust_eh_personality {
  %2 = alloca [24 x i8], align 8
  %3 = alloca [24 x i8], align 8
  %4 = load i64, ptr %0, align 8, !range !354, !noundef !29
  tail call void @llvm.experimental.noalias.scope.decl(metadata !378)
  %5 = shl nuw i64 %4, 1
  %6 = tail call i64 @llvm.umax.i64(i64 %5, i64 range(i64 0, -1) 8)
  %7 = icmp slt i64 %6, 0
  br i1 %7, label %24, label %8, !prof !355

8:                                                ; preds = %1
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %3), !noalias !378
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %2), !noalias !378
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = icmp eq i64 %4, 0
  br i1 %10, label %14, label %11

11:                                               ; preds = %8
  %12 = load ptr, ptr %9, align 8, !alias.scope !378, !nonnull !29, !noundef !29
  store ptr %12, ptr %2, align 8, !alias.scope !381, !noalias !378
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store i64 %4, ptr %13, align 8, !alias.scope !381, !noalias !378
  br label %14

14:                                               ; preds = %11, %8
  %15 = phi i64 [ 1, %11 ], [ 0, %8 ]
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i64 %15, ptr %16, align 8, !alias.scope !381, !noalias !378
  call fastcc void @_ZN5alloc7raw_vec11finish_grow17hd50c6a4d78cdbf94E(ptr noalias noundef align 8 captures(address) dereferenceable(24) %3, i64 noundef 1, i64 noundef %6, ptr noalias noundef readonly align 8 captures(address) dereferenceable(24) %2), !noalias !378
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %2), !noalias !378
  %17 = load i64, ptr %3, align 8, !range !359, !noalias !378, !noundef !29
  %18 = trunc nuw i64 %17 to i1
  %19 = getelementptr inbounds nuw i8, ptr %3, i64 8
  br i1 %18, label %20, label %27

20:                                               ; preds = %14
  %21 = load i64, ptr %19, align 8, !range !360, !noalias !378, !noundef !29
  %22 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %23 = load i64, ptr %22, align 8, !noalias !378
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %3), !noalias !378
  br label %24

24:                                               ; preds = %20, %1
  %25 = phi i64 [ undef, %1 ], [ %23, %20 ]
  %26 = phi i64 [ 0, %1 ], [ %21, %20 ]
  tail call void @_ZN5alloc7raw_vec12handle_error17h2546acf93648cb86E(i64 noundef %26, i64 %25) #53
  unreachable

27:                                               ; preds = %14
  %28 = load ptr, ptr %19, align 8, !noalias !378, !nonnull !29, !noundef !29
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %3), !noalias !378
  store ptr %28, ptr %9, align 8, !alias.scope !378
  store i64 %6, ptr %0, align 8, !alias.scope !378
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #3

; Function Attrs: nonlazybind uwtable
define hidden void @"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$13reserve_exact17h7f250ee6644f84e7E"(ptr noalias noundef align 8 captures(none) dereferenceable(16) %0, i64 noundef %1, i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) %3, i64 noundef %4) unnamed_addr #20 {
  %6 = alloca [24 x i8], align 8
  %7 = alloca [24 x i8], align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !384)
  %8 = icmp eq i64 %4, 0
  %9 = load i64, ptr %0, align 8, !range !354, !alias.scope !384
  %10 = select i1 %8, i64 -1, i64 %9
  %11 = sub i64 %10, %1
  %12 = icmp ugt i64 %2, %11
  br i1 %12, label %13, label %51

13:                                               ; preds = %5
  tail call void @llvm.experimental.noalias.scope.decl(metadata !387)
  br i1 %8, label %48, label %14

14:                                               ; preds = %13
  %15 = add i64 %2, %1
  %16 = icmp ult i64 %15, %1
  br i1 %16, label %48, label %17, !prof !353

17:                                               ; preds = %14
  %18 = add i64 %3, -1
  %19 = add nuw i64 %18, %4
  %20 = sub i64 0, %3
  %21 = and i64 %19, %20
  %22 = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %21, i64 %15)
  %23 = extractvalue { i64, i1 } %22, 0
  %24 = extractvalue { i64, i1 } %22, 1
  %25 = sub nuw i64 -9223372036854775808, %3
  %26 = icmp ugt i64 %23, %25
  %27 = select i1 %24, i1 true, i1 %26
  br i1 %27, label %48, label %28, !prof !355

28:                                               ; preds = %17
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %7), !noalias !390
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %6), !noalias !390
  %29 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %30 = icmp eq i64 %9, 0
  br i1 %30, label %35, label %31

31:                                               ; preds = %28
  %32 = load ptr, ptr %29, align 8, !alias.scope !390, !nonnull !29, !noundef !29
  %33 = mul nuw i64 %9, %4
  store ptr %32, ptr %6, align 8, !alias.scope !391, !noalias !390
  %34 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store i64 %33, ptr %34, align 8, !alias.scope !391, !noalias !390
  br label %35

35:                                               ; preds = %31, %28
  %36 = phi i64 [ %3, %31 ], [ 0, %28 ]
  %37 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i64 %36, ptr %37, align 8, !alias.scope !391, !noalias !390
  call fastcc void @_ZN5alloc7raw_vec11finish_grow17hd50c6a4d78cdbf94E(ptr noalias noundef align 8 captures(address) dereferenceable(24) %7, i64 noundef range(i64 1, -9223372036854775807) %3, i64 noundef %23, ptr noalias noundef readonly align 8 captures(address) dereferenceable(24) %6), !noalias !390
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %6), !noalias !390
  %38 = load i64, ptr %7, align 8, !range !359, !noalias !390, !noundef !29
  %39 = trunc nuw i64 %38 to i1
  %40 = getelementptr inbounds nuw i8, ptr %7, i64 8
  br i1 %39, label %41, label %45

41:                                               ; preds = %35
  %42 = load i64, ptr %40, align 8, !range !360, !noalias !390, !noundef !29
  %43 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %44 = load i64, ptr %43, align 8, !noalias !390
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %7), !noalias !390
  br label %48

45:                                               ; preds = %35
  %46 = load ptr, ptr %40, align 8, !noalias !390, !nonnull !29, !noundef !29
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %7), !noalias !390
  store ptr %46, ptr %29, align 8, !alias.scope !390
  %47 = icmp sgt i64 %15, -1
  tail call void @llvm.assume(i1 %47)
  store i64 %15, ptr %0, align 8, !alias.scope !390
  br label %51

48:                                               ; preds = %41, %17, %14, %13
  %49 = phi i64 [ %44, %41 ], [ undef, %13 ], [ undef, %14 ], [ undef, %17 ]
  %50 = phi i64 [ %42, %41 ], [ 0, %13 ], [ 0, %14 ], [ 0, %17 ]
  tail call void @_ZN5alloc7raw_vec12handle_error17h2546acf93648cb86E(i64 noundef %50, i64 %49) #53
  unreachable

51:                                               ; preds = %45, %5
  %52 = phi i64 [ %2, %45 ], [ %11, %5 ]
  %53 = icmp ule i64 %2, %52
  tail call void @llvm.assume(i1 %53)
  ret void
}

; Function Attrs: nounwind nonlazybind uwtable
define hidden void @"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$15try_allocate_in17h47e3396744964f2cE"(ptr dead_on_unwind noalias noundef writable writeonly sret([24 x i8]) align 8 captures(none) dereferenceable(24) initializes((0, 16)) %0, i64 noundef %1, i1 noundef zeroext %2, i64 noundef range(i64 1, -9223372036854775807) %3, i64 noundef %4) unnamed_addr #22 personality ptr @rust_eh_personality {
  %6 = add i64 %3, -1
  %7 = add nuw i64 %6, %4
  %8 = sub i64 0, %3
  %9 = and i64 %7, %8
  %10 = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %9, i64 %1)
  %11 = extractvalue { i64, i1 } %10, 0
  %12 = extractvalue { i64, i1 } %10, 1
  %13 = sub nuw i64 -9223372036854775808, %3
  %14 = icmp ugt i64 %11, %13
  %15 = select i1 %12, i1 true, i1 %14
  br i1 %15, label %16, label %18, !prof !355

16:                                               ; preds = %5
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 0, ptr %17, align 8
  br label %25

18:                                               ; preds = %5
  %19 = icmp eq i64 %11, 0
  br i1 %19, label %20, label %24

20:                                               ; preds = %18
  %21 = getelementptr i8, ptr null, i64 %3
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 0, ptr %22, align 8
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %21, ptr %23, align 8
  br label %25

24:                                               ; preds = %18
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #37
  br i1 %2, label %27, label %29

25:                                               ; preds = %37, %34, %20, %16
  %26 = phi i64 [ 1, %34 ], [ 1, %16 ], [ 0, %37 ], [ 0, %20 ]
  store i64 %26, ptr %0, align 8
  ret void

27:                                               ; preds = %24
  %28 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc19___rust_alloc_zeroed(i64 noundef range(i64 1, 0) %11, i64 noundef range(i64 1, -9223372036854775807) %3) #37
  br label %31

29:                                               ; preds = %24
  %30 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc12___rust_alloc(i64 noundef %11, i64 noundef range(i64 1, -9223372036854775807) %3) #37
  br label %31

31:                                               ; preds = %29, %27
  %32 = phi ptr [ %28, %27 ], [ %30, %29 ]
  %33 = icmp eq ptr %32, null
  br i1 %33, label %34, label %37

34:                                               ; preds = %31
  %35 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %3, ptr %35, align 8
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %11, ptr %36, align 8
  br label %25

37:                                               ; preds = %31
  %38 = icmp sgt i64 %1, -1
  tail call void @llvm.assume(i1 %38)
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %1, ptr %39, align 8
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %32, ptr %40, align 8
  br label %25
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(none) uwtable
define hidden void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h7e558c65bd3bbcf0E"(ptr noalias noundef readnone align 8 captures(none) dereferenceable(24) %0) unnamed_addr #23 {
  ret void
}

; Function Attrs: nounwind nonlazybind uwtable
define hidden void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17hdf7f45732cc816a4E"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(16) %0) unnamed_addr #22 {
  %2 = load i64, ptr %0, align 8, !range !354, !noundef !29
  %3 = icmp eq i64 %2, 0
  br i1 %3, label %7, label %4

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load ptr, ptr %5, align 8, !nonnull !29, !noundef !29
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %6, i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) 1) #37
  br label %7

7:                                                ; preds = %4, %1
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17h71ba09309df63a21E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  invoke void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h7e558c65bd3bbcf0E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0)
          to label %4 unwind label %2

2:                                                ; preds = %1
  %3 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17hdf7f45732cc816a4E"(ptr noalias noundef nonnull align 8 dereferenceable(16) %0)
          to label %7 unwind label %5

4:                                                ; preds = %1
  tail call void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17hdf7f45732cc816a4E"(ptr noalias noundef nonnull align 8 dereferenceable(16) %0)
  ret void

5:                                                ; preds = %2
  %6 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  tail call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

7:                                                ; preds = %2
  resume { ptr, i32 } %3
}

; Function Attrs: nonlazybind uwtable
define hidden void @"_ZN81_$LT$$RF$$u5b$u8$u5d$$u20$as$u20$alloc..ffi..c_str..CString..new..SpecNewImpl$GT$13spec_new_impl17hd42ed46981b64645E"(ptr dead_on_unwind noalias noundef writable writeonly sret([32 x i8]) align 8 captures(none) dereferenceable(32) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2) unnamed_addr #20 personality ptr @rust_eh_personality {
  %4 = alloca [24 x i8], align 8
  %5 = alloca [24 x i8], align 8
  %6 = alloca [24 x i8], align 8
  %7 = alloca [24 x i8], align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !394)
  %8 = icmp eq i64 %2, -1
  br i1 %8, label %24, label %9, !prof !353

9:                                                ; preds = %3
  %10 = add nuw i64 %2, 1
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %7), !noalias !397
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %5), !noalias !397
  call void @"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$15try_allocate_in17h47e3396744964f2cE"(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %5, i64 noundef %10, i1 noundef zeroext false, i64 noundef 1, i64 noundef 1), !noalias !397
  %11 = load i64, ptr %5, align 8, !range !359, !noalias !397, !noundef !29
  %12 = trunc nuw i64 %11 to i1
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %14 = load i64, ptr %13, align 8, !range !360, !noalias !397, !noundef !29
  %15 = getelementptr inbounds nuw i8, ptr %5, i64 16
  br i1 %12, label %16, label %18, !prof !353

16:                                               ; preds = %9
  %17 = load i64, ptr %15, align 8, !noalias !397
  call void @_ZN5alloc7raw_vec12handle_error17h2546acf93648cb86E(i64 noundef %14, i64 %17) #53, !noalias !397
  unreachable

18:                                               ; preds = %9
  %19 = load ptr, ptr %15, align 8, !noalias !397, !nonnull !29, !noundef !29
  %20 = icmp ult i64 %2, %14
  call void @llvm.assume(i1 %20), !noalias !397
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %5), !noalias !397
  store i64 %14, ptr %7, align 8, !noalias !397
  %21 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store ptr %19, ptr %21, align 8, !noalias !397
  %22 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store i64 0, ptr %22, align 8, !noalias !397
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 %2
  invoke void @"_ZN132_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$alloc..vec..spec_extend..SpecExtend$LT$$RF$T$C$core..slice..iter..Iter$LT$T$GT$$GT$$GT$11spec_extend17h12d2578eb1732a16E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %7, ptr noundef nonnull readonly align 1 %1, ptr noundef nonnull readonly %23)
          to label %25 unwind label %75, !noalias !394

24:                                               ; preds = %3
  tail call void @_ZN4core6option13unwrap_failed17h185c81919a7695e2E(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.677bc98fa4f847c2113641a77bae611c.3) #53, !noalias !397
  unreachable

25:                                               ; preds = %18
  %26 = icmp ult i64 %2, 16
  br i1 %26, label %27, label %29

27:                                               ; preds = %25
  %28 = icmp eq i64 %2, 0
  br i1 %28, label %31, label %36

29:                                               ; preds = %25
  %30 = invoke { i64, i64 } @_ZN4core5slice6memchr14memchr_aligned17heb4bdeeb017d10a2E(i8 noundef 0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2)
          to label %44 unwind label %75

31:                                               ; preds = %41, %36, %27
  %32 = phi i64 [ 0, %27 ], [ %2, %41 ], [ %37, %36 ]
  %33 = phi i64 [ 0, %27 ], [ 0, %41 ], [ 1, %36 ]
  %34 = insertvalue { i64, i64 } poison, i64 %33, 0
  %35 = insertvalue { i64, i64 } %34, i64 %32, 1
  br label %44

36:                                               ; preds = %41, %27
  %37 = phi i64 [ %42, %41 ], [ 0, %27 ]
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 %37
  %39 = load i8, ptr %38, align 1, !alias.scope !399, !noalias !394, !noundef !29
  %40 = icmp eq i8 %39, 0
  br i1 %40, label %31, label %41

41:                                               ; preds = %36
  %42 = add nuw i64 %37, 1
  %43 = icmp eq i64 %42, %2
  br i1 %43, label %31, label %36

44:                                               ; preds = %31, %29
  %45 = phi { i64, i64 } [ %35, %31 ], [ %30, %29 ]
  %46 = extractvalue { i64, i64 } %45, 0
  %47 = trunc nuw i64 %46 to i1
  br i1 %47, label %48, label %51

48:                                               ; preds = %44
  %49 = extractvalue { i64, i64 } %45, 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %0, ptr noundef nonnull align 8 dereferenceable(24) %7, i64 24, i1 false), !noalias !402
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store i64 %49, ptr %50, align 8, !alias.scope !394, !noalias !402
  br label %79

51:                                               ; preds = %44
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %6), !noalias !397
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %6, ptr noundef nonnull align 8 dereferenceable(24) %7, i64 24, i1 false), !noalias !397
  call void @llvm.experimental.noalias.scope.decl(metadata !403)
  %52 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %53 = load i64, ptr %52, align 8, !alias.scope !403, !noalias !394, !noundef !29
  invoke void @"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$13reserve_exact17h7f250ee6644f84e7E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %6, i64 noundef %53, i64 noundef 1, i64 noundef 1, i64 noundef 1)
          to label %54 unwind label %69, !noalias !394

54:                                               ; preds = %51
  %55 = load i64, ptr %52, align 8, !alias.scope !406, !noalias !394, !noundef !29
  %56 = load i64, ptr %6, align 8, !range !354, !alias.scope !406, !noalias !394, !noundef !29
  %57 = icmp eq i64 %55, %56
  br i1 %57, label %58, label %59

58:                                               ; preds = %54
  invoke void @"_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h34bd7e99b107a31bE"(ptr noalias noundef nonnull align 8 dereferenceable(24) %6)
          to label %59 unwind label %69, !noalias !394

59:                                               ; preds = %58, %54
  %60 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %61 = load ptr, ptr %60, align 8, !alias.scope !406, !noalias !394, !nonnull !29, !noundef !29
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 %55
  store i8 0, ptr %62, align 1, !noalias !394
  %63 = add i64 %55, 1
  store i64 %63, ptr %52, align 8, !alias.scope !406, !noalias !394
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4), !noalias !409
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %4, ptr noundef nonnull align 8 dereferenceable(24) %6, i64 24, i1 false), !noalias !394
  %64 = call { ptr, i64 } @"_ZN5alloc3vec16Vec$LT$T$C$A$GT$16into_boxed_slice17hd30854b51e9f897aE"(ptr noalias noundef nonnull align 8 captures(address) dereferenceable(24) %4)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4), !noalias !409
  %65 = extractvalue { ptr, i64 } %64, 0
  %66 = extractvalue { ptr, i64 } %64, 1
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %6), !noalias !397
  %67 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %65, ptr %67, align 8, !alias.scope !394, !noalias !402
  %68 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %66, ptr %68, align 8, !alias.scope !394, !noalias !402
  store i64 -9223372036854775808, ptr %0, align 8, !alias.scope !394, !noalias !402
  br label %79

69:                                               ; preds = %58, %51
  %70 = landingpad { ptr, i32 }
          cleanup
  invoke fastcc void @"_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17h71ba09309df63a21E"(ptr noalias noundef align 8 dereferenceable(24) %6) #56
          to label %73 unwind label %71, !noalias !394

71:                                               ; preds = %69
  %72 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55, !noalias !394
  unreachable

73:                                               ; preds = %75, %69
  %74 = phi { ptr, i32 } [ %76, %75 ], [ %70, %69 ]
  resume { ptr, i32 } %74

75:                                               ; preds = %29, %18
  %76 = landingpad { ptr, i32 }
          cleanup
  invoke fastcc void @"_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17h71ba09309df63a21E"(ptr noalias noundef align 8 dereferenceable(24) %7) #56
          to label %73 unwind label %77, !noalias !394

77:                                               ; preds = %75
  %78 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55, !noalias !394
  unreachable

79:                                               ; preds = %59, %48
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %7), !noalias !397
  ret void
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare i1 @llvm.is.constant.i1(i1) #24

; Function Attrs: cold minsize noreturn nonlazybind optsize uwtable
define hidden void @_ZN5alloc5alloc18handle_alloc_error17h1bbcba5314f57599E(i64 noundef range(i64 1, -9223372036854775807) %0, i64 noundef %1) unnamed_addr #18 {
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc26___rust_alloc_error_handler(i64 noundef %1, i64 noundef %0) #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u32$GT$3fmt17h9bc5442fdc664a66E"(ptr noalias noundef readonly align 4 captures(none) dereferenceable(4), ptr noalias noundef align 8 dereferenceable(24)) unnamed_addr #20

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u64$GT$3fmt17hb8c6372145dbb21aE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8), ptr noalias noundef align 8 dereferenceable(24)) unnamed_addr #20

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8), ptr noalias noundef align 8 dereferenceable(24)) unnamed_addr #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64) #3

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden void @_ZN4core6result13unwrap_failed17hf9aa201d733198edE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %0, i64 noundef %1, ptr noundef nonnull align 1 %2, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(32) %3, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %4) unnamed_addr #19 {
  %6 = alloca [32 x i8], align 8
  %7 = alloca [48 x i8], align 8
  %8 = alloca [16 x i8], align 8
  %9 = alloca [16 x i8], align 8
  store ptr %0, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store i64 %1, ptr %10, align 8
  store ptr %2, ptr %8, align 8
  %11 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store ptr %3, ptr %11, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %7)
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %6)
  store ptr %9, ptr %6, align 8
  %12 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h3e220fe9902f5bc4E", ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store ptr %8, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store ptr @"_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hfbb51a43fdfe804aE", ptr %14, align 8
  store ptr @anon.e8ad111fcdba8f51c5e77cb80a764505.26, ptr %7, align 8
  %15 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i64 2, ptr %15, align 8
  %16 = getelementptr inbounds nuw i8, ptr %7, i64 32
  store ptr null, ptr %16, align 8
  %17 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store ptr %6, ptr %17, align 8
  %18 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store i64 2, ptr %18, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %7, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %4) #53
  unreachable
}

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden void @_ZN4core5slice5index16slice_index_fail17hf7a05389aea37f33E(i64 noundef %0, i64 noundef %1, i64 noundef %2, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %3) unnamed_addr #19 {
  %5 = icmp ugt i64 %0, %2
  br i1 %5, label %8, label %6

6:                                                ; preds = %4
  %7 = icmp ugt i64 %1, %2
  br i1 %7, label %11, label %9

8:                                                ; preds = %4
  tail call void @_ZN4core5slice5index16slice_index_fail8do_panic7runtime17h0ed9b75a9908ef13E(i64 noundef %0, i64 noundef %2, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %3) #53
  unreachable

9:                                                ; preds = %6
  %10 = icmp ugt i64 %0, %1
  br i1 %10, label %13, label %12

11:                                               ; preds = %6
  tail call void @_ZN4core5slice5index16slice_index_fail8do_panic7runtime17h39d1db38294b7f9aE(i64 noundef %1, i64 noundef %2, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %3) #53
  unreachable

12:                                               ; preds = %9
  tail call void @_ZN4core5slice5index16slice_index_fail8do_panic7runtime17h27e1d069eeae91b9E(i64 noundef %1, i64 noundef %2, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %3) #53
  unreachable

13:                                               ; preds = %9
  tail call void @_ZN4core5slice5index16slice_index_fail8do_panic7runtime17h7d0ef4262754f28bE(i64 noundef %0, i64 noundef %1, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %3) #53
  unreachable
}

; Function Attrs: cold noreturn nonlazybind uwtable
define hidden void @_ZN4core5slice5index16slice_index_fail8do_panic7runtime17h0ed9b75a9908ef13E(i64 noundef %0, i64 noundef %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) unnamed_addr #25 {
  %4 = alloca [32 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = alloca [8 x i8], align 8
  %7 = alloca [8 x i8], align 8
  store i64 %0, ptr %7, align 8
  store i64 %1, ptr %6, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %4)
  store ptr %7, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %6, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %10, align 8
  store ptr @anon.080693e073b7ed74ca255f40e933ae70.59, ptr %5, align 8
  %11 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 2, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %4, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 2, ptr %14, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %5, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) #53
  unreachable
}

; Function Attrs: cold noreturn nonlazybind uwtable
define hidden void @_ZN4core5slice5index16slice_index_fail8do_panic7runtime17h39d1db38294b7f9aE(i64 noundef %0, i64 noundef %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) unnamed_addr #25 {
  %4 = alloca [32 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = alloca [8 x i8], align 8
  %7 = alloca [8 x i8], align 8
  store i64 %0, ptr %7, align 8
  store i64 %1, ptr %6, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %4)
  store ptr %7, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %6, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %10, align 8
  store ptr @anon.080693e073b7ed74ca255f40e933ae70.61, ptr %5, align 8
  %11 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 2, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %4, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 2, ptr %14, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %5, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) #53
  unreachable
}

; Function Attrs: cold noreturn nonlazybind uwtable
define hidden void @_ZN4core5slice5index16slice_index_fail8do_panic7runtime17h27e1d069eeae91b9E(i64 noundef %0, i64 noundef %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) unnamed_addr #25 {
  %4 = alloca [32 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = alloca [8 x i8], align 8
  %7 = alloca [8 x i8], align 8
  store i64 %0, ptr %7, align 8
  store i64 %1, ptr %6, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %4)
  store ptr %7, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %6, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %10, align 8
  store ptr @anon.080693e073b7ed74ca255f40e933ae70.61, ptr %5, align 8
  %11 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 2, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %4, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 2, ptr %14, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %5, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) #53
  unreachable
}

; Function Attrs: cold noreturn nonlazybind uwtable
define hidden void @_ZN4core5slice5index16slice_index_fail8do_panic7runtime17h7d0ef4262754f28bE(i64 noundef %0, i64 noundef %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) unnamed_addr #25 {
  %4 = alloca [32 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = alloca [8 x i8], align 8
  %7 = alloca [8 x i8], align 8
  store i64 %0, ptr %7, align 8
  store i64 %1, ptr %6, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %4)
  store ptr %7, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %6, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %10, align 8
  store ptr @anon.080693e073b7ed74ca255f40e933ae70.64, ptr %5, align 8
  %11 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 2, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %4, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 2, ptr %14, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %5, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) #53
  unreachable
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(inaccessiblemem: write) uwtable
define hidden noundef range(i64 0, -1) i64 @_ZN4core3ptr12align_offset17h42b7cc0bf94364a3E(ptr noundef %0, i64 noundef %1) unnamed_addr #26 {
  %3 = add i64 %1, -1
  %4 = ptrtoint ptr %0 to i64
  %5 = add i64 %3, %4
  %6 = sub i64 0, %1
  %7 = and i64 %5, %6
  %8 = sub i64 %7, %4
  %9 = icmp ult i64 %8, %1
  tail call void @llvm.assume(i1 %9)
  ret i64 %8
}

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr dead_on_return noalias noundef readonly align 8 captures(address) dereferenceable(48) %0, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %1) unnamed_addr #19 {
  %3 = alloca [24 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %3)
  store ptr %0, ptr %3, align 8
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %1, ptr %4, align 8
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i8 1, ptr %5, align 8
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 17
  store i8 0, ptr %6, align 1
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc17rust_begin_unwind(ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %3) #53
  unreachable
}

; Function Attrs: cold noinline noreturn nounwind nonlazybind uwtable
define hidden void @_ZN4core9panicking14panic_nounwind17h24cc238e0985977fE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %0, i64 noundef %1) unnamed_addr #27 {
  %3 = alloca [16 x i8], align 8
  %4 = alloca [48 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %4)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store ptr %0, ptr %3, align 8
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 %1, ptr %5, align 8
  store ptr %3, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 1, ptr %6, align 8
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store ptr null, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i64 0, ptr %9, align 8
  call void @_ZN4core9panicking18panic_nounwind_fmt17hc0b47d79028b5977E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %4, i1 noundef zeroext false, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.8f5ef5ec36c943a0d1df6b6363c9934d.6) #54
  unreachable
}

; Function Attrs: cold noinline noreturn nounwind nonlazybind uwtable
define hidden void @_ZN4core9panicking18panic_nounwind_fmt17hc0b47d79028b5977E(ptr dead_on_return noalias noundef readonly align 8 captures(none) dereferenceable(48) %0, i1 noundef zeroext %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) unnamed_addr #27 personality ptr @rust_eh_personality {
  %4 = alloca [24 x i8], align 8
  %5 = alloca [48 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) %5, ptr noundef nonnull align 8 dereferenceable(48) %0, i64 48, i1 false)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4)
  store ptr %5, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %2, ptr %6, align 8
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i8 0, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 17
  %9 = zext i1 %1 to i8
  store i8 %9, ptr %8, align 1
  invoke void @_RNvCs1QLEhZ2QfLZ_7___rustc17rust_begin_unwind(ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %4) #53
          to label %12 unwind label %10

10:                                               ; preds = %3
  %11 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking19panic_cannot_unwind17h490ff4ea5b3a81fcE() #55
  unreachable

12:                                               ; preds = %3
  unreachable
}

; Function Attrs: cold minsize noinline noreturn nounwind nonlazybind optsize uwtable
define hidden void @_ZN4core9panicking19panic_cannot_unwind17h490ff4ea5b3a81fcE() unnamed_addr #28 {
  tail call void @_ZN4core9panicking14panic_nounwind17h24cc238e0985977fE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.8f5ef5ec36c943a0d1df6b6363c9934d.23, i64 noundef 38) #54
  unreachable
}

; Function Attrs: cold minsize noinline noreturn nounwind nonlazybind optsize uwtable
define hidden void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() unnamed_addr #28 {
  tail call void @_ZN4core9panicking26panic_nounwind_nobacktrace17h866eed907602bf2eE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.8f5ef5ec36c943a0d1df6b6363c9934d.9, i64 noundef 36) #54
  unreachable
}

; Function Attrs: cold noinline noreturn nounwind nonlazybind uwtable
define hidden void @_ZN4core9panicking26panic_nounwind_nobacktrace17h866eed907602bf2eE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %0, i64 noundef %1) unnamed_addr #27 {
  %3 = alloca [16 x i8], align 8
  %4 = alloca [48 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %4)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store ptr %0, ptr %3, align 8
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 %1, ptr %5, align 8
  store ptr %3, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 1, ptr %6, align 8
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store ptr null, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i64 0, ptr %9, align 8
  call void @_ZN4core9panicking18panic_nounwind_fmt17hc0b47d79028b5977E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %4, i1 noundef zeroext true, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.8f5ef5ec36c943a0d1df6b6363c9934d.24) #54
  unreachable
}

; Function Attrs: cold minsize noinline noreturn nonlazybind optsize uwtable
define hidden void @_ZN4core9panicking18panic_bounds_check17h74d7c20cf9e1a9efE(i64 noundef %0, i64 noundef %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) unnamed_addr #29 {
  %4 = alloca [32 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = alloca [8 x i8], align 8
  %7 = alloca [8 x i8], align 8
  store i64 %0, ptr %7, align 8
  store i64 %1, ptr %6, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %4)
  store ptr %6, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %7, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr @"_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17h5ab573645e1becf5E", ptr %10, align 8
  store ptr @anon.8f5ef5ec36c943a0d1df6b6363c9934d.12, ptr %5, align 8
  %11 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 2, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %4, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 2, ptr %14, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %5, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) #53
  unreachable
}

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden void @_ZN4core9panicking5panic17h80104d8006620ef0E(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %0, i64 noundef %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) unnamed_addr #19 {
  %4 = alloca [16 x i8], align 8
  %5 = alloca [48 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4)
  store ptr %0, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 %1, ptr %6, align 8
  store ptr %4, ptr %5, align 8
  %7 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 1, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 0, ptr %10, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %5, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN42_$LT$$RF$T$u20$as$u20$core..fmt..Debug$GT$3fmt17hfbb51a43fdfe804aE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(16), ptr noalias noundef align 8 dereferenceable(24)) unnamed_addr #20

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h3e220fe9902f5bc4E"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(16), ptr noalias noundef readonly align 8 captures(none) dereferenceable(24)) unnamed_addr #20

; Function Attrs: nonlazybind uwtable
define hidden noundef zeroext i1 @_ZN4core3fmt5write17h919175a03bb9497fE(ptr noundef nonnull align 1 %0, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) %1, ptr dead_on_return noalias noundef readonly align 8 captures(none) dereferenceable(48) %2) unnamed_addr #20 personality ptr @rust_eh_personality {
  %4 = alloca [24 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4)
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 -536870880, ptr %5, align 8
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i16 0, ptr %6, align 4
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 22
  store i16 0, ptr %7, align 2
  store ptr %0, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %1, ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %10 = load ptr, ptr %9, align 8, !align !410, !noundef !29
  %11 = icmp eq ptr %10, null
  br i1 %11, label %26, label %12

12:                                               ; preds = %3
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 40
  %14 = load i64, ptr %13, align 8, !noundef !29
  %15 = mul nuw nsw i64 %14, 48
  %16 = getelementptr inbounds nuw i8, ptr %10, i64 %15
  %17 = icmp eq i64 %14, 0
  br i1 %17, label %51, label %18

18:                                               ; preds = %12
  %19 = load ptr, ptr %2, align 8, !nonnull !29, !align !410, !noundef !29
  %20 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %21 = load i64, ptr %20, align 8, !noundef !29
  %22 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %23 = load ptr, ptr %22, align 8, !nonnull !29, !align !410
  %24 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %25 = load i64, ptr %24, align 8
  br label %70

26:                                               ; preds = %3
  %27 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %28 = load ptr, ptr %27, align 8, !nonnull !29, !align !410, !noundef !29
  %29 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %30 = load i64, ptr %29, align 8, !noundef !29
  %31 = shl i64 %30, 4
  %32 = getelementptr inbounds nuw i8, ptr %28, i64 %31
  %33 = icmp eq i64 %30, 0
  br i1 %33, label %51, label %34

34:                                               ; preds = %26
  %35 = load ptr, ptr %2, align 8, !nonnull !29, !align !410, !noundef !29
  %36 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %37 = load i64, ptr %36, align 8, !noundef !29
  %38 = add i64 %31, -16
  %39 = lshr exact i64 %38, 4
  %40 = add nuw nsw i64 %39, 1
  br label %41

41:                                               ; preds = %68, %34
  %42 = phi i64 [ 0, %34 ], [ %45, %68 ]
  %43 = phi ptr [ %28, %34 ], [ %44, %68 ]
  %44 = getelementptr inbounds nuw i8, ptr %43, i64 16
  %45 = add nuw nsw i64 %42, 1
  %46 = icmp ult i64 %42, %37
  call void @llvm.assume(i1 %46)
  %47 = getelementptr inbounds nuw { ptr, i64 }, ptr %35, i64 %42
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 8
  %49 = load i64, ptr %48, align 8, !noundef !29
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %63, label %56

51:                                               ; preds = %144, %68, %26, %12
  %52 = phi i64 [ 0, %26 ], [ 0, %12 ], [ %40, %68 ], [ %74, %144 ]
  %53 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %54 = load i64, ptr %53, align 8, !noundef !29
  %55 = icmp ult i64 %52, %54
  br i1 %55, label %80, label %91

56:                                               ; preds = %41
  %57 = load ptr, ptr %4, align 8, !nonnull !29, !align !411, !noundef !29
  %58 = load ptr, ptr %8, align 8, !nonnull !29, !align !410, !noundef !29
  %59 = load ptr, ptr %47, align 8, !nonnull !29, !align !411, !noundef !29
  %60 = getelementptr inbounds nuw i8, ptr %58, i64 24
  %61 = load ptr, ptr %60, align 8, !invariant.load !29, !nonnull !29
  %62 = call noundef zeroext i1 %61(ptr noundef nonnull align 1 %57, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %59, i64 noundef %49)
  br i1 %62, label %92, label %63

63:                                               ; preds = %56, %41
  %64 = load ptr, ptr %43, align 8, !nonnull !29, !noundef !29
  %65 = getelementptr inbounds nuw i8, ptr %43, i64 8
  %66 = load ptr, ptr %65, align 8, !nonnull !29, !noundef !29
  %67 = call noundef zeroext i1 %66(ptr noundef nonnull %64, ptr noalias noundef nonnull align 8 dereferenceable(24) %4)
  br i1 %67, label %92, label %68

68:                                               ; preds = %63
  %69 = icmp eq ptr %44, %32
  br i1 %69, label %51, label %41

70:                                               ; preds = %144, %18
  %71 = phi i64 [ 0, %18 ], [ %74, %144 ]
  %72 = phi ptr [ %10, %18 ], [ %73, %144 ]
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 48
  %74 = add nuw nsw i64 %71, 1
  %75 = icmp ult i64 %71, %21
  call void @llvm.assume(i1 %75)
  %76 = getelementptr inbounds nuw { ptr, i64 }, ptr %19, i64 %71
  %77 = getelementptr inbounds nuw i8, ptr %76, i64 8
  %78 = load i64, ptr %77, align 8, !noundef !29
  %79 = icmp eq i64 %78, 0
  br i1 %79, label %101, label %94

80:                                               ; preds = %51
  %81 = load ptr, ptr %2, align 8, !nonnull !29, !align !410
  %82 = getelementptr inbounds nuw { ptr, i64 }, ptr %81, i64 %52
  %83 = load ptr, ptr %4, align 8, !nonnull !29, !align !411, !noundef !29
  %84 = load ptr, ptr %8, align 8, !nonnull !29, !align !410, !noundef !29
  %85 = load ptr, ptr %82, align 8, !nonnull !29, !align !411, !noundef !29
  %86 = getelementptr inbounds nuw i8, ptr %82, i64 8
  %87 = load i64, ptr %86, align 8, !noundef !29
  %88 = getelementptr inbounds nuw i8, ptr %84, i64 24
  %89 = load ptr, ptr %88, align 8, !invariant.load !29, !nonnull !29
  %90 = call noundef zeroext i1 %89(ptr noundef nonnull align 1 %83, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %85, i64 noundef %87)
  br i1 %90, label %92, label %91

91:                                               ; preds = %80, %51
  br label %92

92:                                               ; preds = %132, %94, %91, %80, %63, %56
  %93 = phi i1 [ false, %91 ], [ true, %94 ], [ true, %132 ], [ true, %56 ], [ true, %63 ], [ true, %80 ]
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4)
  ret i1 %93

94:                                               ; preds = %70
  %95 = load ptr, ptr %4, align 8, !nonnull !29, !align !411, !noundef !29
  %96 = load ptr, ptr %8, align 8, !nonnull !29, !align !410, !noundef !29
  %97 = load ptr, ptr %76, align 8, !nonnull !29, !align !411, !noundef !29
  %98 = getelementptr inbounds nuw i8, ptr %96, i64 24
  %99 = load ptr, ptr %98, align 8, !invariant.load !29, !nonnull !29
  %100 = call noundef zeroext i1 %99(ptr noundef nonnull align 1 %95, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %97, i64 noundef %78)
  br i1 %100, label %92, label %101

101:                                              ; preds = %94, %70
  call void @llvm.experimental.noalias.scope.decl(metadata !412)
  call void @llvm.experimental.noalias.scope.decl(metadata !415)
  call void @llvm.experimental.noalias.scope.decl(metadata !417)
  %102 = getelementptr inbounds nuw i8, ptr %72, i64 16
  call void @llvm.experimental.noalias.scope.decl(metadata !419)
  call void @llvm.experimental.noalias.scope.decl(metadata !422)
  %103 = load i16, ptr %102, align 8, !range !424, !alias.scope !425, !noalias !426, !noundef !29
  switch i16 %103, label %104 [
    i16 0, label %105
    i16 1, label %108
    i16 2, label %117
  ]

104:                                              ; preds = %117, %101
  unreachable

105:                                              ; preds = %101
  %106 = getelementptr inbounds nuw i8, ptr %72, i64 18
  %107 = load i16, ptr %106, align 2, !alias.scope !425, !noalias !426, !noundef !29
  br label %117

108:                                              ; preds = %101
  %109 = getelementptr inbounds nuw i8, ptr %72, i64 24
  %110 = load i64, ptr %109, align 8, !alias.scope !425, !noalias !426, !noundef !29
  %111 = icmp ult i64 %110, %25
  call void @llvm.assume(i1 %111)
  %112 = getelementptr inbounds nuw { { ptr, [1 x i64] } }, ptr %23, i64 %110
  %113 = load ptr, ptr %112, align 8, !alias.scope !427, !noalias !428, !noundef !29
  %114 = icmp eq ptr %113, null
  call void @llvm.assume(i1 %114)
  %115 = getelementptr inbounds nuw i8, ptr %112, i64 8
  %116 = load i16, ptr %115, align 8, !alias.scope !427, !noalias !428, !noundef !29
  br label %117

117:                                              ; preds = %108, %105, %101
  %118 = phi i16 [ %107, %105 ], [ %116, %108 ], [ 0, %101 ]
  call void @llvm.experimental.noalias.scope.decl(metadata !429)
  call void @llvm.experimental.noalias.scope.decl(metadata !432)
  %119 = load i16, ptr %72, align 8, !range !424, !alias.scope !434, !noalias !435, !noundef !29
  switch i16 %119, label %104 [
    i16 0, label %120
    i16 1, label %123
    i16 2, label %132
  ]

120:                                              ; preds = %117
  %121 = getelementptr inbounds nuw i8, ptr %72, i64 2
  %122 = load i16, ptr %121, align 2, !alias.scope !434, !noalias !435, !noundef !29
  br label %132

123:                                              ; preds = %117
  %124 = getelementptr inbounds nuw i8, ptr %72, i64 8
  %125 = load i64, ptr %124, align 8, !alias.scope !434, !noalias !435, !noundef !29
  %126 = icmp ult i64 %125, %25
  call void @llvm.assume(i1 %126)
  %127 = getelementptr inbounds nuw { { ptr, [1 x i64] } }, ptr %23, i64 %125
  %128 = load ptr, ptr %127, align 8, !alias.scope !436, !noalias !437, !noundef !29
  %129 = icmp eq ptr %128, null
  call void @llvm.assume(i1 %129)
  %130 = getelementptr inbounds nuw i8, ptr %127, i64 8
  %131 = load i16, ptr %130, align 8, !alias.scope !436, !noalias !437, !noundef !29
  br label %132

132:                                              ; preds = %123, %120, %117
  %133 = phi i16 [ %122, %120 ], [ %131, %123 ], [ 0, %117 ]
  %134 = getelementptr inbounds nuw i8, ptr %72, i64 40
  %135 = load i32, ptr %134, align 8, !alias.scope !415, !noalias !438, !noundef !29
  %136 = getelementptr inbounds nuw i8, ptr %72, i64 32
  %137 = load i64, ptr %136, align 8, !alias.scope !415, !noalias !438, !noundef !29
  %138 = icmp ult i64 %137, %25
  call void @llvm.assume(i1 %138)
  %139 = getelementptr inbounds nuw { { ptr, [1 x i64] } }, ptr %23, i64 %137
  store i32 %135, ptr %5, align 8, !alias.scope !412, !noalias !439
  store i16 %118, ptr %6, align 4, !alias.scope !412, !noalias !439
  store i16 %133, ptr %7, align 2, !alias.scope !412, !noalias !439
  %140 = load ptr, ptr %139, align 8, !alias.scope !417, !noalias !440, !nonnull !29, !noundef !29
  %141 = getelementptr inbounds nuw i8, ptr %139, i64 8
  %142 = load ptr, ptr %141, align 8, !alias.scope !417, !noalias !440, !nonnull !29, !noundef !29
  %143 = call noundef zeroext i1 %142(ptr noundef nonnull %140, ptr noalias noundef nonnull align 8 dereferenceable(24) %4), !noalias !439
  br i1 %143, label %92, label %144

144:                                              ; preds = %132
  %145 = icmp eq ptr %73, %16
  br i1 %145, label %51, label %70
}

; Function Attrs: nonlazybind uwtable
define hidden { i64, i64 } @_ZN4core5slice6memchr14memchr_aligned17heb4bdeeb017d10a2E(i8 noundef %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2) unnamed_addr #20 personality ptr @rust_eh_personality {
  %4 = tail call noundef i64 @_ZN4core3ptr12align_offset17h42b7cc0bf94364a3E(ptr noundef nonnull readonly align 1 %1, i64 noundef 8)
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %6, label %8

6:                                                ; preds = %3
  %7 = add i64 %2, -16
  br label %14

8:                                                ; preds = %3
  %9 = tail call noundef i64 @llvm.umin.i64(i64 %2, i64 range(i64 1, 0) %4)
  %10 = icmp eq i64 %2, 0
  br i1 %10, label %11, label %52

11:                                               ; preds = %57, %8
  %12 = add i64 %2, -16
  %13 = icmp ugt i64 %9, %12
  br i1 %13, label %34, label %14

14:                                               ; preds = %11, %6
  %15 = phi i64 [ %7, %6 ], [ %12, %11 ]
  %16 = phi i64 [ 0, %6 ], [ %9, %11 ]
  %17 = zext i8 %0 to i64
  %18 = mul nuw i64 %17, 72340172838076673
  br label %19

19:                                               ; preds = %39, %14
  %20 = phi i64 [ %40, %39 ], [ %16, %14 ]
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 %20
  %22 = load i64, ptr %21, align 8, !alias.scope !441, !noundef !29
  %23 = getelementptr i8, ptr %21, i64 8
  %24 = load i64, ptr %23, align 8, !alias.scope !441, !noundef !29
  %25 = xor i64 %22, %18
  %26 = sub i64 72340172838076672, %25
  %27 = or i64 %26, %25
  %28 = xor i64 %24, %18
  %29 = sub i64 72340172838076672, %28
  %30 = or i64 %29, %28
  %31 = and i64 %27, -9187201950435737472
  %32 = and i64 %31, %30
  %33 = icmp eq i64 %32, -9187201950435737472
  br i1 %33, label %39, label %34

34:                                               ; preds = %39, %19, %11
  %35 = phi i64 [ %9, %11 ], [ %40, %39 ], [ %20, %19 ]
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 %35
  %37 = sub i64 %2, %35
  %38 = icmp eq i64 %2, %35
  br i1 %38, label %60, label %42

39:                                               ; preds = %19
  %40 = add i64 %20, 16
  %41 = icmp ugt i64 %40, %15
  br i1 %41, label %34, label %19

42:                                               ; preds = %47, %34
  %43 = phi i64 [ %48, %47 ], [ 0, %34 ]
  %44 = getelementptr inbounds nuw i8, ptr %36, i64 %43
  %45 = load i8, ptr %44, align 1, !alias.scope !441, !noundef !29
  %46 = icmp eq i8 %45, %0
  br i1 %46, label %50, label %47

47:                                               ; preds = %42
  %48 = add nuw i64 %43, 1
  %49 = icmp eq i64 %48, %37
  br i1 %49, label %60, label %42

50:                                               ; preds = %42
  %51 = add i64 %43, %35
  br label %60

52:                                               ; preds = %57, %8
  %53 = phi i64 [ %58, %57 ], [ 0, %8 ]
  %54 = getelementptr inbounds nuw i8, ptr %1, i64 %53
  %55 = load i8, ptr %54, align 1, !alias.scope !441, !noundef !29
  %56 = icmp eq i8 %55, %0
  br i1 %56, label %60, label %57

57:                                               ; preds = %52
  %58 = add nuw i64 %53, 1
  %59 = icmp eq i64 %58, %9
  br i1 %59, label %11, label %52

60:                                               ; preds = %52, %50, %47, %34
  %61 = phi i64 [ %51, %50 ], [ undef, %34 ], [ undef, %47 ], [ %53, %52 ]
  %62 = phi i64 [ 1, %50 ], [ 0, %34 ], [ 0, %47 ], [ 1, %52 ]
  %63 = insertvalue { i64, i64 } poison, i64 %62, 0
  %64 = insertvalue { i64, i64 } %63, i64 %61, 1
  ret { i64, i64 } %64
}

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden void @_ZN4core6option13expect_failed17hb93f3b2511d2da6dE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %0, i64 noundef %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) unnamed_addr #19 {
  %4 = alloca [16 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = alloca [16 x i8], align 8
  store ptr %0, ptr %6, align 8
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i64 %1, ptr %7, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4)
  store ptr %6, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h3e220fe9902f5bc4E", ptr %8, align 8
  store ptr @anon.8afe5fb67ae3f1afa5e6dc92f8599610.14, ptr %5, align 8
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 1, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %10, align 8
  %11 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %4, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 1, ptr %12, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %5, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %2) #53
  unreachable
}

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden void @_ZN4core6option13unwrap_failed17h185c81919a7695e2E(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %0) unnamed_addr #19 {
  tail call void @_ZN4core9panicking5panic17h80104d8006620ef0E(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.8afe5fb67ae3f1afa5e6dc92f8599610.15, i64 noundef 43, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %0) #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN4core3ffi5c_str4CStr19from_bytes_with_nul17h883c9be065a77d17E(ptr dead_on_unwind noalias noundef writable writeonly sret([24 x i8]) align 8 captures(none) dereferenceable(24) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2) unnamed_addr #20 {
  %4 = icmp ult i64 %2, 16
  br i1 %4, label %5, label %7

5:                                                ; preds = %3
  %6 = icmp eq i64 %2, 0
  br i1 %6, label %9, label %14

7:                                                ; preds = %3
  %8 = tail call { i64, i64 } @_ZN4core5slice6memchr14memchr_aligned17heb4bdeeb017d10a2E(i8 noundef 0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2)
  br label %22

9:                                                ; preds = %19, %14, %5
  %10 = phi i64 [ 0, %5 ], [ %2, %19 ], [ %15, %14 ]
  %11 = phi i64 [ 0, %5 ], [ 0, %19 ], [ 1, %14 ]
  %12 = insertvalue { i64, i64 } poison, i64 %11, 0
  %13 = insertvalue { i64, i64 } %12, i64 %10, 1
  br label %22

14:                                               ; preds = %19, %5
  %15 = phi i64 [ %20, %19 ], [ 0, %5 ]
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 %15
  %17 = load i8, ptr %16, align 1, !alias.scope !444, !noundef !29
  %18 = icmp eq i8 %17, 0
  br i1 %18, label %9, label %19

19:                                               ; preds = %14
  %20 = add nuw nsw i64 %15, 1
  %21 = icmp eq i64 %20, %2
  br i1 %21, label %9, label %14

22:                                               ; preds = %9, %7
  %23 = phi { i64, i64 } [ %13, %9 ], [ %8, %7 ]
  %24 = extractvalue { i64, i64 } %23, 0
  %25 = extractvalue { i64, i64 } %23, 1
  %26 = trunc nuw i64 %24 to i1
  br i1 %26, label %27, label %32

27:                                               ; preds = %22
  %28 = add i64 %25, 1
  %29 = icmp eq i64 %28, %2
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %31 = getelementptr inbounds nuw i8, ptr %0, i64 16
  br i1 %29, label %37, label %36

32:                                               ; preds = %22
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 1, ptr %33, align 8
  br label %34

34:                                               ; preds = %37, %36, %32
  %35 = phi i64 [ 0, %37 ], [ 1, %36 ], [ 1, %32 ]
  store i64 %35, ptr %0, align 8
  ret void

36:                                               ; preds = %27
  store i64 0, ptr %30, align 8
  store i64 %25, ptr %31, align 8
  br label %34

37:                                               ; preds = %27
  store ptr %1, ptr %30, align 8
  store i64 %2, ptr %31, align 8
  br label %34
}

; Function Attrs: inlinehint nonlazybind uwtable
define hidden i64 @"_ZN49_$LT$usize$u20$as$u20$core..iter..range..Step$GT$17forward_unchecked17h4a46ddd3bf136cddE"(i64 %0, i64 %1) unnamed_addr #6 !dbg !447 {
  %3 = alloca [8 x i8], align 8
  %4 = alloca [8 x i8], align 8
  store i64 %0, ptr %4, align 8
    #dbg_declare(ptr %4, !452, !DIExpression(), !454)
    #dbg_declare(ptr %4, !455, !DIExpression(), !461)
  store i64 %1, ptr %3, align 8
    #dbg_declare(ptr %3, !453, !DIExpression(), !463)
    #dbg_declare(ptr %3, !460, !DIExpression(), !464)
  br label %5, !dbg !465

5:                                                ; preds = %2
  call void @"_ZN4core3num23_$LT$impl$u20$usize$GT$13unchecked_add18precondition_check17h5021e0cd12831d11E"(i64 %0, i64 %1, ptr align 8 @4) #37, !dbg !467
  br label %6, !dbg !467

6:                                                ; preds = %5
  %7 = add nuw i64 %0, %1, !dbg !468
  ret i64 %7, !dbg !469
}

; Function Attrs: nonlazybind uwtable
define hidden noundef range(i32 0, 10) i32 @_RNvCs1QLEhZ2QfLZ_7___rustc18___rust_start_panic(ptr noundef nonnull align 1 %0, ptr noalias noundef readonly align 8 captures(none) dereferenceable(56) %1) unnamed_addr #20 personality ptr @rust_eh_personality {
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %4 = load ptr, ptr %3, align 8, !invariant.load !29, !nonnull !29
  %5 = tail call { ptr, ptr } %4(ptr noundef nonnull align 1 %0)
  %6 = extractvalue { ptr, ptr } %5, 0
  %7 = extractvalue { ptr, ptr } %5, 1
  %8 = icmp ne ptr %6, null
  tail call void @llvm.assume(i1 %8)
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #37, !noalias !470
  %9 = tail call noundef align 8 dereferenceable_or_null(56) ptr @_RNvCs1QLEhZ2QfLZ_7___rustc12___rust_alloc(i64 noundef 56, i64 noundef 8) #37, !noalias !470
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %18, !prof !353

11:                                               ; preds = %2
  invoke void @_ZN5alloc5alloc18handle_alloc_error17h1bbcba5314f57599E(i64 noundef 8, i64 noundef 56) #53
          to label %12 unwind label %13, !noalias !475

12:                                               ; preds = %11
  unreachable

13:                                               ; preds = %11
  %14 = landingpad { ptr, i32 }
          cleanup
  invoke fastcc void @"_ZN4core3ptr49drop_in_place$LT$panic_unwind..imp..Exception$GT$17h9f5a13717cc43ee5E"(ptr nonnull align 1 %6, ptr nonnull readonly align 8 dereferenceable(32) %7) #56
          to label %17 unwind label %15

15:                                               ; preds = %13
  %16 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  tail call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55, !noalias !475
  unreachable

17:                                               ; preds = %13
  resume { ptr, i32 } %14

18:                                               ; preds = %2
  store i64 6076294132934528845, ptr %9, align 8, !noalias !475
  %19 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store ptr @_ZN12panic_unwind3imp5panic17exception_cleanup17h3e1ca1354f737ef9E, ptr %19, align 8, !noalias !475
  %20 = getelementptr inbounds nuw i8, ptr %9, i64 16
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %20, i8 0, i64 16, i1 false), !noalias !475
  %21 = getelementptr inbounds nuw i8, ptr %9, i64 32
  store ptr @_ZN12panic_unwind3imp6CANARY17hc59233b2ced56dcdE, ptr %21, align 8, !noalias !475
  %22 = getelementptr inbounds nuw i8, ptr %9, i64 40
  store ptr %6, ptr %22, align 8, !noalias !475
  %23 = getelementptr inbounds nuw i8, ptr %9, i64 48
  store ptr %7, ptr %23, align 8, !noalias !475
  %24 = tail call noundef range(i32 0, 10) i32 @_Unwind_RaiseException(ptr noundef nonnull %9)
  ret i32 %24
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr49drop_in_place$LT$panic_unwind..imp..Exception$GT$17h9f5a13717cc43ee5E"(ptr %0, ptr readonly captures(address_is_null) %1) unnamed_addr #20 personality ptr @rust_eh_personality {
  %3 = icmp ne ptr %1, null
  tail call void @llvm.assume(i1 %3)
  %4 = load ptr, ptr %1, align 8, !invariant.load !29
  %5 = icmp eq ptr %4, null
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = icmp ne ptr %0, null
  tail call void @llvm.assume(i1 %7)
  invoke void %4(ptr noundef nonnull %0)
          to label %8 unwind label %18

8:                                                ; preds = %6, %2
  %9 = icmp ne ptr %0, null
  tail call void @llvm.assume(i1 %9)
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i64, ptr %10, align 8, !range !354, !invariant.load !29
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !range !476, !invariant.load !29
  %14 = add i64 %13, -1
  %15 = icmp sgt i64 %14, -1
  tail call void @llvm.assume(i1 %15)
  %16 = icmp eq i64 %11, 0
  br i1 %16, label %29, label %17

17:                                               ; preds = %8
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %0, i64 noundef range(i64 1, -9223372036854775808) %11, i64 noundef range(i64 1, -9223372036854775807) %13) #37
  br label %29

18:                                               ; preds = %6
  %19 = landingpad { ptr, i32 }
          cleanup
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %21 = load i64, ptr %20, align 8, !range !354, !invariant.load !29
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %23 = load i64, ptr %22, align 8, !range !476, !invariant.load !29
  %24 = add i64 %23, -1
  %25 = icmp sgt i64 %24, -1
  tail call void @llvm.assume(i1 %25)
  %26 = icmp eq i64 %21, 0
  br i1 %26, label %28, label %27

27:                                               ; preds = %18
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %0, i64 noundef range(i64 1, -9223372036854775808) %21, i64 noundef range(i64 1, -9223372036854775807) %23) #37
  br label %28

28:                                               ; preds = %27, %18
  resume { ptr, i32 } %19

29:                                               ; preds = %17, %8
  ret void
}

; Function Attrs: noreturn nounwind nonlazybind uwtable
declare hidden void @_ZN12panic_unwind3imp5panic17exception_cleanup17h3e1ca1354f737ef9E(i32 range(i32 0, 10), ptr noundef) unnamed_addr #30

; Function Attrs: nonlazybind uwtable
declare noundef range(i32 0, 10) i32 @_Unwind_RaiseException(ptr noundef) unnamed_addr #20

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden void @_RNvCs1QLEhZ2QfLZ_7___rustc10rust_panic(ptr noundef nonnull align 1 %0, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(56) %1) unnamed_addr #19 {
  %3 = alloca [0 x i8], align 1
  %4 = alloca [16 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = alloca [4 x i8], align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %6)
  %7 = tail call noundef i32 @_RNvCs1QLEhZ2QfLZ_7___rustc18___rust_start_panic(ptr noundef nonnull align 1 %0, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(56) %1)
  store i32 %7, ptr %6, align 4
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4)
  store ptr %6, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u32$GT$3fmt17h9bc5442fdc664a66E", ptr %8, align 8
  store ptr @anon.babb0264256341099826814eb4f3574c.2, ptr %5, align 8
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 2, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %10, align 8
  %11 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %4, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 1, ptr %12, align 8
  %13 = call noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %3, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %5)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %5)
  call fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E"(ptr %13)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %4)
  call void @_ZN3std7process5abort17hea5afc4016cb7294E() #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E"(ptr %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = ptrtoint ptr %0 to i64
  %3 = and i64 %2, 3
  %4 = icmp eq i64 %3, 1
  br i1 %4, label %6, label %5, !prof !477

5:                                                ; preds = %37, %1
  ret void

6:                                                ; preds = %1
  %7 = getelementptr i8, ptr %0, i64 -1
  %8 = icmp ne ptr %7, null
  tail call void @llvm.assume(i1 %8)
  %9 = load ptr, ptr %7, align 8
  %10 = getelementptr i8, ptr %0, i64 7
  %11 = load ptr, ptr %10, align 8, !nonnull !29, !align !410, !noundef !29
  %12 = load ptr, ptr %11, align 8, !invariant.load !29
  %13 = icmp eq ptr %12, null
  br i1 %13, label %16, label %14

14:                                               ; preds = %6
  %15 = icmp ne ptr %9, null
  tail call void @llvm.assume(i1 %15)
  invoke void %12(ptr noundef nonnull %9)
          to label %16 unwind label %26

16:                                               ; preds = %14, %6
  %17 = icmp ne ptr %9, null
  tail call void @llvm.assume(i1 %17)
  %18 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %19 = load i64, ptr %18, align 8, !range !354, !invariant.load !29
  %20 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %21 = load i64, ptr %20, align 8, !range !476, !invariant.load !29
  %22 = add i64 %21, -1
  %23 = icmp sgt i64 %22, -1
  tail call void @llvm.assume(i1 %23)
  %24 = icmp eq i64 %19, 0
  br i1 %24, label %37, label %25

25:                                               ; preds = %16
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %9, i64 noundef range(i64 1, -9223372036854775808) %19, i64 noundef range(i64 1, -9223372036854775807) %21) #37
  br label %37

26:                                               ; preds = %14
  %27 = landingpad { ptr, i32 }
          cleanup
  %28 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %29 = load i64, ptr %28, align 8, !range !354, !invariant.load !29
  %30 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %31 = load i64, ptr %30, align 8, !range !476, !invariant.load !29
  %32 = add i64 %31, -1
  %33 = icmp sgt i64 %32, -1
  tail call void @llvm.assume(i1 %33)
  %34 = icmp eq i64 %29, 0
  br i1 %34, label %36, label %35

35:                                               ; preds = %26
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %9, i64 noundef range(i64 1, -9223372036854775808) %29, i64 noundef range(i64 1, -9223372036854775807) %31) #37
  br label %36

36:                                               ; preds = %35, %26
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %7, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %27

37:                                               ; preds = %25, %16
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %7, i64 noundef 24, i64 noundef 8) #37
  br label %5
}

; Function Attrs: noreturn nonlazybind uwtable
define hidden void @_RNvCs1QLEhZ2QfLZ_7___rustc17rust_begin_unwind(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %0) unnamed_addr #31 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = load ptr, ptr %3, align 8, !nonnull !29, !align !410, !noundef !29
  %5 = load ptr, ptr %0, align 8, !nonnull !29, !align !410, !noundef !29
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %2)
  store ptr %5, ptr %2, align 8
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr %4, ptr %6, align 8
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr %0, ptr %7, align 8
  call void @_ZN3std3sys9backtrace26__rust_end_short_backtrace17h31a73642178d42ccE(ptr noalias noundef nonnull align 8 captures(address) dereferenceable(24) %2) #53
  unreachable
}

; Function Attrs: noinline noreturn nonlazybind uwtable
define hidden void @_ZN3std3sys9backtrace26__rust_end_short_backtrace17h31a73642178d42ccE(ptr dead_on_return noalias noundef readonly align 8 captures(address) dereferenceable(24) %0) unnamed_addr #32 {
  tail call fastcc void @"_ZN3std9panicking13panic_handler28_$u7b$$u7b$closure$u7d$$u7d$17h0385f3ecdf1ec86eE"(ptr noalias noundef readonly align 8 captures(address) dereferenceable(24) %0) #53
  unreachable
}

; Function Attrs: inlinehint noreturn nonlazybind uwtable
define hidden fastcc void @"_ZN3std9panicking13panic_handler28_$u7b$$u7b$closure$u7d$$u7d$17h0385f3ecdf1ec86eE"(ptr dead_on_return noalias noundef nonnull readonly align 8 captures(address) dereferenceable(24) %0) unnamed_addr #33 personality ptr @rust_eh_personality {
  %2 = alloca [32 x i8], align 8
  %3 = alloca [16 x i8], align 8
  %4 = load ptr, ptr %0, align 8, !nonnull !29, !align !410, !noundef !29
  %5 = load ptr, ptr %4, align 8, !nonnull !29, !align !410, !noundef !29
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %7 = load i64, ptr %6, align 8, !noundef !29
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %9 = load i64, ptr %8, align 8, !noundef !29
  switch i64 %7, label %12 [
    i64 0, label %10
    i64 1, label %38
  ]

10:                                               ; preds = %1
  %11 = icmp eq i64 %9, 0
  br i1 %11, label %24, label %12

12:                                               ; preds = %38, %10, %1
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %2)
  %13 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store ptr %0, ptr %13, align 8
  store i64 -9223372036854775808, ptr %2, align 8
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %15 = load ptr, ptr %14, align 8, !nonnull !29, !align !410, !noundef !29
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %17 = load ptr, ptr %16, align 8, !nonnull !29, !align !410, !noundef !29
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %19 = load i8, ptr %18, align 8, !range !478, !noundef !29
  %20 = trunc nuw i8 %19 to i1
  %21 = getelementptr inbounds nuw i8, ptr %17, i64 17
  %22 = load i8, ptr %21, align 1, !range !478, !noundef !29
  %23 = trunc nuw i8 %22 to i1
  invoke void @_ZN3std9panicking15panic_with_hook17h5be4c5c4cf1661f8E(ptr noundef nonnull align 1 %2, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(56) @anon.babb0264256341099826814eb4f3574c.73, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %15, i1 noundef zeroext %20, i1 noundef zeroext %23) #53
          to label %54 unwind label %44

24:                                               ; preds = %40, %10
  %25 = phi i64 [ %43, %40 ], [ 0, %10 ]
  %26 = phi ptr [ %41, %40 ], [ inttoptr (i64 1 to ptr), %10 ]
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store ptr %26, ptr %3, align 8
  %27 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 %25, ptr %27, align 8
  %28 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %29 = load ptr, ptr %28, align 8, !nonnull !29, !align !410, !noundef !29
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %31 = load ptr, ptr %30, align 8, !nonnull !29, !align !410, !noundef !29
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 16
  %33 = load i8, ptr %32, align 8, !range !478, !noundef !29
  %34 = trunc nuw i8 %33 to i1
  %35 = getelementptr inbounds nuw i8, ptr %31, i64 17
  %36 = load i8, ptr %35, align 1, !range !478, !noundef !29
  %37 = trunc nuw i8 %36 to i1
  call void @_ZN3std9panicking15panic_with_hook17h5be4c5c4cf1661f8E(ptr noundef nonnull align 1 %3, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(56) @anon.babb0264256341099826814eb4f3574c.72, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %29, i1 noundef zeroext %34, i1 noundef zeroext %37) #53
  unreachable

38:                                               ; preds = %1
  %39 = icmp eq i64 %9, 0
  br i1 %39, label %40, label %12

40:                                               ; preds = %38
  %41 = load ptr, ptr %5, align 8, !nonnull !29, !align !411, !noundef !29
  %42 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %43 = load i64, ptr %42, align 8, !noundef !29
  br label %24

44:                                               ; preds = %12
  %45 = landingpad { ptr, i32 }
          cleanup
  %46 = load i64, ptr %2, align 8, !range !360, !alias.scope !479, !noundef !29
  %47 = icmp eq i64 %46, -9223372036854775808
  br i1 %47, label %58, label %48

48:                                               ; preds = %44
  invoke void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h17ee718c0c7a349dE"(ptr noalias noundef nonnull align 8 dereferenceable(32) %2)
          to label %53 unwind label %49

49:                                               ; preds = %48
  %50 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  invoke void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(32) %2)
          to label %57 unwind label %51

51:                                               ; preds = %49
  %52 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

53:                                               ; preds = %48
  invoke void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(32) %2)
          to label %58 unwind label %55

54:                                               ; preds = %12
  unreachable

55:                                               ; preds = %53
  %56 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  br label %57

57:                                               ; preds = %55, %49
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

58:                                               ; preds = %53, %44
  resume { ptr, i32 } %45
}

; Function Attrs: minsize noreturn nonlazybind optsize uwtable
define hidden void @_ZN3std9panicking15panic_with_hook17h5be4c5c4cf1661f8E(ptr noundef nonnull align 1 %0, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(56) %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %2, i1 noundef zeroext %3, i1 noundef zeroext %4) unnamed_addr #34 personality ptr @rust_eh_personality {
  %6 = alloca [0 x i8], align 1
  %7 = alloca [48 x i8], align 8
  %8 = alloca [32 x i8], align 8
  %9 = alloca [32 x i8], align 8
  %10 = alloca [24 x i8], align 8
  %11 = alloca [32 x i8], align 8
  %12 = alloca [48 x i8], align 8
  %13 = alloca [32 x i8], align 8
  %14 = alloca [48 x i8], align 8
  %15 = alloca [16 x i8], align 8
  %16 = alloca [8 x i8], align 8
  %17 = alloca [16 x i8], align 8
  store ptr %0, ptr %17, align 8
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store ptr %1, ptr %18, align 8
  store ptr %2, ptr %16, align 8
  %19 = tail call noundef i8 @_ZN3std9panicking11panic_count8increase17h294bbb53e68a902aE(i1 noundef zeroext true)
  %20 = icmp eq i8 %19, 2
  br i1 %20, label %23, label %21, !prof !361

21:                                               ; preds = %5
  %22 = trunc nuw i8 %19 to i1
  br i1 %22, label %41, label %59

23:                                               ; preds = %5
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %10)
  %24 = load atomic i32, ptr @_ZN3std9panicking4HOOK17hf5ae89eefcc7069aE monotonic, align 4
  %25 = icmp ult i32 %24, 1073741822
  br i1 %25, label %26, label %30, !prof !484

26:                                               ; preds = %23
  %27 = add nuw nsw i32 %24, 1
  %28 = cmpxchg weak ptr @_ZN3std9panicking4HOOK17hf5ae89eefcc7069aE, i32 %24, i32 %27 acquire monotonic, align 4
  %29 = extractvalue { i32, i1 } %28, 1
  br i1 %29, label %31, label %30, !prof !361

30:                                               ; preds = %26, %23
  tail call void @_ZN3std3sys4sync6rwlock5futex6RwLock14read_contended17h32383009711a8da9E(ptr noundef nonnull align 4 @_ZN3std9panicking4HOOK17hf5ae89eefcc7069aE)
  br label %31

31:                                               ; preds = %30, %26
  %32 = load atomic i8, ptr getelementptr inbounds nuw (i8, ptr @_ZN3std9panicking4HOOK17hf5ae89eefcc7069aE, i64 8) monotonic, align 1, !noalias !485
  %33 = icmp ne i8 %32, 0
  call void @_ZN3std4sync6poison10map_result17h2d2b88de1fc79d43E(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %10, i1 noundef zeroext %33, ptr noundef nonnull align 8 @_ZN3std9panicking4HOOK17hf5ae89eefcc7069aE)
  %34 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %35 = load ptr, ptr %34, align 8, !nonnull !29
  %36 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %37 = load ptr, ptr %36, align 8, !nonnull !29, !align !488
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %10)
  %38 = load ptr, ptr %35, align 8, !align !411, !noundef !29
  %39 = getelementptr inbounds nuw i8, ptr %35, i64 8
  %40 = icmp eq ptr %38, null
  br i1 %40, label %75, label %69

41:                                               ; preds = %21
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %15)
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %43 = load ptr, ptr %42, align 8, !invariant.load !29, !nonnull !29
  %44 = tail call { ptr, i64 } %43(ptr noundef nonnull align 1 %0)
  %45 = extractvalue { ptr, i64 } %44, 0
  %46 = icmp eq ptr %45, null
  %47 = extractvalue { ptr, i64 } %44, 1
  %48 = select i1 %46, ptr inttoptr (i64 1 to ptr), ptr %45
  %49 = select i1 %46, i64 0, i64 %47
  store ptr %48, ptr %15, align 8
  %50 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store i64 %49, ptr %50, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %14)
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %13)
  store ptr %16, ptr %13, align 8
  %51 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h05ab7ecd738f8002E", ptr %51, align 8
  %52 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store ptr %15, ptr %52, align 8
  %53 = getelementptr inbounds nuw i8, ptr %13, i64 24
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17hbbba634b9e74e25cE", ptr %53, align 8
  store ptr @anon.babb0264256341099826814eb4f3574c.82, ptr %14, align 8
  %54 = getelementptr inbounds nuw i8, ptr %14, i64 8
  store i64 3, ptr %54, align 8
  %55 = getelementptr inbounds nuw i8, ptr %14, i64 32
  store ptr null, ptr %55, align 8
  %56 = getelementptr inbounds nuw i8, ptr %14, i64 16
  store ptr %13, ptr %56, align 8
  %57 = getelementptr inbounds nuw i8, ptr %14, i64 24
  store i64 2, ptr %57, align 8
  %58 = call noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %6, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %14)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %14)
  call fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E"(ptr %58)
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %13)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %15)
  br label %68

59:                                               ; preds = %21
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %12)
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %11)
  store ptr %16, ptr %11, align 8
  %60 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h05ab7ecd738f8002E", ptr %60, align 8
  %61 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store ptr %17, ptr %61, align 8
  %62 = getelementptr inbounds nuw i8, ptr %11, i64 24
  store ptr @"_ZN52_$LT$$RF$mut$u20$T$u20$as$u20$core..fmt..Display$GT$3fmt17h343b7c4136dd9f77E", ptr %62, align 8
  store ptr @anon.babb0264256341099826814eb4f3574c.79, ptr %12, align 8
  %63 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store i64 3, ptr %63, align 8
  %64 = getelementptr inbounds nuw i8, ptr %12, i64 32
  store ptr null, ptr %64, align 8
  %65 = getelementptr inbounds nuw i8, ptr %12, i64 16
  store ptr %11, ptr %65, align 8
  %66 = getelementptr inbounds nuw i8, ptr %12, i64 24
  store i64 2, ptr %66, align 8
  %67 = call noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %6, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %12)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %12)
  call fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E"(ptr %67)
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %11)
  br label %68

68:                                               ; preds = %59, %41
  call void @_ZN3std7process5abort17hea5afc4016cb7294E() #53
  unreachable

69:                                               ; preds = %31
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %8)
  %70 = load ptr, ptr %17, align 8, !nonnull !29, !align !411, !noundef !29
  %71 = load ptr, ptr %18, align 8, !nonnull !29, !align !410, !noundef !29
  %72 = getelementptr inbounds nuw i8, ptr %71, i64 40
  %73 = load ptr, ptr %72, align 8, !invariant.load !29, !nonnull !29
  %74 = invoke { ptr, ptr } %73(ptr noundef nonnull align 1 %70)
          to label %97 unwind label %81

75:                                               ; preds = %31
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %9)
  %76 = load ptr, ptr %17, align 8, !nonnull !29, !align !411, !noundef !29
  %77 = load ptr, ptr %18, align 8, !nonnull !29, !align !410, !noundef !29
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 40
  %79 = load ptr, ptr %78, align 8, !invariant.load !29, !nonnull !29
  %80 = invoke { ptr, ptr } %79(ptr noundef nonnull align 1 %76)
          to label %83 unwind label %81

81:                                               ; preds = %97, %83, %75, %69
  %82 = landingpad { ptr, i32 }
          cleanup
  invoke fastcc void @"_ZN4core3ptr91drop_in_place$LT$std..sync..poison..rwlock..RwLockReadGuard$LT$std..panicking..Hook$GT$$GT$17h32227c1bc498d326E"(ptr nonnull %37) #56
          to label %121 unwind label %119

83:                                               ; preds = %75
  %84 = extractvalue { ptr, ptr } %80, 0
  %85 = extractvalue { ptr, ptr } %80, 1
  store ptr %84, ptr %9, align 8
  %86 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store ptr %85, ptr %86, align 8
  %87 = load ptr, ptr %16, align 8, !nonnull !29, !align !410, !noundef !29
  %88 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store ptr %87, ptr %88, align 8
  %89 = getelementptr inbounds nuw i8, ptr %9, i64 24
  %90 = zext i1 %3 to i8
  store i8 %90, ptr %89, align 8
  %91 = getelementptr inbounds nuw i8, ptr %9, i64 25
  %92 = zext i1 %4 to i8
  store i8 %92, ptr %91, align 1
  invoke void @_ZN3std9panicking12default_hook17h9dffdeeaffc4a531E(ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(32) %9)
          to label %93 unwind label %81

93:                                               ; preds = %83
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %9)
  br label %94

94:                                               ; preds = %111, %93
  %95 = phi ptr [ %71, %111 ], [ %77, %93 ]
  %96 = phi ptr [ %70, %111 ], [ %76, %93 ]
  call fastcc void @"_ZN4core3ptr91drop_in_place$LT$std..sync..poison..rwlock..RwLockReadGuard$LT$std..panicking..Hook$GT$$GT$17h32227c1bc498d326E"(ptr nonnull %37)
  call void @_ZN3std9panicking11panic_count19finished_panic_hook17h7ab0510843eb6188E()
  br i1 %3, label %118, label %112, !prof !361

97:                                               ; preds = %69
  %98 = extractvalue { ptr, ptr } %74, 0
  %99 = extractvalue { ptr, ptr } %74, 1
  store ptr %98, ptr %8, align 8
  %100 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store ptr %99, ptr %100, align 8
  %101 = load ptr, ptr %16, align 8, !nonnull !29, !align !410, !noundef !29
  %102 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store ptr %101, ptr %102, align 8
  %103 = getelementptr inbounds nuw i8, ptr %8, i64 24
  %104 = zext i1 %3 to i8
  store i8 %104, ptr %103, align 8
  %105 = getelementptr inbounds nuw i8, ptr %8, i64 25
  %106 = zext i1 %4 to i8
  store i8 %106, ptr %105, align 1
  %107 = load ptr, ptr %35, align 8, !nonnull !29, !noundef !29
  %108 = load ptr, ptr %39, align 8, !nonnull !29, !align !410, !noundef !29
  %109 = getelementptr inbounds nuw i8, ptr %108, i64 40
  %110 = load ptr, ptr %109, align 8, !invariant.load !29, !nonnull !29
  invoke void %110(ptr noundef nonnull align 1 %107, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(32) %8)
          to label %111 unwind label %81

111:                                              ; preds = %97
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %8)
  br label %94

112:                                              ; preds = %94
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %7)
  store ptr @anon.babb0264256341099826814eb4f3574c.84, ptr %7, align 8
  %113 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i64 1, ptr %113, align 8
  %114 = getelementptr inbounds nuw i8, ptr %7, i64 32
  store ptr null, ptr %114, align 8
  %115 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %115, align 8
  %116 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store i64 0, ptr %116, align 8
  %117 = call noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %6, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %7)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %7)
  call fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E"(ptr %117)
  call void @_ZN3std7process5abort17hea5afc4016cb7294E() #53
  unreachable

118:                                              ; preds = %94
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc10rust_panic(ptr noundef nonnull align 1 %96, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(56) %95) #53
  unreachable

119:                                              ; preds = %81
  %120 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

121:                                              ; preds = %81
  resume { ptr, i32 } %82
}

; Function Attrs: cold nonlazybind uwtable
define hidden void @_ZN3std3sys4sync6rwlock5futex6RwLock14read_contended17h32383009711a8da9E(ptr noundef nonnull align 4 %0) unnamed_addr #16 personality ptr @rust_eh_personality {
  %2 = alloca [48 x i8], align 8
  %3 = load atomic i32, ptr %0 monotonic, align 4
  %4 = icmp eq i32 %3, 1073741823
  br i1 %4, label %5, label %12

5:                                                ; preds = %5, %1
  %6 = phi i32 [ %7, %5 ], [ 100, %1 ]
  tail call void @llvm.x86.sse2.pause() #37
  %7 = add nsw i32 %6, -1
  %8 = load atomic i32, ptr %0 monotonic, align 4
  %9 = icmp ne i32 %8, 1073741823
  %10 = icmp eq i32 %7, 0
  %11 = select i1 %9, i1 true, i1 %10
  br i1 %11, label %12, label %5

12:                                               ; preds = %5, %1
  %13 = phi i32 [ %3, %1 ], [ %8, %5 ]
  br label %14

14:                                               ; preds = %85, %12
  %15 = phi i32 [ %86, %85 ], [ %13, %12 ]
  %16 = phi i1 [ true, %85 ], [ false, %12 ]
  br i1 %16, label %17, label %44

17:                                               ; preds = %41, %14
  %18 = phi i32 [ %43, %41 ], [ %15, %14 ]
  %19 = and i32 %18, 1073741823
  %20 = and i32 %18, 1073741824
  %21 = icmp ne i32 %20, 0
  %22 = add nsw i32 %19, -1073741822
  %23 = icmp ult i32 %22, -1073741821
  %24 = or i1 %21, %23
  br i1 %24, label %25, label %37

25:                                               ; preds = %17
  %26 = icmp samesign ult i32 %19, 1073741822
  %27 = icmp eq i32 %20, 0
  %28 = icmp ult i32 %18, 1073741824
  %29 = and i1 %26, %28
  br i1 %29, label %37, label %30

30:                                               ; preds = %25
  %31 = icmp eq i32 %19, 1073741822
  br i1 %31, label %63, label %32, !prof !353

32:                                               ; preds = %30
  br i1 %27, label %33, label %72

33:                                               ; preds = %32
  %34 = or disjoint i32 %18, 1073741824
  %35 = cmpxchg ptr %0, i32 %18, i32 %34 monotonic monotonic, align 4
  %36 = extractvalue { i32, i1 } %35, 1
  br i1 %36, label %72, label %41

37:                                               ; preds = %25, %17
  %38 = add nuw i32 %18, 1
  %39 = cmpxchg weak ptr %0, i32 %18, i32 %38 acquire monotonic, align 4
  %40 = extractvalue { i32, i1 } %39, 1
  br i1 %40, label %61, label %41

41:                                               ; preds = %37, %33
  %42 = phi { i32, i1 } [ %39, %37 ], [ %35, %33 ]
  %43 = extractvalue { i32, i1 } %42, 0
  br label %17

44:                                               ; preds = %56, %14
  %45 = phi i32 [ %58, %56 ], [ %15, %14 ]
  %46 = and i32 %45, 1073741823
  %47 = icmp samesign ult i32 %46, 1073741822
  %48 = and i32 %45, 1073741824
  %49 = icmp eq i32 %48, 0
  %50 = icmp ult i32 %45, 1073741824
  %51 = and i1 %47, %50
  br i1 %51, label %52, label %59

52:                                               ; preds = %44
  %53 = add nuw nsw i32 %45, 1
  %54 = cmpxchg weak ptr %0, i32 %45, i32 %53 acquire monotonic, align 4
  %55 = extractvalue { i32, i1 } %54, 1
  br i1 %55, label %61, label %56

56:                                               ; preds = %68, %52
  %57 = phi { i32, i1 } [ %54, %52 ], [ %70, %68 ]
  %58 = extractvalue { i32, i1 } %57, 0
  br label %44

59:                                               ; preds = %44
  %60 = icmp eq i32 %46, 1073741822
  br i1 %60, label %63, label %62, !prof !353

61:                                               ; preds = %52, %37
  ret void

62:                                               ; preds = %59
  br i1 %49, label %68, label %72

63:                                               ; preds = %59, %30
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %2)
  store ptr @anon.babb0264256341099826814eb4f3574c.51, ptr %2, align 8
  %64 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i64 1, ptr %64, align 8
  %65 = getelementptr inbounds nuw i8, ptr %2, i64 32
  store ptr null, ptr %65, align 8
  %66 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %66, align 8
  %67 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store i64 0, ptr %67, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %2, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.babb0264256341099826814eb4f3574c.53) #53
  unreachable

68:                                               ; preds = %62
  %69 = or disjoint i32 %45, 1073741824
  %70 = cmpxchg ptr %0, i32 %45, i32 %69 monotonic monotonic, align 4
  %71 = extractvalue { i32, i1 } %70, 1
  br i1 %71, label %72, label %56

72:                                               ; preds = %68, %62, %33, %32
  %73 = phi i32 [ %18, %33 ], [ %18, %32 ], [ %45, %68 ], [ %45, %62 ]
  %74 = or i32 %73, 1073741824
  %75 = tail call noundef zeroext i1 @_ZN3std3sys3pal4unix5futex10futex_wait17h26dd62589d2120e7E(ptr noundef nonnull align 4 %0, i32 noundef %74, i64 undef, i32 noundef 1000000000)
  %76 = load atomic i32, ptr %0 monotonic, align 4
  %77 = icmp eq i32 %76, 1073741823
  br i1 %77, label %78, label %85

78:                                               ; preds = %78, %72
  %79 = phi i32 [ %80, %78 ], [ 100, %72 ]
  tail call void @llvm.x86.sse2.pause() #37
  %80 = add nsw i32 %79, -1
  %81 = load atomic i32, ptr %0 monotonic, align 4
  %82 = icmp ne i32 %81, 1073741823
  %83 = icmp eq i32 %80, 0
  %84 = select i1 %82, i1 true, i1 %83
  br i1 %84, label %85, label %78

85:                                               ; preds = %78, %72
  %86 = phi i32 [ %76, %72 ], [ %81, %78 ]
  br label %14
}

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17hbbba634b9e74e25cE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(16), ptr noalias noundef align 8 dereferenceable(24)) unnamed_addr #20

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr91drop_in_place$LT$std..sync..poison..rwlock..RwLockReadGuard$LT$std..panicking..Hook$GT$$GT$17h32227c1bc498d326E"(ptr %0) unnamed_addr #20 {
  %2 = icmp ne ptr %0, null
  tail call void @llvm.assume(i1 %2)
  %3 = atomicrmw sub ptr %0, i32 1 release, align 4
  %4 = add i32 %3, -1
  %5 = and i32 %4, -1073741825
  %6 = icmp eq i32 %5, -2147483648
  br i1 %6, label %7, label %8, !prof !489

7:                                                ; preds = %1
  tail call void @_ZN3std3sys4sync6rwlock5futex6RwLock22wake_writer_or_readers17h6e8edbb853018a1cE(ptr noundef nonnull align 4 %0, i32 noundef %4)
  br label %8

8:                                                ; preds = %7, %1
  ret void
}

; Function Attrs: minsize nonlazybind optsize uwtable
define hidden void @_ZN3std9panicking12default_hook17h9dffdeeaffc4a531E(ptr noalias noundef readonly align 8 captures(none) dereferenceable(32) %0) unnamed_addr #35 personality ptr @rust_eh_personality {
  %2 = alloca [0 x i8], align 1
  %3 = alloca [16 x i8], align 8
  %4 = alloca [24 x i8], align 8
  %5 = alloca [8 x i8], align 8
  %6 = alloca [24 x i8], align 8
  %7 = alloca [16 x i8], align 8
  %8 = alloca [8 x i8], align 8
  %9 = alloca [1 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %9)
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 25
  %11 = load i8, ptr %10, align 1, !range !478, !noundef !29
  %12 = trunc nuw i8 %11 to i1
  br i1 %12, label %18, label %13

13:                                               ; preds = %1
  %14 = tail call noundef i64 @_ZN3std9panicking11panic_count9get_count17h77020e9322062869E()
  %15 = icmp ugt i64 %14, 1
  br i1 %15, label %18, label %16

16:                                               ; preds = %13
  %17 = tail call noundef i8 @_ZN3std5panic19get_backtrace_style17h869e43e10d5d55e1E()
  br label %18

18:                                               ; preds = %16, %13, %1
  %19 = phi i8 [ %17, %16 ], [ 3, %1 ], [ 1, %13 ]
  store i8 %19, ptr %9, align 1
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %8)
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %21 = load ptr, ptr %20, align 8, !nonnull !29, !align !410, !noundef !29
  store ptr %21, ptr %8, align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %7)
  %22 = load ptr, ptr %0, align 8, !nonnull !29, !align !411, !noundef !29
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %24 = load ptr, ptr %23, align 8, !nonnull !29, !align !410, !noundef !29
  %25 = tail call { ptr, i64 } @_ZN3std9panicking14payload_as_str17he5938f4fa5416c87E(ptr noundef nonnull align 1 %22, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(32) %24)
  %26 = extractvalue { ptr, i64 } %25, 0
  %27 = extractvalue { ptr, i64 } %25, 1
  store ptr %26, ptr %7, align 8
  %28 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i64 %27, ptr %28, align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %6)
  store ptr %8, ptr %6, align 8
  %29 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr %7, ptr %29, align 8
  %30 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store ptr %9, ptr %30, align 8
  %31 = call { i64, ptr } @_ZN3std2io5stdio22try_set_output_capture17h69ba45833ddbbce1E(ptr noundef null)
  %32 = extractvalue { i64, ptr } %31, 0
  %33 = extractvalue { i64, ptr } %31, 1
  %34 = trunc nuw i64 %32 to i1
  %35 = icmp eq ptr %33, null
  %36 = select i1 %34, i1 true, i1 %35
  br i1 %36, label %72, label %37

37:                                               ; preds = %18
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %5)
  store ptr %33, ptr %5, align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4)
  %38 = getelementptr inbounds nuw i8, ptr %33, i64 16
  invoke void @"_ZN3std4sync6poison5mutex14Mutex$LT$T$GT$4lock17h7d328904cb45fd4bE"(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %4, ptr noundef nonnull align 8 %38)
          to label %41 unwind label %39

39:                                               ; preds = %49, %37
  %40 = landingpad { ptr, i32 }
          cleanup
  br label %67

41:                                               ; preds = %37
  %42 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %43 = load ptr, ptr %42, align 8, !nonnull !29, !align !410
  %44 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %45 = load i8, ptr %44, align 8, !range !478
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4)
  %46 = getelementptr inbounds nuw i8, ptr %43, i64 8
  invoke fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$17h34b910bef98d2d0fE"(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %6, ptr noundef nonnull align 1 %46, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(80) @anon.babb0264256341099826814eb4f3574c.67)
          to label %49 unwind label %47

47:                                               ; preds = %41
  %48 = landingpad { ptr, i32 }
          cleanup
  invoke fastcc void @"_ZN4core3ptr90drop_in_place$LT$std..sync..poison..mutex..MutexGuard$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$17h02a8e7c16cb69f8aE"(ptr nonnull %43, i8 %45) #56
          to label %67 unwind label %64

49:                                               ; preds = %41
  invoke fastcc void @"_ZN4core3ptr90drop_in_place$LT$std..sync..poison..mutex..MutexGuard$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$17h02a8e7c16cb69f8aE"(ptr nonnull %43, i8 %45)
          to label %50 unwind label %39

50:                                               ; preds = %49
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  %51 = call { i64, ptr } @_ZN3std2io5stdio22try_set_output_capture17h69ba45833ddbbce1E(ptr noundef nonnull %33)
  %52 = extractvalue { i64, ptr } %51, 0
  %53 = trunc nuw i64 %52 to i1
  br i1 %53, label %62, label %54

54:                                               ; preds = %50
  %55 = extractvalue { i64, ptr } %51, 1
  %56 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %55, ptr %56, align 8
  store i64 1, ptr %3, align 8
  %57 = icmp eq ptr %55, null
  br i1 %57, label %62, label %58

58:                                               ; preds = %54
  %59 = atomicrmw sub ptr %55, i64 1 release, align 8, !noalias !490
  %60 = icmp eq i64 %59, 1
  br i1 %60, label %61, label %62

61:                                               ; preds = %58
  fence acquire
  call void @"_ZN5alloc4sync16Arc$LT$T$C$A$GT$9drop_slow17he9c118ac83d419dfE"(ptr noalias noundef nonnull align 8 dereferenceable(8) %56)
  br label %62

62:                                               ; preds = %61, %58, %54, %50
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %5)
  br label %63

63:                                               ; preds = %72, %62
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %6)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %7)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8)
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %9)
  ret void

64:                                               ; preds = %71, %47
  %65 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

66:                                               ; preds = %71, %67
  resume { ptr, i32 } %68

67:                                               ; preds = %47, %39
  %68 = phi { ptr, i32 } [ %40, %39 ], [ %48, %47 ]
  %69 = atomicrmw sub ptr %33, i64 1 release, align 8, !noalias !499
  %70 = icmp eq i64 %69, 1
  br i1 %70, label %71, label %66

71:                                               ; preds = %67
  fence acquire
  invoke void @"_ZN5alloc4sync16Arc$LT$T$C$A$GT$9drop_slow17he9c118ac83d419dfE"(ptr noalias noundef nonnull align 8 dereferenceable(8) %5)
          to label %66 unwind label %64

72:                                               ; preds = %18
  call fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$17h34b910bef98d2d0fE"(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %6, ptr noundef nonnull align 1 %2, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(80) @anon.babb0264256341099826814eb4f3574c.68)
  br label %63
}

; Function Attrs: nonlazybind uwtable
define hidden { ptr, i64 } @_ZN3std9panicking14payload_as_str17he5938f4fa5416c87E(ptr noundef nonnull align 1 %0, ptr noalias noundef readonly align 8 captures(none) dereferenceable(32) %1) unnamed_addr #20 {
  %3 = alloca [16 x i8], align 8
  %4 = alloca [16 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4)
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %6 = load ptr, ptr %5, align 8, !invariant.load !29, !nonnull !29
  call void %6(ptr noalias noundef nonnull sret([16 x i8]) align 8 captures(address) dereferenceable(16) %4, ptr noundef nonnull align 1 %0)
  %7 = load i128, ptr %4, align 8, !noundef !29
  %8 = icmp eq i128 %7, -93652901832424836513689306266955195027
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %4)
  br i1 %8, label %12, label %9

9:                                                ; preds = %2
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  call void %6(ptr noalias noundef nonnull sret([16 x i8]) align 8 captures(address) dereferenceable(16) %3, ptr noundef nonnull align 1 %0)
  %10 = load i128, ptr %3, align 8, !noundef !29
  %11 = icmp eq i128 %10, 147670966674170015111608506034036716200
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  br i1 %11, label %13, label %21

12:                                               ; preds = %2
  br label %15

13:                                               ; preds = %9
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %15

15:                                               ; preds = %13, %12
  %16 = phi i64 [ 8, %12 ], [ 16, %13 ]
  %17 = phi ptr [ %0, %12 ], [ %14, %13 ]
  %18 = load ptr, ptr %17, align 8, !nonnull !29, !noundef !29
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 %16
  %20 = load i64, ptr %19, align 8, !noundef !29
  br label %21

21:                                               ; preds = %15, %9
  %22 = phi i64 [ 12, %9 ], [ %20, %15 ]
  %23 = phi ptr [ @anon.babb0264256341099826814eb4f3574c.75, %9 ], [ %18, %15 ]
  %24 = insertvalue { ptr, i64 } poison, ptr %23, 0
  %25 = insertvalue { ptr, i64 } %24, i64 %22, 1
  ret { ptr, i64 } %25
}

; Function Attrs: nonlazybind uwtable
define hidden void @"_ZN3std4sync6poison5mutex14Mutex$LT$T$GT$4lock17h7d328904cb45fd4bE"(ptr dead_on_unwind noalias noundef writable sret([24 x i8]) align 8 captures(address) dereferenceable(24) %0, ptr noundef nonnull align 8 %1) unnamed_addr #20 {
  %3 = cmpxchg ptr %1, i32 0, i32 1 acquire monotonic, align 4
  %4 = extractvalue { i32, i1 } %3, 1
  br i1 %4, label %6, label %5, !prof !361

5:                                                ; preds = %2
  tail call void @_ZN3std3sys4sync5mutex5futex5Mutex14lock_contended17hd6d171911655b783E(ptr noundef nonnull align 4 %1)
  br label %6

6:                                                ; preds = %5, %2
  %7 = load atomic i64, ptr @_ZN3std9panicking11panic_count18GLOBAL_PANIC_COUNT17hae1b51fffade13abE monotonic, align 8
  %8 = and i64 %7, 9223372036854775807
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %14, label %10, !prof !361

10:                                               ; preds = %6
  %11 = tail call noundef zeroext i1 @_ZN3std9panicking11panic_count17is_zero_slow_path17h0941546548007a1dE()
  %12 = xor i1 %11, true
  %13 = zext i1 %12 to i8
  br label %14

14:                                               ; preds = %10, %6
  %15 = phi i8 [ %13, %10 ], [ 0, %6 ]
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %17 = load atomic i8, ptr %16 monotonic, align 1
  %18 = icmp ne i8 %17, 0
  tail call void @_ZN3std4sync6poison10map_result17h6348d1c69c45512aE(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %0, i1 noundef zeroext %18, i8 noundef %15, ptr noundef nonnull align 8 %1)
  ret void
}

; Function Attrs: inlinehint minsize nonlazybind optsize uwtable
define hidden fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$17h34b910bef98d2d0fE"(ptr noalias noundef nonnull readonly align 8 captures(none) dereferenceable(24) %0, ptr noundef nonnull align 1 %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(80) %2) unnamed_addr #36 personality ptr @rust_eh_personality {
  %4 = alloca [32 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = tail call { ptr, i1 } @_ZN3std3sys9backtrace4lock17hee307f8c9af6c2caE()
  %7 = extractvalue { ptr, i1 } %6, 0
  %8 = extractvalue { ptr, i1 } %6, 1
  %9 = zext i1 %8 to i8
  %10 = load ptr, ptr %0, align 8, !nonnull !29, !align !410, !noundef !29
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %12 = load ptr, ptr %11, align 8, !nonnull !29, !align !410, !noundef !29
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %4)
  store ptr %10, ptr %4, align 8
  %13 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %12, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %1, ptr %14, align 8
  %15 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr %2, ptr %15, align 8
  invoke void @_ZN3std6thread7current16try_with_current17h16830d05b1d122d7E(ptr noalias noundef nonnull align 8 captures(address) dereferenceable(32) %4)
          to label %19 unwind label %16

16:                                               ; preds = %37, %35, %27, %3
  %17 = landingpad { ptr, i32 }
          cleanup
  %18 = icmp ne ptr %7, null
  call void @llvm.assume(i1 %18)
  invoke fastcc void @"_ZN4core3ptr55drop_in_place$LT$std..sys..backtrace..BacktraceLock$GT$17h514319a6ede2e129E"(ptr nonnull %7, i8 %9) #56
          to label %48 unwind label %46

19:                                               ; preds = %3
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %4)
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %21 = load ptr, ptr %20, align 8, !nonnull !29, !align !411, !noundef !29
  %22 = load i8, ptr %21, align 1, !range !504, !noundef !29
  switch i8 %22, label %25 [
    i8 3, label %23
    i8 0, label %26
    i8 1, label %27
    i8 2, label %32
  ]

23:                                               ; preds = %35, %32, %19
  %24 = icmp ne ptr %7, null
  call void @llvm.assume(i1 %24)
  call fastcc void @"_ZN4core3ptr55drop_in_place$LT$std..sys..backtrace..BacktraceLock$GT$17h514319a6ede2e129E"(ptr nonnull %7, i8 %9)
  ret void

25:                                               ; preds = %19
  unreachable

26:                                               ; preds = %19
  br label %27

27:                                               ; preds = %26, %19
  %28 = phi i1 [ false, %26 ], [ true, %19 ]
  %29 = getelementptr inbounds nuw i8, ptr %2, i64 72
  %30 = load ptr, ptr %29, align 8
  %31 = invoke fastcc noundef ptr @_ZN3std3sys9backtrace13BacktraceLock5print17h26597924daa51d83E(ptr noundef nonnull align 1 %1, ptr %30, i1 noundef zeroext %28)
          to label %35 unwind label %16

32:                                               ; preds = %19
  %33 = atomicrmw xchg ptr @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$11FIRST_PANIC17h1c9f85ba2097d41cE", i8 0 monotonic, align 1
  %34 = icmp eq i8 %33, 0
  br i1 %34, label %23, label %37

35:                                               ; preds = %45, %27
  %36 = phi ptr [ %44, %45 ], [ %31, %27 ]
  invoke fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E"(ptr %36)
          to label %23 unwind label %16

37:                                               ; preds = %32
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  store ptr @anon.babb0264256341099826814eb4f3574c.70, ptr %5, align 8
  %38 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 1, ptr %38, align 8
  %39 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %39, align 8
  %40 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %40, align 8
  %41 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 0, ptr %41, align 8
  %42 = getelementptr inbounds nuw i8, ptr %2, i64 72
  %43 = load ptr, ptr %42, align 8, !invariant.load !29, !nonnull !29
  %44 = invoke noundef ptr %43(ptr noundef nonnull align 1 %1, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %5)
          to label %45 unwind label %16

45:                                               ; preds = %37
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %5)
  br label %35

46:                                               ; preds = %16
  %47 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

48:                                               ; preds = %16
  resume { ptr, i32 } %17
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr90drop_in_place$LT$std..sync..poison..mutex..MutexGuard$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$17h02a8e7c16cb69f8aE"(ptr %0, i8 %1) unnamed_addr #20 {
  %3 = icmp ne ptr %0, null
  tail call void @llvm.assume(i1 %3)
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %5 = trunc nuw i8 %1 to i1
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = load atomic i64, ptr @_ZN3std9panicking11panic_count18GLOBAL_PANIC_COUNT17hae1b51fffade13abE monotonic, align 8
  %8 = and i64 %7, 9223372036854775807
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %13, label %10, !prof !361

10:                                               ; preds = %6
  %11 = tail call noundef zeroext i1 @_ZN3std9panicking11panic_count17is_zero_slow_path17h0941546548007a1dE()
  br i1 %11, label %13, label %12

12:                                               ; preds = %10
  store atomic i8 1, ptr %4 monotonic, align 1
  br label %13

13:                                               ; preds = %12, %10, %6, %2
  %14 = atomicrmw xchg ptr %0, i32 0 release, align 4
  %15 = icmp eq i32 %14, 2
  br i1 %15, label %16, label %17, !prof !353

16:                                               ; preds = %13
  tail call void @_ZN3std3sys4sync5mutex5futex5Mutex4wake17h98256de9aeb3cee5E(ptr noundef nonnull align 4 %0)
  br label %17

17:                                               ; preds = %16, %13
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden { ptr, i1 } @_ZN3std3sys9backtrace4lock17hee307f8c9af6c2caE() unnamed_addr #20 {
  %1 = alloca [24 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1)
  %2 = cmpxchg ptr @_ZN3std3sys9backtrace4lock4LOCK17hf34eb78a76761272E, i32 0, i32 1 acquire monotonic, align 4, !noalias !505
  %3 = extractvalue { i32, i1 } %2, 1
  br i1 %3, label %5, label %4, !prof !361

4:                                                ; preds = %0
  tail call void @_ZN3std3sys4sync5mutex5futex5Mutex14lock_contended17hd6d171911655b783E(ptr noundef nonnull align 4 @_ZN3std3sys9backtrace4lock4LOCK17hf34eb78a76761272E), !noalias !505
  br label %5

5:                                                ; preds = %4, %0
  %6 = load atomic i64, ptr @_ZN3std9panicking11panic_count18GLOBAL_PANIC_COUNT17hae1b51fffade13abE monotonic, align 8, !noalias !505
  %7 = and i64 %6, 9223372036854775807
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %13, label %9, !prof !361

9:                                                ; preds = %5
  %10 = tail call noundef zeroext i1 @_ZN3std9panicking11panic_count17is_zero_slow_path17h0941546548007a1dE(), !noalias !505
  %11 = xor i1 %10, true
  %12 = zext i1 %11 to i8
  br label %13

13:                                               ; preds = %9, %5
  %14 = phi i8 [ %12, %9 ], [ 0, %5 ]
  %15 = load atomic i8, ptr getelementptr inbounds nuw (i8, ptr @_ZN3std3sys9backtrace4lock4LOCK17hf34eb78a76761272E, i64 4) monotonic, align 1, !noalias !505
  %16 = icmp ne i8 %15, 0
  call void @_ZN3std4sync6poison10map_result17h50aec882be779b28E(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %1, i1 noundef zeroext %16, i8 noundef %14, ptr noundef nonnull align 4 @_ZN3std3sys9backtrace4lock4LOCK17hf34eb78a76761272E)
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %18 = load ptr, ptr %17, align 8, !nonnull !29, !align !488
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %20 = load i8, ptr %19, align 8, !range !478
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1)
  %21 = trunc nuw i8 %20 to i1
  %22 = insertvalue { ptr, i1 } poison, ptr %18, 0
  %23 = insertvalue { ptr, i1 } %22, i1 %21, 1
  ret { ptr, i1 } %23
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr55drop_in_place$LT$std..sys..backtrace..BacktraceLock$GT$17h514319a6ede2e129E"(ptr %0, i8 %1) unnamed_addr #20 {
  %3 = icmp ne ptr %0, null
  tail call void @llvm.assume(i1 %3)
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %5 = trunc nuw i8 %1 to i1
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = load atomic i64, ptr @_ZN3std9panicking11panic_count18GLOBAL_PANIC_COUNT17hae1b51fffade13abE monotonic, align 8
  %8 = and i64 %7, 9223372036854775807
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %13, label %10, !prof !361

10:                                               ; preds = %6
  %11 = tail call noundef zeroext i1 @_ZN3std9panicking11panic_count17is_zero_slow_path17h0941546548007a1dE()
  br i1 %11, label %13, label %12

12:                                               ; preds = %10
  store atomic i8 1, ptr %4 monotonic, align 1
  br label %13

13:                                               ; preds = %12, %10, %6, %2
  %14 = atomicrmw xchg ptr %0, i32 0 release, align 4
  %15 = icmp eq i32 %14, 2
  br i1 %15, label %16, label %17, !prof !353

16:                                               ; preds = %13
  tail call void @_ZN3std3sys4sync5mutex5futex5Mutex4wake17h98256de9aeb3cee5E(ptr noundef nonnull align 4 %0)
  br label %17

17:                                               ; preds = %16, %13
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc noundef ptr @_ZN3std3sys9backtrace13BacktraceLock5print17h26597924daa51d83E(ptr noundef nonnull align 1 %0, ptr readonly captures(none) %1, i1 noundef zeroext %2) unnamed_addr #20 {
  %4 = alloca [16 x i8], align 8
  %5 = alloca [1 x i8], align 1
  %6 = alloca [48 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %6)
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %5)
  %7 = zext i1 %2 to i8
  store i8 %7, ptr %5, align 1
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4)
  store ptr %5, ptr %4, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN98_$LT$std..sys..backtrace..BacktraceLock..print..DisplayBacktrace$u20$as$u20$core..fmt..Display$GT$3fmt17hb65127cd6f8382acE", ptr %8, align 8
  store ptr @anon.babb0264256341099826814eb4f3574c.60, ptr %6, align 8
  %9 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i64 1, ptr %9, align 8
  %10 = getelementptr inbounds nuw i8, ptr %6, i64 32
  store ptr null, ptr %10, align 8
  %11 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store ptr %4, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store i64 1, ptr %12, align 8
  %13 = call noundef ptr %1(ptr noundef nonnull align 1 %0, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %6)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %4)
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %5)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %6)
  ret ptr %13
}

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN98_$LT$std..sys..backtrace..BacktraceLock..print..DisplayBacktrace$u20$as$u20$core..fmt..Display$GT$3fmt17hb65127cd6f8382acE"(ptr noalias noundef readonly align 1 captures(none) dereferenceable(1), ptr noalias noundef align 8 dereferenceable(24)) unnamed_addr #20

; Function Attrs: cold nonlazybind uwtable
define hidden void @_ZN3std3sys4sync6rwlock5futex6RwLock22wake_writer_or_readers17h6e8edbb853018a1cE(ptr noundef nonnull align 4 %0, i32 noundef %1) unnamed_addr #16 {
  %3 = and i32 %1, 1073741823
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %7, !prof !361

5:                                                ; preds = %2
  %6 = icmp eq i32 %1, -2147483648
  br i1 %6, label %8, label %12

7:                                                ; preds = %2
  tail call void @_ZN4core9panicking5panic17h80104d8006620ef0E(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.babb0264256341099826814eb4f3574c.54, i64 noundef 36, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.babb0264256341099826814eb4f3574c.55) #53
  unreachable

8:                                                ; preds = %5
  %9 = cmpxchg ptr %0, i32 -2147483648, i32 0 monotonic monotonic, align 4
  %10 = extractvalue { i32, i1 } %9, 1
  %11 = extractvalue { i32, i1 } %9, 0
  br i1 %10, label %14, label %12

12:                                               ; preds = %8, %5
  %13 = phi i32 [ %1, %5 ], [ %11, %8 ]
  switch i32 %13, label %18 [
    i32 -1073741824, label %19
    i32 1073741824, label %26
  ]

14:                                               ; preds = %8
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %16 = atomicrmw add ptr %15, i32 1 release, align 4
  %17 = tail call noundef zeroext i1 @_ZN3std3sys3pal4unix5futex10futex_wake17hc3e243608aa5a740E(ptr noundef nonnull align 4 %15)
  br label %18

18:                                               ; preds = %29, %26, %22, %19, %14, %12
  ret void

19:                                               ; preds = %12
  %20 = cmpxchg ptr %0, i32 -1073741824, i32 1073741824 monotonic monotonic, align 4
  %21 = extractvalue { i32, i1 } %20, 1
  br i1 %21, label %22, label %18

22:                                               ; preds = %19
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %24 = atomicrmw add ptr %23, i32 1 release, align 4
  %25 = tail call noundef zeroext i1 @_ZN3std3sys3pal4unix5futex10futex_wake17hc3e243608aa5a740E(ptr noundef nonnull align 4 %23)
  br i1 %25, label %18, label %26

26:                                               ; preds = %22, %12
  %27 = cmpxchg ptr %0, i32 1073741824, i32 0 monotonic monotonic, align 4
  %28 = extractvalue { i32, i1 } %27, 1
  br i1 %28, label %29, label %18

29:                                               ; preds = %26
  tail call void @_ZN3std3sys3pal4unix5futex14futex_wake_all17h0970ed6199f67cb9E(ptr noundef nonnull align 4 %0)
  br label %18
}

; Function Attrs: nounwind
declare void @llvm.x86.sse2.pause() unnamed_addr #37

; Function Attrs: nofree nosync nounwind nonlazybind memory(none) uwtable
declare noundef ptr @__errno_location() unnamed_addr #38

; Function Attrs: mustprogress nounwind nonlazybind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable
declare void @free(ptr allocptr noundef captures(none)) unnamed_addr #39

; Function Attrs: nounwind nonlazybind uwtable
declare noundef i64 @syscall(i64 noundef, ...) unnamed_addr #22

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #3

; Function Attrs: cold nofree noreturn nounwind nonlazybind uwtable
define hidden void @_ZN3std3sys3pal4unix14abort_internal17hb7e3b71be536cdfcE() unnamed_addr #40 {
  tail call void @abort() #54
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind nonlazybind uwtable
declare void @abort() unnamed_addr #40

; Function Attrs: nofree nosync nounwind nonlazybind memory(read, inaccessiblemem: none) uwtable
define hidden { i64, ptr } @_ZN3std3sys3pal4unix3cvt17h2967415d3ce1c61bE(i64 noundef %0) unnamed_addr #41 personality ptr @rust_eh_personality {
  %2 = icmp eq i64 %0, -1
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = inttoptr i64 %0 to ptr
  br label %12

5:                                                ; preds = %1
  %6 = tail call noundef ptr @__errno_location() #37
  %7 = load i32, ptr %6, align 4, !noundef !29
  %8 = sext i32 %7 to i64
  %9 = shl nsw i64 %8, 32
  %10 = getelementptr i8, ptr null, i64 %9
  %11 = getelementptr i8, ptr %10, i64 2
  br label %12

12:                                               ; preds = %5, %3
  %13 = phi ptr [ %11, %5 ], [ %4, %3 ]
  %14 = phi i64 [ 1, %5 ], [ 0, %3 ]
  %15 = insertvalue { i64, ptr } poison, i64 %14, 0
  %16 = insertvalue { i64, ptr } %15, ptr %13, 1
  ret { i64, ptr } %16
}

; Function Attrs: nofree nosync nounwind nonlazybind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define hidden void @_ZN3std3sys3pal4unix3cvt17h4ef7b5dd8aa3bbb4E(ptr dead_on_unwind noalias noundef writable writeonly sret([16 x i8]) align 8 captures(none) dereferenceable(16) initializes((0, 4)) %0, i32 noundef %1) unnamed_addr #42 personality ptr @rust_eh_personality {
  %3 = icmp eq i32 %1, -1
  br i1 %3, label %6, label %4

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 %1, ptr %5, align 4
  br label %14

6:                                                ; preds = %2
  %7 = tail call noundef ptr @__errno_location() #37
  %8 = load i32, ptr %7, align 4, !noundef !29
  %9 = sext i32 %8 to i64
  %10 = shl nsw i64 %9, 32
  %11 = getelementptr i8, ptr null, i64 %10
  %12 = getelementptr i8, ptr %11, i64 2
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %12, ptr %13, align 8
  br label %14

14:                                               ; preds = %6, %4
  %15 = phi i32 [ 0, %4 ], [ 1, %6 ]
  store i32 %15, ptr %0, align 8
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden noundef ptr @_ZN3std2io5Write9write_all17h60dffe48e453ccd8E(ptr noalias nonnull readnone align 1 captures(none) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2) unnamed_addr #20 personality ptr @rust_eh_personality {
  %4 = alloca [4 x i8], align 4
  %5 = icmp eq i64 %2, 0
  br i1 %5, label %15, label %6

6:                                                ; preds = %77, %3
  %7 = phi ptr [ %78, %77 ], [ %1, %3 ]
  %8 = phi i64 [ %79, %77 ], [ %2, %3 ]
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %4), !noalias !508
  %9 = call noundef i32 @"_ZN76_$LT$std..sys..fd..unix..FileDesc$u20$as$u20$std..os..fd..raw..FromRawFd$GT$11from_raw_fd17h3ef52e8d28c3d9ffE"(i32 noundef 2), !noalias !508
  store i32 %9, ptr %4, align 4, !noalias !508
  %10 = call { i64, ptr } @_ZN3std3sys2fd4unix8FileDesc5write17h8d5efdf88bb3ba3aE(ptr noalias noundef nonnull readonly align 4 captures(address, read_provenance) dereferenceable(4) %4, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %7, i64 noundef %8)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %4), !noalias !508
  %11 = extractvalue { i64, ptr } %10, 0
  %12 = extractvalue { i64, ptr } %10, 1
  %13 = ptrtoint ptr %12 to i64
  %14 = trunc nuw i64 %11 to i1
  br i1 %14, label %17, label %23

15:                                               ; preds = %77, %39, %34, %31, %23, %20, %3
  %16 = phi ptr [ null, %3 ], [ %12, %20 ], [ %12, %39 ], [ %12, %31 ], [ %12, %34 ], [ @anon.bc75b2dae71f64fa323d019d8562410b.3, %23 ], [ null, %77 ]
  ret ptr %16

17:                                               ; preds = %6
  %18 = and i64 %13, 3
  switch i64 %18, label %19 [
    i64 2, label %31
    i64 3, label %20
    i64 0, label %34
    i64 1, label %39
  ], !prof !511

19:                                               ; preds = %17
  unreachable

20:                                               ; preds = %17
  %21 = and i64 %13, -4294967296
  %22 = icmp eq i64 %21, 150323855360
  br i1 %22, label %77, label %15

23:                                               ; preds = %6
  %24 = icmp eq ptr %12, null
  br i1 %24, label %15, label %25

25:                                               ; preds = %23
  %26 = icmp ult i64 %8, %13
  br i1 %26, label %27, label %28, !prof !353

27:                                               ; preds = %25
  call void @_ZN4core5slice5index16slice_index_fail17hf7a05389aea37f33E(i64 noundef range(i64 1, 0) %13, i64 noundef %8, i64 noundef %8, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.bc75b2dae71f64fa323d019d8562410b.5) #53
  unreachable

28:                                               ; preds = %25
  %29 = sub nuw i64 %8, %13
  %30 = getelementptr inbounds nuw i8, ptr %7, i64 %13
  br label %77

31:                                               ; preds = %17
  %32 = and i64 %13, -4294967296
  %33 = icmp eq i64 %32, 17179869184
  br i1 %33, label %77, label %15

34:                                               ; preds = %17
  %35 = icmp ne ptr %12, null
  call void @llvm.assume(i1 %35)
  %36 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %37 = load i8, ptr %36, align 8, !range !512, !noundef !29
  %38 = icmp eq i8 %37, 35
  br i1 %38, label %77, label %15

39:                                               ; preds = %17
  %40 = getelementptr i8, ptr %12, i64 -1
  %41 = icmp ne ptr %40, null
  call void @llvm.assume(i1 %41)
  %42 = getelementptr i8, ptr %12, i64 15
  %43 = load i8, ptr %42, align 8, !range !512, !noundef !29
  %44 = icmp eq i8 %43, 35
  br i1 %44, label %45, label %15

45:                                               ; preds = %39
  %46 = getelementptr i8, ptr %12, i64 -1
  %47 = icmp ne ptr %46, null
  call void @llvm.assume(i1 %47)
  %48 = load ptr, ptr %46, align 8
  %49 = getelementptr i8, ptr %12, i64 7
  %50 = load ptr, ptr %49, align 8, !nonnull !29, !align !410, !noundef !29
  %51 = load ptr, ptr %50, align 8, !invariant.load !29
  %52 = icmp eq ptr %51, null
  br i1 %52, label %55, label %53

53:                                               ; preds = %45
  %54 = icmp ne ptr %48, null
  call void @llvm.assume(i1 %54)
  invoke void %51(ptr noundef nonnull %48)
          to label %55 unwind label %65

55:                                               ; preds = %53, %45
  %56 = icmp ne ptr %48, null
  call void @llvm.assume(i1 %56)
  %57 = getelementptr inbounds nuw i8, ptr %50, i64 8
  %58 = load i64, ptr %57, align 8, !range !354, !invariant.load !29
  %59 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %60 = load i64, ptr %59, align 8, !range !476, !invariant.load !29
  %61 = add i64 %60, -1
  %62 = icmp sgt i64 %61, -1
  call void @llvm.assume(i1 %62)
  %63 = icmp eq i64 %58, 0
  br i1 %63, label %76, label %64

64:                                               ; preds = %55
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %48, i64 noundef range(i64 1, 0) %58, i64 noundef range(i64 1, -9223372036854775807) %60) #37
  br label %76

65:                                               ; preds = %53
  %66 = landingpad { ptr, i32 }
          cleanup
  %67 = getelementptr inbounds nuw i8, ptr %50, i64 8
  %68 = load i64, ptr %67, align 8, !range !354, !invariant.load !29
  %69 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %70 = load i64, ptr %69, align 8, !range !476, !invariant.load !29
  %71 = add i64 %70, -1
  %72 = icmp sgt i64 %71, -1
  call void @llvm.assume(i1 %72)
  %73 = icmp eq i64 %68, 0
  br i1 %73, label %75, label %74

74:                                               ; preds = %65
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %48, i64 noundef range(i64 1, 0) %68, i64 noundef range(i64 1, -9223372036854775807) %70) #37
  br label %75

75:                                               ; preds = %74, %65
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %46, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %66

76:                                               ; preds = %64, %55
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %46, i64 noundef 24, i64 noundef 8) #37
  br label %77

77:                                               ; preds = %76, %34, %31, %28, %20
  %78 = phi ptr [ %30, %28 ], [ %7, %76 ], [ %7, %20 ], [ %7, %31 ], [ %7, %34 ]
  %79 = phi i64 [ %29, %28 ], [ %8, %76 ], [ %8, %20 ], [ %8, %31 ], [ %8, %34 ]
  %80 = icmp eq i64 %79, 0
  br i1 %80, label %15, label %6
}

; Function Attrs: nonlazybind uwtable
define hidden noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %0, ptr dead_on_return noalias noundef align 8 captures(address) dereferenceable(48) %1) unnamed_addr #20 {
  %3 = load ptr, ptr %1, align 8, !nonnull !29, !align !410, !noundef !29
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !noundef !29
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %7 = load i64, ptr %6, align 8, !noundef !29
  switch i64 %5, label %10 [
    i64 0, label %8
    i64 1, label %17
  ]

8:                                                ; preds = %2
  %9 = icmp eq i64 %7, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %17, %8, %2
  br label %11

11:                                               ; preds = %19, %10, %8
  %12 = phi i64 [ undef, %10 ], [ %22, %19 ], [ 0, %8 ]
  %13 = phi ptr [ null, %10 ], [ %20, %19 ], [ inttoptr (i64 1 to ptr), %8 ]
  %14 = icmp ne ptr %13, null
  %15 = tail call i1 @llvm.is.constant.i1(i1 %14)
  %16 = and i1 %15, %14
  br i1 %16, label %25, label %23

17:                                               ; preds = %2
  %18 = icmp eq i64 %7, 0
  br i1 %18, label %19, label %10

19:                                               ; preds = %17
  %20 = load ptr, ptr %3, align 8, !nonnull !29, !align !411, !noundef !29
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %22 = load i64, ptr %21, align 8, !noundef !29
  br label %11

23:                                               ; preds = %11
  %24 = tail call noundef ptr @_ZN3std2io17default_write_fmt17hb29cce0e5403eb27E(ptr noalias noundef nonnull align 1 %0, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %1)
  br label %27

25:                                               ; preds = %11
  %26 = tail call noundef ptr @_ZN3std2io5Write9write_all17h60dffe48e453ccd8E(ptr noalias nonnull align 1 poison, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %13, i64 noundef %12)
  br label %27

27:                                               ; preds = %25, %23
  %28 = phi ptr [ %26, %25 ], [ %24, %23 ]
  ret ptr %28
}

; Function Attrs: cold nonlazybind uwtable
define hidden noundef nonnull ptr @"_ZN3std3sys12thread_local6native5eager16Storage$LT$T$GT$10initialize17h9c6b3546246d7a0bE"(ptr noundef nonnull returned align 8 %0) unnamed_addr #16 {
  tail call void @_ZN3std3sys12thread_local11destructors10linux_like8register17hebc998dae8cbee2eE(ptr noundef nonnull %0, ptr noundef nonnull @_ZN3std3sys12thread_local6native5eager7destroy17h34a61ecd74b80e0bE)
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 1, ptr %2, align 1
  ret ptr %0
}

; Function Attrs: nounwind nonlazybind uwtable
declare hidden void @_ZN3std3sys12thread_local6native5eager7destroy17h34a61ecd74b80e0bE(ptr noundef initializes((8, 9))) unnamed_addr #22

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr47drop_in_place$LT$std..ffi..os_str..OsString$GT$17h3dcbd6d392c6cd9fE.543"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  invoke void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h17ee718c0c7a349dE"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0)
          to label %7 unwind label %2

2:                                                ; preds = %1
  %3 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0)
          to label %6 unwind label %4

4:                                                ; preds = %2
  %5 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  tail call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

6:                                                ; preds = %2
  resume { ptr, i32 } %3

7:                                                ; preds = %1
  tail call void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0)
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden noundef range(i8 0, 4) i8 @_ZN3std5panic19get_backtrace_style17h869e43e10d5d55e1E() unnamed_addr #20 personality ptr @rust_eh_personality {
  %1 = alloca [24 x i8], align 8
  %2 = alloca [24 x i8], align 8
  %3 = alloca [24 x i8], align 8
  %4 = load atomic i8, ptr @_ZN3std5panic14SHOULD_CAPTURE17h6ec7b3e7f01a7b14E monotonic, align 1
  switch i8 %4, label %5 [
    i8 1, label %10
    i8 2, label %8
    i8 3, label %9
  ]

5:                                                ; preds = %0
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %3)
  call void @_ZN3std3env6var_os17h1bb40d473090480cE(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %3, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.bc75b2dae71f64fa323d019d8562410b.79, i64 noundef 14)
  %6 = load i64, ptr %3, align 8, !range !360, !noundef !29
  %7 = icmp eq i64 %6, -9223372036854775808
  br i1 %7, label %38, label %12

8:                                                ; preds = %0
  br label %10

9:                                                ; preds = %0
  br label %10

10:                                               ; preds = %47, %46, %45, %43, %38, %9, %8, %0
  %11 = phi i8 [ 1, %8 ], [ 2, %9 ], [ 0, %0 ], [ 0, %45 ], [ 1, %46 ], [ 2, %47 ], [ %40, %38 ], [ 3, %43 ]
  ret i8 %11

12:                                               ; preds = %5
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %14 = load ptr, ptr %13, align 8
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %16 = load i64, ptr %15, align 8, !noundef !29
  switch i64 %16, label %37 [
    i64 1, label %17
    i64 4, label %27
  ]

17:                                               ; preds = %12
  %18 = icmp ne ptr %14, null
  call void @llvm.assume(i1 %18)
  %19 = load i8, ptr %14, align 1
  %20 = icmp eq i8 %19, 48
  br i1 %20, label %21, label %37

21:                                               ; preds = %17
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %2)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %2, ptr noundef nonnull align 8 dereferenceable(24) %3, i64 24, i1 false)
  invoke void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h17ee718c0c7a349dE"(ptr noalias noundef nonnull align 8 dereferenceable(24) %2)
          to label %26 unwind label %22

22:                                               ; preds = %21
  %23 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %2)
          to label %48 unwind label %24

24:                                               ; preds = %22
  %25 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

26:                                               ; preds = %21
  call void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %2)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %2)
  br label %38

27:                                               ; preds = %12
  %28 = icmp ne ptr %14, null
  call void @llvm.assume(i1 %28)
  %29 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(4) %14, ptr noundef nonnull dereferenceable(4) @anon.bc75b2dae71f64fa323d019d8562410b.80, i64 4)
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %31, label %37

31:                                               ; preds = %27
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %1, ptr noundef nonnull align 8 dereferenceable(24) %3, i64 24, i1 false)
  invoke void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h17ee718c0c7a349dE"(ptr noalias noundef nonnull align 8 dereferenceable(24) %1)
          to label %36 unwind label %32

32:                                               ; preds = %31
  %33 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %1)
          to label %48 unwind label %34

34:                                               ; preds = %32
  %35 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

36:                                               ; preds = %31
  call void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %1)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1)
  br label %38

37:                                               ; preds = %27, %17, %12
  call fastcc void @"_ZN4core3ptr47drop_in_place$LT$std..ffi..os_str..OsString$GT$17h3dcbd6d392c6cd9fE.543"(ptr noalias noundef align 8 dereferenceable(24) %3)
  br label %38

38:                                               ; preds = %37, %36, %26, %5
  %39 = phi i8 [ 2, %36 ], [ 1, %37 ], [ 3, %26 ], [ 3, %5 ]
  %40 = phi i8 [ 1, %36 ], [ 0, %37 ], [ 2, %26 ], [ 2, %5 ]
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %3)
  %41 = cmpxchg ptr @_ZN3std5panic14SHOULD_CAPTURE17h6ec7b3e7f01a7b14E, i8 0, i8 %39 monotonic monotonic, align 1
  %42 = extractvalue { i8, i1 } %41, 1
  br i1 %42, label %10, label %43

43:                                               ; preds = %38
  %44 = extractvalue { i8, i1 } %41, 0
  switch i8 %44, label %10 [
    i8 1, label %45
    i8 2, label %46
    i8 3, label %47
  ]

45:                                               ; preds = %43
  br label %10

46:                                               ; preds = %43
  br label %10

47:                                               ; preds = %43
  br label %10

48:                                               ; preds = %32, %22
  %49 = phi { ptr, i32 } [ %23, %22 ], [ %33, %32 ]
  resume { ptr, i32 } %49
}

; Function Attrs: cold noinline noreturn nonlazybind uwtable
define hidden void @_ZN3std6thread5local18panic_access_error17hc3041f24e5541bf5E(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) %0) unnamed_addr #19 {
  %2 = alloca [0 x i8], align 1
  %3 = alloca [16 x i8], align 8
  %4 = alloca [48 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %4)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store ptr %2, ptr %3, align 8
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr @"_ZN68_$LT$std..thread..local..AccessError$u20$as$u20$core..fmt..Debug$GT$3fmt17ha462707a28a28644E", ptr %5, align 8
  store ptr @anon.bc75b2dae71f64fa323d019d8562410b.85, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 1, ptr %6, align 8
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store ptr null, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %3, ptr %8, align 8
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i64 1, ptr %9, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %4, ptr noalias noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(24) %0) #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN68_$LT$std..thread..local..AccessError$u20$as$u20$core..fmt..Debug$GT$3fmt17ha462707a28a28644E"(ptr noalias nonnull readonly align 1 captures(none), ptr noalias noundef align 8 dereferenceable(24)) unnamed_addr #20

; Function Attrs: nonlazybind uwtable
define hidden noundef i64 @"_ZN3std6thread5local17LocalKey$LT$T$GT$4with17h19b509eed3b1d66eE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = load ptr, ptr %0, align 8, !nonnull !29, !noundef !29
  %3 = tail call noundef ptr %2(ptr noalias noundef align 8 dereferenceable_or_null(24) null)
  %4 = icmp eq ptr %3, null
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  tail call void @_ZN3std6thread5local18panic_access_error17hc3041f24e5541bf5E(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.bc75b2dae71f64fa323d019d8562410b.83) #53
  unreachable

6:                                                ; preds = %1
  %7 = load i64, ptr %3, align 8, !noundef !29
  ret i64 %7
}

; Function Attrs: nonlazybind uwtable
define hidden void @"_ZN3std6thread5local17LocalKey$LT$T$GT$4with17h598dcda1be864c12E"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = load ptr, ptr %0, align 8, !nonnull !29, !noundef !29
  %3 = tail call noundef ptr %2(ptr noalias noundef align 8 dereferenceable_or_null(24) null)
  %4 = icmp eq ptr %3, null
  br i1 %4, label %5, label %6

5:                                                ; preds = %1
  tail call void @_ZN3std6thread5local18panic_access_error17hc3041f24e5541bf5E(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.bc75b2dae71f64fa323d019d8562410b.83) #53
  unreachable

6:                                                ; preds = %1
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i8 0, ptr %7, align 8
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden noundef zeroext i1 @"_ZN3std6thread5local17LocalKey$LT$T$GT$4with17h62db825c2b415b0cE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = load ptr, ptr %0, align 8, !nonnull !29, !noundef !29
  %3 = tail call noundef ptr %2(ptr noalias noundef align 8 dereferenceable_or_null(24) null)
  %4 = icmp eq ptr %3, null
  br i1 %4, label %8, label %5

5:                                                ; preds = %1
  %6 = load i64, ptr %3, align 8, !noundef !29
  %7 = icmp eq i64 %6, 0
  ret i1 %7

8:                                                ; preds = %1
  tail call void @_ZN3std6thread5local18panic_access_error17hc3041f24e5541bf5E(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.bc75b2dae71f64fa323d019d8562410b.83) #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
define hidden noundef range(i8 1, 3) i8 @"_ZN3std6thread5local17LocalKey$LT$T$GT$4with17hcc20d9026ff6a60fE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8) %0, ptr noalias noundef readonly align 1 captures(none) dereferenceable(1) %1) unnamed_addr #20 personality ptr @rust_eh_personality {
  %3 = load ptr, ptr %0, align 8, !nonnull !29, !noundef !29
  %4 = load i8, ptr %1, align 1
  %5 = tail call noundef ptr %3(ptr noalias noundef align 8 dereferenceable_or_null(24) null)
  %6 = icmp eq ptr %5, null
  br i1 %6, label %14, label %7

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %9 = load i8, ptr %8, align 8, !range !478, !noundef !29
  %10 = trunc nuw i8 %9 to i1
  br i1 %10, label %15, label %11

11:                                               ; preds = %7
  %12 = load i64, ptr %5, align 8, !noundef !29
  %13 = add i64 %12, 1
  store i64 %13, ptr %5, align 8
  store i8 %4, ptr %8, align 8
  br label %15

14:                                               ; preds = %2
  tail call void @_ZN3std6thread5local18panic_access_error17hc3041f24e5541bf5E(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.bc75b2dae71f64fa323d019d8562410b.83) #53
  unreachable

15:                                               ; preds = %11, %7
  %16 = phi i8 [ 1, %7 ], [ 2, %11 ]
  ret i8 %16
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(none) uwtable
define hidden noundef nonnull ptr @_ZN3std6thread6Thread8from_raw17h7c7925a72024cc22E(ptr noundef readnone captures(ret: address, provenance) %0) unnamed_addr #23 personality ptr @rust_eh_personality {
  %2 = getelementptr inbounds i8, ptr %0, i64 -16
  ret ptr %2
}

; Function Attrs: nonlazybind uwtable
define hidden noundef range(i64 1, 0) i64 @_ZN3std6thread8ThreadId3new17hba8a0cb27b9eece5E() unnamed_addr #20 {
  %1 = load atomic i64, ptr @_ZN3std6thread8ThreadId3new7COUNTER17h101f9f62173dcc92E monotonic, align 8
  br label %2

2:                                                ; preds = %5, %0
  %3 = phi i64 [ %1, %0 ], [ %9, %5 ]
  %4 = icmp eq i64 %3, -1
  br i1 %4, label %10, label %5, !prof !353

5:                                                ; preds = %2
  %6 = add nuw i64 %3, 1
  %7 = cmpxchg weak ptr @_ZN3std6thread8ThreadId3new7COUNTER17h101f9f62173dcc92E, i64 %3, i64 %6 monotonic monotonic, align 8
  %8 = extractvalue { i64, i1 } %7, 1
  %9 = extractvalue { i64, i1 } %7, 0
  br i1 %8, label %11, label %2

10:                                               ; preds = %2
  tail call fastcc void @_ZN3std6thread8ThreadId3new9exhausted17ha0cc6f67d05aa2d0E() #53
  unreachable

11:                                               ; preds = %5
  ret i64 %6
}

; Function Attrs: cold noreturn nonlazybind uwtable
define hidden fastcc void @_ZN3std6thread8ThreadId3new9exhausted17ha0cc6f67d05aa2d0E() unnamed_addr #25 {
  %1 = alloca [48 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %1)
  store ptr @anon.bc75b2dae71f64fa323d019d8562410b.90, ptr %1, align 8
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i64 1, ptr %2, align 8
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store ptr null, ptr %3, align 8
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %4, align 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i64 0, ptr %5, align 8
  call void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %1, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.bc75b2dae71f64fa323d019d8562410b.91) #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h05ab7ecd738f8002E"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8), ptr noalias noundef readonly align 8 captures(none) dereferenceable(24)) unnamed_addr #20

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN3std3env6var_os17h1bb40d473090480cE(ptr dead_on_unwind noalias noundef writable sret([24 x i8]) align 8 captures(address) dereferenceable(24) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2) unnamed_addr #20 personality ptr @rust_eh_personality {
  tail call void @_ZN3std3sys3env4unix6getenv17h5ad91241c8ca908eE(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2)
  ret void
}

; Function Attrs: cold noreturn nonlazybind uwtable
define hidden void @_RNvCs1QLEhZ2QfLZ_7___rustc8___rg_oom(i64 noundef %0, i64 noundef %1) unnamed_addr #25 {
  %3 = add i64 %1, -1
  %4 = icmp sgt i64 %3, -1
  tail call void @llvm.assume(i1 %4)
  tail call void @_ZN3std5alloc8rust_oom17h9419ca2d425caab7E(i64 noundef %1, i64 noundef %0) #53
  unreachable
}

; Function Attrs: cold noreturn nonlazybind uwtable
define hidden void @_ZN3std5alloc8rust_oom17h9419ca2d425caab7E(i64 noundef range(i64 1, -9223372036854775807) %0, i64 noundef %1) unnamed_addr #25 {
  %3 = load atomic ptr, ptr @_ZN3std5alloc4HOOK17h7130ea174ab09dc3E acquire, align 8
  %4 = icmp eq ptr %3, null
  %5 = select i1 %4, ptr @_ZN3std5alloc24default_alloc_error_hook17h341a346933ce5c23E, ptr %3
  tail call void %5(i64 noundef %0, i64 noundef %1)
  tail call void @_ZN3std7process5abort17hea5afc4016cb7294E() #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
declare hidden void @_ZN3std5alloc24default_alloc_error_hook17h341a346933ce5c23E(i64 range(i64 1, -9223372036854775807), i64 noundef) unnamed_addr #20

; Function Attrs: noinline nonlazybind uwtable
define hidden void @"_ZN5alloc4sync16Arc$LT$T$C$A$GT$9drop_slow17he9c118ac83d419dfE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8) %0) unnamed_addr #21 personality ptr @rust_eh_personality {
  %2 = load ptr, ptr %0, align 8, !nonnull !29, !noundef !29
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 24
  invoke void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h17ee718c0c7a349dE"(ptr noalias noundef nonnull align 8 dereferenceable(24) %3)
          to label %11 unwind label %4

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          cleanup
  %6 = load i64, ptr %3, align 8, !alias.scope !513
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %17, label %8

8:                                                ; preds = %4
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %10 = load ptr, ptr %9, align 8, !alias.scope !522, !nonnull !29, !noundef !29
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %10, i64 noundef %6, i64 noundef range(i64 1, -9223372036854775807) 1) #37, !noalias !523
  br label %17

11:                                               ; preds = %1
  %12 = load i64, ptr %3, align 8, !alias.scope !513
  %13 = icmp eq i64 %12, 0
  br i1 %13, label %24, label %14

14:                                               ; preds = %11
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %16 = load ptr, ptr %15, align 8, !alias.scope !522, !nonnull !29, !noundef !29
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %16, i64 noundef %12, i64 noundef range(i64 1, -9223372036854775807) 1) #37, !noalias !526
  br label %24

17:                                               ; preds = %8, %4
  %18 = icmp eq ptr %2, inttoptr (i64 -1 to ptr)
  br i1 %18, label %32, label %19

19:                                               ; preds = %17
  %20 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %21 = atomicrmw sub ptr %20, i64 1 release, align 8
  %22 = icmp eq i64 %21, 1
  br i1 %22, label %23, label %32

23:                                               ; preds = %19
  fence acquire
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %2, i64 noundef 48, i64 noundef 8) #37
  br label %32

24:                                               ; preds = %14, %11
  %25 = icmp eq ptr %2, inttoptr (i64 -1 to ptr)
  br i1 %25, label %31, label %26

26:                                               ; preds = %24
  %27 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %28 = atomicrmw sub ptr %27, i64 1 release, align 8
  %29 = icmp eq i64 %28, 1
  br i1 %29, label %30, label %31

30:                                               ; preds = %26
  fence acquire
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %2, i64 noundef 48, i64 noundef 8) #37
  br label %31

31:                                               ; preds = %30, %26, %24
  ret void

32:                                               ; preds = %23, %19, %17
  resume { ptr, i32 } %5
}

; Function Attrs: cold nounwind nonlazybind uwtable
define hidden fastcc void @_ZN5alloc7raw_vec11finish_grow17haadee78458b87993E(ptr dead_on_unwind noalias noundef nonnull writable writeonly align 8 captures(none) dereferenceable(24) initializes((0, 24)) %0, i64 noundef range(i64 1, -9223372036854775807) %1, i64 noundef %2, ptr dead_on_return noalias noundef nonnull readonly align 8 captures(none) dereferenceable(24) %3) unnamed_addr #17 {
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %6 = load i64, ptr %5, align 8, !range !360, !noundef !29
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %23, label %8

8:                                                ; preds = %4
  %9 = load ptr, ptr %3, align 8, !nonnull !29, !noundef !29
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %11 = load i64, ptr %10, align 8, !noundef !29
  %12 = icmp eq i64 %6, %1
  tail call void @llvm.assume(i1 %12)
  %13 = icmp eq i64 %11, 0
  br i1 %13, label %14, label %20

14:                                               ; preds = %8
  %15 = icmp eq i64 %2, 0
  br i1 %15, label %16, label %18

16:                                               ; preds = %14
  %17 = getelementptr i8, ptr null, i64 %1
  br label %29

18:                                               ; preds = %14
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #37
  %19 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc12___rust_alloc(i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) %1) #37
  br label %29

20:                                               ; preds = %8
  %21 = icmp uge i64 %2, %11
  tail call void @llvm.assume(i1 %21)
  %22 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_realloc(ptr noundef nonnull %9, i64 noundef %11, i64 noundef range(i64 1, -9223372036854775807) %1, i64 noundef %2) #37
  br label %29

23:                                               ; preds = %4
  %24 = icmp eq i64 %2, 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %23
  %26 = getelementptr i8, ptr null, i64 %1
  br label %29

27:                                               ; preds = %23
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #37
  %28 = tail call noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc12___rust_alloc(i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) %1) #37
  br label %29

29:                                               ; preds = %27, %25, %20, %18, %16
  %30 = phi ptr [ %22, %20 ], [ %17, %16 ], [ %19, %18 ], [ %26, %25 ], [ %28, %27 ]
  %31 = icmp eq ptr %30, null
  %32 = inttoptr i64 %1 to ptr
  %33 = select i1 %31, ptr %32, ptr %30
  %34 = zext i1 %31 to i64
  %35 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %33, ptr %35, align 8
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %2, ptr %36, align 8
  store i64 %34, ptr %0, align 8
  ret void
}

; Function Attrs: noinline nonlazybind uwtable
define hidden void @"_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h48aa27d81eb31477E"(ptr noalias noundef align 8 captures(none) dereferenceable(16) %0) unnamed_addr #21 personality ptr @rust_eh_personality {
  %2 = alloca [24 x i8], align 8
  %3 = alloca [24 x i8], align 8
  %4 = load i64, ptr %0, align 8, !range !354, !noundef !29
  tail call void @llvm.experimental.noalias.scope.decl(metadata !529)
  %5 = shl nuw i64 %4, 1
  %6 = tail call i64 @llvm.umax.i64(i64 %5, i64 range(i64 0, -1) 4)
  %7 = shl i64 %6, 4
  %8 = icmp samesign ugt i64 %4, 576460752303423487
  %9 = icmp ugt i64 %7, 9223372036854775800
  %10 = select i1 %8, i1 true, i1 %9
  br i1 %10, label %28, label %11, !prof !355

11:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %3), !noalias !529
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %2), !noalias !529
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %13 = icmp eq i64 %4, 0
  br i1 %13, label %18, label %14

14:                                               ; preds = %11
  %15 = load ptr, ptr %12, align 8, !alias.scope !529, !nonnull !29, !noundef !29
  %16 = shl nuw nsw i64 %4, 4
  store ptr %15, ptr %2, align 8, !alias.scope !532, !noalias !529
  %17 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store i64 %16, ptr %17, align 8, !alias.scope !532, !noalias !529
  br label %18

18:                                               ; preds = %14, %11
  %19 = phi i64 [ 8, %14 ], [ 0, %11 ]
  %20 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i64 %19, ptr %20, align 8, !alias.scope !532, !noalias !529
  call fastcc void @_ZN5alloc7raw_vec11finish_grow17haadee78458b87993E(ptr noalias noundef align 8 captures(address) dereferenceable(24) %3, i64 noundef 8, i64 noundef %7, ptr noalias noundef readonly align 8 captures(address) dereferenceable(24) %2), !noalias !529
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %2), !noalias !529
  %21 = load i64, ptr %3, align 8, !range !359, !noalias !529, !noundef !29
  %22 = trunc nuw i64 %21 to i1
  %23 = getelementptr inbounds nuw i8, ptr %3, i64 8
  br i1 %22, label %24, label %31

24:                                               ; preds = %18
  %25 = load i64, ptr %23, align 8, !range !360, !noalias !529, !noundef !29
  %26 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %27 = load i64, ptr %26, align 8, !noalias !529
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %3), !noalias !529
  br label %28

28:                                               ; preds = %24, %1
  %29 = phi i64 [ undef, %1 ], [ %27, %24 ]
  %30 = phi i64 [ 0, %1 ], [ %25, %24 ]
  tail call void @_ZN5alloc7raw_vec12handle_error17h2546acf93648cb86E(i64 noundef %30, i64 %29) #53
  unreachable

31:                                               ; preds = %18
  %32 = load ptr, ptr %23, align 8, !noalias !529, !nonnull !29, !noundef !29
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %3), !noalias !529
  store ptr %32, ptr %12, align 8, !alias.scope !529
  %33 = icmp sgt i64 %6, -1
  tail call void @llvm.assume(i1 %33)
  store i64 %6, ptr %0, align 8, !alias.scope !529
  ret void
}

; Function Attrs: nounwind nonlazybind uwtable
define hidden void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(16) %0) unnamed_addr #22 {
  %2 = load i64, ptr %0, align 8
  %3 = icmp eq i64 %2, 0
  br i1 %3, label %7, label %4

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load ptr, ptr %5, align 8, !nonnull !29, !noundef !29
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %6, i64 noundef %2, i64 noundef range(i64 1, -9223372036854775807) 1) #37
  br label %7

7:                                                ; preds = %4, %1
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(argmem: write) uwtable
define hidden void @_ZN3std4sync6poison10map_result17h2d2b88de1fc79d43E(ptr dead_on_unwind noalias noundef writable writeonly sret([24 x i8]) align 8 captures(none) dereferenceable(24) initializes((0, 24)) %0, i1 noundef zeroext %1, ptr noundef nonnull align 8 %2) unnamed_addr #43 {
  %4 = zext i1 %1 to i64
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %5, ptr %6, align 8
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %2, ptr %7, align 8
  store i64 %4, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(argmem: write) uwtable
define hidden void @_ZN3std4sync6poison10map_result17h50aec882be779b28E(ptr dead_on_unwind noalias noundef writable writeonly sret([24 x i8]) align 8 captures(none) dereferenceable(24) initializes((0, 17)) %0, i1 noundef zeroext %1, i8 noundef %2, ptr noundef nonnull align 4 %3) unnamed_addr #43 {
  %5 = zext i1 %1 to i64
  %6 = and i8 %2, 1
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %3, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i8 %6, ptr %8, align 8
  store i64 %5, ptr %0, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(argmem: write) uwtable
define hidden void @_ZN3std4sync6poison10map_result17h6348d1c69c45512aE(ptr dead_on_unwind noalias noundef writable writeonly sret([24 x i8]) align 8 captures(none) dereferenceable(24) initializes((0, 17)) %0, i1 noundef zeroext %1, i8 noundef %2, ptr noundef nonnull align 8 %3) unnamed_addr #43 {
  %5 = zext i1 %1 to i64
  %6 = and i8 %2, 1
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %3, ptr %7, align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i8 %6, ptr %8, align 8
  store i64 %5, ptr %0, align 8
  ret void
}

; Function Attrs: cold noinline nonlazybind uwtable
define hidden noundef zeroext i1 @_ZN3std9panicking11panic_count17is_zero_slow_path17h0941546548007a1dE() unnamed_addr #44 {
  %1 = tail call noundef zeroext i1 @"_ZN3std6thread5local17LocalKey$LT$T$GT$4with17h62db825c2b415b0cE"(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(8) @anon.bb9016958efe0580f15baffae1ffbb38.6)
  ret i1 %1
}

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN3std9panicking11panic_count19finished_panic_hook17h7ab0510843eb6188E() unnamed_addr #20 {
  tail call void @"_ZN3std6thread5local17LocalKey$LT$T$GT$4with17h598dcda1be864c12E"(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(8) @anon.bb9016958efe0580f15baffae1ffbb38.6)
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden noundef range(i8 0, 3) i8 @_ZN3std9panicking11panic_count8increase17h294bbb53e68a902aE(i1 noundef zeroext %0) unnamed_addr #20 {
  %2 = alloca [1 x i8], align 1
  %3 = zext i1 %0 to i8
  store i8 %3, ptr %2, align 1
  %4 = atomicrmw add ptr @_ZN3std9panicking11panic_count18GLOBAL_PANIC_COUNT17hae1b51fffade13abE, i64 1 monotonic, align 8
  %5 = icmp sgt i64 %4, -1
  br i1 %5, label %6, label %8

6:                                                ; preds = %1
  %7 = call noundef i8 @"_ZN3std6thread5local17LocalKey$LT$T$GT$4with17hcc20d9026ff6a60fE"(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(8) @anon.bb9016958efe0580f15baffae1ffbb38.6, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) dereferenceable(1) %2)
  br label %8

8:                                                ; preds = %6, %1
  %9 = phi i8 [ %7, %6 ], [ 0, %1 ]
  ret i8 %9
}

; Function Attrs: nonlazybind uwtable
define hidden noundef i64 @_ZN3std9panicking11panic_count9get_count17h77020e9322062869E() unnamed_addr #20 {
  %1 = tail call noundef i64 @"_ZN3std6thread5local17LocalKey$LT$T$GT$4with17h19b509eed3b1d66eE"(ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(8) @anon.bb9016958efe0580f15baffae1ffbb38.6)
  ret i64 %1
}

; Function Attrs: nonlazybind uwtable
define hidden { i64, ptr } @_ZN3std2io5stdio22try_set_output_capture17h69ba45833ddbbce1E(ptr noundef %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = alloca [8 x i8], align 8
  %3 = icmp eq ptr %0, null
  br i1 %3, label %4, label %12

4:                                                ; preds = %1
  %5 = load atomic i8, ptr @_ZN3std2io5stdio19OUTPUT_CAPTURE_USED17h71b5b109888cde8aE.0 monotonic, align 1
  %6 = icmp eq i8 %5, 0
  br i1 %6, label %7, label %12

7:                                                ; preds = %43, %4
  %8 = phi ptr [ %44, %43 ], [ null, %4 ]
  %9 = phi i64 [ %45, %43 ], [ 0, %4 ]
  %10 = insertvalue { i64, ptr } poison, i64 %9, 0
  %11 = insertvalue { i64, ptr } %10, ptr %8, 1
  ret { i64, ptr } %11

12:                                               ; preds = %4, %1
  store atomic i8 1, ptr @_ZN3std2io5stdio19OUTPUT_CAPTURE_USED17h71b5b109888cde8aE.0 monotonic, align 1
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2)
  store ptr %0, ptr %2, align 8
  %13 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr @"_ZN3std2io5stdio14OUTPUT_CAPTURE29_$u7b$$u7b$constant$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$3VAL17h8917e6542b635b84E")
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 8
  %15 = load i8, ptr %14, align 1, !range !535, !noundef !29
  switch i8 %15, label %16 [
    i8 0, label %17
    i8 1, label %29
    i8 2, label %22
  ], !prof !536

16:                                               ; preds = %12
  unreachable

17:                                               ; preds = %12
  %18 = invoke noundef ptr @"_ZN3std3sys12thread_local6native5eager16Storage$LT$T$GT$10initialize17h9c6b3546246d7a0bE"(ptr noundef nonnull align 8 %13)
          to label %19 unwind label %33

19:                                               ; preds = %17
  %20 = icmp eq ptr %18, null
  %21 = load ptr, ptr %2, align 8
  br i1 %20, label %22, label %29

22:                                               ; preds = %19, %12
  %23 = phi ptr [ %0, %12 ], [ %21, %19 ]
  %24 = icmp eq ptr %23, null
  br i1 %24, label %43, label %25

25:                                               ; preds = %22
  %26 = atomicrmw sub ptr %23, i64 1 release, align 8, !noalias !537
  %27 = icmp eq i64 %26, 1
  br i1 %27, label %28, label %43

28:                                               ; preds = %25
  fence acquire
  call void @"_ZN5alloc4sync16Arc$LT$T$C$A$GT$9drop_slow17he9c118ac83d419dfE"(ptr noalias noundef nonnull align 8 dereferenceable(8) %2)
  br label %43

29:                                               ; preds = %19, %12
  %30 = phi ptr [ %21, %19 ], [ %0, %12 ]
  %31 = phi ptr [ %18, %19 ], [ %13, %12 ]
  %32 = load ptr, ptr %31, align 8, !noundef !29
  store ptr %30, ptr %31, align 8
  br label %43

33:                                               ; preds = %17
  %34 = landingpad { ptr, i32 }
          cleanup
  tail call void @llvm.experimental.noalias.scope.decl(metadata !546)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !549)
  %35 = load ptr, ptr %2, align 8, !alias.scope !552, !noundef !29
  %36 = icmp eq ptr %35, null
  br i1 %36, label %46, label %37

37:                                               ; preds = %33
  %38 = atomicrmw sub ptr %35, i64 1 release, align 8, !noalias !553
  %39 = icmp eq i64 %38, 1
  br i1 %39, label %40, label %46

40:                                               ; preds = %37
  fence acquire
  invoke void @"_ZN5alloc4sync16Arc$LT$T$C$A$GT$9drop_slow17he9c118ac83d419dfE"(ptr noalias noundef nonnull align 8 dereferenceable(8) %2)
          to label %46 unwind label %41

41:                                               ; preds = %40
  %42 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

43:                                               ; preds = %29, %28, %25, %22
  %44 = phi ptr [ %32, %29 ], [ undef, %22 ], [ undef, %25 ], [ undef, %28 ]
  %45 = phi i64 [ 0, %29 ], [ 1, %22 ], [ 1, %25 ], [ 1, %28 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2)
  br label %7

46:                                               ; preds = %40, %37, %33
  resume { ptr, i32 } %34
}

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN3std3sys3env4unix6getenv17h5ad91241c8ca908eE(ptr dead_on_unwind noalias noundef writable writeonly sret([24 x i8]) align 8 captures(none) dereferenceable(24) initializes((0, 8)) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2) unnamed_addr #20 {
  %4 = alloca [24 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4)
  %5 = icmp ugt i64 %2, 383
  br i1 %5, label %7, label %6, !prof !353

6:                                                ; preds = %3
  call void @_ZN3std3sys3pal6common14small_c_string19run_with_cstr_stack17h04f77ca42ae6c514E(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %4, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2, ptr noundef nonnull align 1 inttoptr (i64 1 to ptr), ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) @anon.7b33aeb34b5389b82a5a31bdbf8976e8.13)
  br label %8

7:                                                ; preds = %3
  call void @_ZN3std3sys3pal6common14small_c_string24run_with_cstr_allocating17hd8a64456e9e76a17E(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %4, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2, ptr noundef nonnull align 1 inttoptr (i64 1 to ptr), ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) @anon.7b33aeb34b5389b82a5a31bdbf8976e8.13)
  br label %8

8:                                                ; preds = %7, %6
  %9 = load i64, ptr %4, align 8, !range !558, !noundef !29
  %10 = icmp eq i64 %9, -9223372036854775807
  br i1 %10, label %14, label %11

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %13, ptr noundef nonnull align 8 dereferenceable(16) %12, i64 16, i1 false)
  br label %15

14:                                               ; preds = %8
  call fastcc void @"_ZN4core3ptr127drop_in_place$LT$core..result..Result$LT$core..option..Option$LT$std..ffi..os_str..OsString$GT$$C$std..io..error..Error$GT$$GT$17h42f99ab6f0674bf9E"(ptr noalias noundef align 8 dereferenceable(24) %4)
  br label %15

15:                                               ; preds = %14, %11
  %16 = phi i64 [ -9223372036854775808, %14 ], [ %9, %11 ]
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4)
  store i64 %16, ptr %0, align 8
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr127drop_in_place$LT$core..result..Result$LT$core..option..Option$LT$std..ffi..os_str..OsString$GT$$C$std..io..error..Error$GT$$GT$17h42f99ab6f0674bf9E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = load i64, ptr %0, align 8, !range !558, !noundef !29
  switch i64 %2, label %3 [
    i64 -9223372036854775807, label %4
    i64 -9223372036854775808, label %42
  ]

3:                                                ; preds = %1
  tail call void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %0)
  br label %42

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load ptr, ptr %5, align 8, !nonnull !29, !noundef !29
  %7 = ptrtoint ptr %6 to i64
  %8 = and i64 %7, 3
  %9 = icmp eq i64 %8, 1
  br i1 %9, label %10, label %42, !prof !559

10:                                               ; preds = %4
  %11 = getelementptr i8, ptr %6, i64 -1
  %12 = icmp ne ptr %11, null
  tail call void @llvm.assume(i1 %12)
  %13 = load ptr, ptr %11, align 8
  %14 = getelementptr i8, ptr %6, i64 7
  %15 = load ptr, ptr %14, align 8, !nonnull !29, !align !410, !noundef !29
  %16 = load ptr, ptr %15, align 8, !invariant.load !29
  %17 = icmp eq ptr %16, null
  br i1 %17, label %20, label %18

18:                                               ; preds = %10
  %19 = icmp ne ptr %13, null
  tail call void @llvm.assume(i1 %19)
  invoke void %16(ptr noundef nonnull %13)
          to label %20 unwind label %30

20:                                               ; preds = %18, %10
  %21 = icmp ne ptr %13, null
  tail call void @llvm.assume(i1 %21)
  %22 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %23 = load i64, ptr %22, align 8, !range !354, !invariant.load !29
  %24 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %25 = load i64, ptr %24, align 8, !range !476, !invariant.load !29
  %26 = add i64 %25, -1
  %27 = icmp sgt i64 %26, -1
  tail call void @llvm.assume(i1 %27)
  %28 = icmp eq i64 %23, 0
  br i1 %28, label %41, label %29

29:                                               ; preds = %20
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %13, i64 noundef range(i64 1, -9223372036854775808) %23, i64 noundef range(i64 1, -9223372036854775807) %25) #37
  br label %41

30:                                               ; preds = %18
  %31 = landingpad { ptr, i32 }
          cleanup
  %32 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %33 = load i64, ptr %32, align 8, !range !354, !invariant.load !29
  %34 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %35 = load i64, ptr %34, align 8, !range !476, !invariant.load !29
  %36 = add i64 %35, -1
  %37 = icmp sgt i64 %36, -1
  tail call void @llvm.assume(i1 %37)
  %38 = icmp eq i64 %33, 0
  br i1 %38, label %40, label %39

39:                                               ; preds = %30
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %13, i64 noundef range(i64 1, -9223372036854775808) %33, i64 noundef range(i64 1, -9223372036854775807) %35) #37
  br label %40

40:                                               ; preds = %39, %30
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %11, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %31

41:                                               ; preds = %29, %20
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %11, i64 noundef 24, i64 noundef 8) #37
  br label %42

42:                                               ; preds = %41, %4, %3, %1
  ret void
}

; Function Attrs: mustprogress nofree norecurse nounwind nonlazybind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define hidden noundef i64 @_ZN3std6thread11main_thread3get17h422e7de94df43145E() unnamed_addr #45 {
  %1 = load atomic i64, ptr @_ZN3std6thread11main_thread4MAIN17he08dbfe964edeb58E.0 monotonic, align 8
  ret i64 %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(none) uwtable
define hidden void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h17ee718c0c7a349dE"(ptr noalias noundef readnone align 8 captures(none) dereferenceable(24) %0) unnamed_addr #23 {
  ret void
}

; Function Attrs: nofree nounwind nonlazybind uwtable
define hidden noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc11___rdl_alloc(i64 noundef %0, i64 noundef %1) unnamed_addr #46 personality ptr @rust_eh_personality {
  %3 = alloca [8 x i8], align 8
  %4 = add i64 %1, -1
  %5 = icmp sgt i64 %4, -1
  tail call void @llvm.assume(i1 %5)
  %6 = icmp ult i64 %1, 17
  %7 = icmp ule i64 %1, %0
  %8 = and i1 %6, %7
  br i1 %8, label %15, label %9

9:                                                ; preds = %2
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %3)
  store ptr null, ptr %3, align 8
  %10 = tail call noundef i64 @llvm.umax.i64(i64 range(i64 1, -9223372036854775807) %1, i64 8)
  %11 = call noundef i32 @posix_memalign(ptr noundef nonnull %3, i64 noundef %10, i64 noundef %0) #37
  %12 = icmp eq i32 %11, 0
  %13 = load ptr, ptr %3, align 8
  %14 = select i1 %12, ptr %13, ptr null
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %3)
  br label %17

15:                                               ; preds = %2
  %16 = tail call noundef ptr @malloc(i64 noundef %0) #37
  br label %17

17:                                               ; preds = %15, %9
  %18 = phi ptr [ %16, %15 ], [ %14, %9 ]
  ret ptr %18
}

; Function Attrs: nofree nounwind nonlazybind uwtable
declare noundef i32 @posix_memalign(ptr noundef, i64 noundef, i64 noundef) unnamed_addr #46

; Function Attrs: mustprogress nofree nounwind nonlazybind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) uwtable
declare noalias noundef ptr @malloc(i64 noundef) unnamed_addr #47

; Function Attrs: mustprogress nounwind nonlazybind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable
define hidden void @_RNvCs1QLEhZ2QfLZ_7___rustc13___rdl_dealloc(ptr noundef captures(none) %0, i64 noundef %1, i64 noundef %2) unnamed_addr #48 {
  %4 = add i64 %2, -1
  %5 = icmp sgt i64 %4, -1
  tail call void @llvm.assume(i1 %5)
  tail call void @free(ptr noundef %0) #37
  ret void
}

; Function Attrs: nounwind nonlazybind uwtable
define hidden noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc13___rdl_realloc(ptr noundef captures(none) %0, i64 noundef %1, i64 noundef %2, i64 noundef %3) unnamed_addr #22 personality ptr @rust_eh_personality {
  %5 = alloca [8 x i8], align 8
  %6 = add i64 %2, -1
  %7 = icmp sgt i64 %6, -1
  tail call void @llvm.assume(i1 %7)
  %8 = icmp ult i64 %2, 17
  %9 = icmp ule i64 %2, %3
  %10 = and i1 %8, %9
  br i1 %10, label %20, label %11

11:                                               ; preds = %4
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %5)
  store ptr null, ptr %5, align 8
  %12 = tail call noundef i64 @llvm.umax.i64(i64 range(i64 1, -9223372036854775807) %2, i64 8)
  %13 = call noundef i32 @posix_memalign(ptr noundef nonnull %5, i64 noundef %12, i64 noundef %3) #37
  %14 = icmp ne i32 %13, 0
  %15 = load ptr, ptr %5, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %5)
  %16 = icmp eq ptr %15, null
  %17 = select i1 %14, i1 true, i1 %16
  br i1 %17, label %22, label %18

18:                                               ; preds = %11
  %19 = call noundef i64 @llvm.umin.i64(i64 %3, i64 %1)
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %15, ptr align 1 %0, i64 %19, i1 false)
  call void @free(ptr noundef %0) #37
  br label %22

20:                                               ; preds = %4
  %21 = tail call noundef ptr @realloc(ptr noundef %0, i64 noundef %3) #37
  br label %22

22:                                               ; preds = %20, %18, %11
  %23 = phi ptr [ %21, %20 ], [ %15, %18 ], [ null, %11 ]
  ret ptr %23
}

; Function Attrs: mustprogress nounwind nonlazybind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable
declare noalias noundef ptr @realloc(ptr allocptr noundef captures(none), i64 noundef) unnamed_addr #49

; Function Attrs: nofree nounwind nonlazybind uwtable
define hidden noundef ptr @_RNvCs1QLEhZ2QfLZ_7___rustc18___rdl_alloc_zeroed(i64 noundef %0, i64 noundef %1) unnamed_addr #46 personality ptr @rust_eh_personality {
  %3 = alloca [8 x i8], align 8
  %4 = add i64 %1, -1
  %5 = icmp sgt i64 %4, -1
  tail call void @llvm.assume(i1 %5)
  %6 = icmp ult i64 %1, 17
  %7 = icmp ule i64 %1, %0
  %8 = and i1 %6, %7
  br i1 %8, label %16, label %9

9:                                                ; preds = %2
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %3)
  store ptr null, ptr %3, align 8
  %10 = tail call noundef i64 @llvm.umax.i64(i64 range(i64 1, -9223372036854775807) %1, i64 8)
  %11 = call noundef i32 @posix_memalign(ptr noundef nonnull %3, i64 noundef %10, i64 noundef %0) #37
  %12 = icmp ne i32 %11, 0
  %13 = load ptr, ptr %3, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %3)
  %14 = icmp eq ptr %13, null
  %15 = select i1 %12, i1 true, i1 %14
  br i1 %15, label %19, label %18

16:                                               ; preds = %2
  %17 = tail call noundef ptr @calloc(i64 noundef %0, i64 noundef 1) #37
  br label %19

18:                                               ; preds = %9
  call void @llvm.memset.p0.i64(ptr nonnull align 1 %13, i8 0, i64 %0, i1 false)
  br label %19

19:                                               ; preds = %18, %16, %9
  %20 = phi ptr [ %17, %16 ], [ null, %9 ], [ %13, %18 ]
  ret ptr %20
}

; Function Attrs: mustprogress nofree nounwind nonlazybind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) uwtable
declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) unnamed_addr #50

; Function Attrs: nonlazybind uwtable
define hidden noundef ptr @_ZN3std2io5Write9write_fmt17h7f43f95a9f0ae4faE(ptr noalias noundef align 8 dereferenceable(24) %0, ptr dead_on_return noalias noundef align 8 captures(address) dereferenceable(48) %1) unnamed_addr #20 personality ptr @rust_eh_personality {
  %3 = load ptr, ptr %1, align 8, !nonnull !29, !align !410, !noundef !29
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !noundef !29
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %7 = load i64, ptr %6, align 8, !noundef !29
  switch i64 %5, label %10 [
    i64 0, label %8
    i64 1, label %17
  ]

8:                                                ; preds = %2
  %9 = icmp eq i64 %7, 0
  br i1 %9, label %11, label %10

10:                                               ; preds = %17, %8, %2
  br label %11

11:                                               ; preds = %19, %10, %8
  %12 = phi i64 [ undef, %10 ], [ %22, %19 ], [ 0, %8 ]
  %13 = phi ptr [ null, %10 ], [ %20, %19 ], [ inttoptr (i64 1 to ptr), %8 ]
  %14 = icmp ne ptr %13, null
  %15 = tail call i1 @llvm.is.constant.i1(i1 %14)
  %16 = and i1 %15, %14
  br i1 %16, label %25, label %23

17:                                               ; preds = %2
  %18 = icmp eq i64 %7, 0
  br i1 %18, label %19, label %10

19:                                               ; preds = %17
  %20 = load ptr, ptr %3, align 8, !nonnull !29, !align !411, !noundef !29
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %22 = load i64, ptr %21, align 8, !noundef !29
  br label %11

23:                                               ; preds = %11
  %24 = tail call noundef ptr @_ZN3std2io17default_write_fmt17hc990884f3953a138E(ptr noalias noundef nonnull align 8 dereferenceable(24) %0, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %1)
  br label %38

25:                                               ; preds = %11
  tail call void @llvm.experimental.noalias.scope.decl(metadata !560)
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %27 = load ptr, ptr %0, align 8, !alias.scope !560, !noalias !563, !nonnull !29, !align !411, !noundef !29
  %28 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %29 = load i64, ptr %28, align 8, !alias.scope !560, !noalias !563, !noundef !29
  tail call void @llvm.experimental.noalias.scope.decl(metadata !565)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !568)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !570)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !572)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !575)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !577)
  %30 = load i64, ptr %26, align 8, !alias.scope !579, !noalias !580, !noundef !29
  %31 = tail call noundef i64 @llvm.umin.i64(i64 %29, i64 %30)
  %32 = sub nuw i64 %29, %31
  %33 = getelementptr inbounds nuw i8, ptr %27, i64 %31
  %34 = tail call noundef i64 @llvm.umin.i64(i64 %32, i64 %12)
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %33, ptr nonnull readonly align 1 %13, i64 %34, i1 false), !alias.scope !581, !noalias !585
  %35 = add i64 %34, %30
  store i64 %35, ptr %26, align 8, !alias.scope !579, !noalias !580
  %36 = icmp ult i64 %32, %12
  %37 = select i1 %36, ptr @anon.a3b51bfd7273ae0b2428f9302b4a6a52.3, ptr null
  br label %38

38:                                               ; preds = %25, %23
  %39 = phi ptr [ %37, %25 ], [ %24, %23 ]
  ret ptr %39
}

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN3std3sys3pal6common14small_c_string19run_with_cstr_stack17h04f77ca42ae6c514E(ptr dead_on_unwind noalias noundef writable sret([24 x i8]) align 8 captures(address) dereferenceable(24) %0, ptr noalias noundef nonnull readonly align 1 captures(none) %1, i64 noundef %2, ptr noundef nonnull align 1 %3, ptr noalias noundef readonly align 8 captures(none) dereferenceable(48) %4) unnamed_addr #20 {
  %6 = alloca [24 x i8], align 8
  %7 = alloca [384 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 384, ptr nonnull %7)
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %7, ptr nonnull align 1 %1, i64 %2, i1 false)
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 %2
  store i8 0, ptr %8, align 1
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %6)
  %9 = add i64 %2, 1
  call void @_ZN4core3ffi5c_str4CStr19from_bytes_with_nul17h883c9be065a77d17E(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %6, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %7, i64 noundef %9)
  %10 = load i64, ptr %6, align 8, !range !359, !noundef !29
  %11 = trunc nuw i64 %10 to i1
  br i1 %11, label %12, label %14

12:                                               ; preds = %5
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr @anon.a3b51bfd7273ae0b2428f9302b4a6a52.5, ptr %13, align 8
  store i64 -9223372036854775807, ptr %0, align 8
  br label %21

14:                                               ; preds = %5
  %15 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %16 = load ptr, ptr %15, align 8, !nonnull !29, !align !411, !noundef !29
  %17 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %18 = load i64, ptr %17, align 8, !noundef !29
  %19 = getelementptr inbounds nuw i8, ptr %4, i64 40
  %20 = load ptr, ptr %19, align 8, !invariant.load !29, !nonnull !29
  call void %20(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %0, ptr noundef nonnull align 1 %3, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %16, i64 noundef %18)
  br label %21

21:                                               ; preds = %14, %12
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %6)
  call void @llvm.lifetime.end.p0(i64 384, ptr nonnull %7)
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr105drop_in_place$LT$core..result..Result$LT$alloc..ffi..c_str..CString$C$alloc..ffi..c_str..NulError$GT$$GT$17h04a5a0d1226776dcE.915"(ptr noalias noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = load i64, ptr %0, align 8, !range !360, !noundef !29
  %3 = icmp eq i64 %2, -9223372036854775808
  br i1 %3, label %4, label %11

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load ptr, ptr %5, align 8, !nonnull !29, !align !411, !noundef !29
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load i64, ptr %7, align 8
  store i8 0, ptr %6, align 1
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %18, label %10

10:                                               ; preds = %4
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %6, i64 noundef range(i64 1, 0) %8, i64 noundef 1) #37
  br label %18

11:                                               ; preds = %1
  invoke void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h17ee718c0c7a349dE"(ptr noalias noundef nonnull align 8 dereferenceable(32) %0)
          to label %17 unwind label %12

12:                                               ; preds = %11
  %13 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(32) %0)
          to label %16 unwind label %14

14:                                               ; preds = %12
  %15 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  tail call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

16:                                               ; preds = %12
  resume { ptr, i32 } %13

17:                                               ; preds = %11
  tail call void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(32) %0)
  br label %18

18:                                               ; preds = %17, %10, %4
  ret void
}

; Function Attrs: cold noinline nonlazybind uwtable
define hidden void @_ZN3std3sys3pal6common14small_c_string24run_with_cstr_allocating17hd8a64456e9e76a17E(ptr dead_on_unwind noalias noundef writable sret([24 x i8]) align 8 captures(address) dereferenceable(24) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2, ptr noundef nonnull align 1 %3, ptr noalias noundef readonly align 8 captures(none) dereferenceable(48) %4) unnamed_addr #44 personality ptr @rust_eh_personality {
  %6 = alloca [32 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %6)
  call void @"_ZN81_$LT$$RF$$u5b$u8$u5d$$u20$as$u20$alloc..ffi..c_str..CString..new..SpecNewImpl$GT$13spec_new_impl17hd42ed46981b64645E"(ptr noalias noundef nonnull sret([32 x i8]) align 8 captures(address) dereferenceable(32) %6, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %1, i64 noundef %2)
  %7 = load i64, ptr %6, align 8, !range !360, !noundef !29
  %8 = icmp eq i64 %7, -9223372036854775808
  br i1 %8, label %11, label %9

9:                                                ; preds = %5
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr @anon.a3b51bfd7273ae0b2428f9302b4a6a52.5, ptr %10, align 8
  store i64 -9223372036854775807, ptr %0, align 8
  br label %28

11:                                               ; preds = %5
  %12 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %13 = load ptr, ptr %12, align 8, !nonnull !29, !align !411, !noundef !29
  %14 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %15 = load i64, ptr %14, align 8, !noundef !29
  %16 = getelementptr inbounds nuw i8, ptr %4, i64 40
  %17 = load ptr, ptr %16, align 8, !invariant.load !29, !nonnull !29
  invoke void %17(ptr noalias noundef nonnull sret([24 x i8]) align 8 captures(address) dereferenceable(24) %0, ptr noundef nonnull align 1 %3, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %13, i64 noundef %15)
          to label %22 unwind label %18

18:                                               ; preds = %11
  %19 = landingpad { ptr, i32 }
          cleanup
  store i8 0, ptr %13, align 1
  %20 = icmp eq i64 %15, 0
  br i1 %20, label %25, label %21

21:                                               ; preds = %18
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %13, i64 noundef range(i64 1, 0) %15, i64 noundef 1) #37
  br label %25

22:                                               ; preds = %11
  store i8 0, ptr %13, align 1
  %23 = icmp eq i64 %15, 0
  br i1 %23, label %28, label %24

24:                                               ; preds = %22
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %13, i64 noundef range(i64 1, 0) %15, i64 noundef 1) #37
  br label %28

25:                                               ; preds = %21, %18
  %26 = load i64, ptr %6, align 8, !range !360, !noundef !29
  %27 = icmp eq i64 %26, -9223372036854775808
  br i1 %27, label %33, label %35

28:                                               ; preds = %24, %22, %9
  %29 = load i64, ptr %6, align 8, !range !360, !noundef !29
  %30 = icmp eq i64 %29, -9223372036854775808
  br i1 %30, label %42, label %36

31:                                               ; preds = %35
  %32 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

33:                                               ; preds = %37, %35, %25
  %34 = phi { ptr, i32 } [ %19, %35 ], [ %19, %25 ], [ %38, %37 ]
  resume { ptr, i32 } %34

35:                                               ; preds = %25
  invoke fastcc void @"_ZN4core3ptr105drop_in_place$LT$core..result..Result$LT$alloc..ffi..c_str..CString$C$alloc..ffi..c_str..NulError$GT$$GT$17h04a5a0d1226776dcE.915"(ptr noalias noundef align 8 dereferenceable(32) %6) #56
          to label %33 unwind label %31

36:                                               ; preds = %28
  invoke void @"_ZN70_$LT$alloc..vec..Vec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h17ee718c0c7a349dE"(ptr noalias noundef nonnull align 8 dereferenceable(32) %6)
          to label %41 unwind label %37

37:                                               ; preds = %36
  %38 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(32) %6)
          to label %33 unwind label %39

39:                                               ; preds = %37
  %40 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable

41:                                               ; preds = %36
  call void @"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"(ptr noalias noundef nonnull align 8 dereferenceable(32) %6)
  br label %42

42:                                               ; preds = %41, %28
  call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %6)
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN3std3sys12thread_local5guard3key6enable17h932fe352ce3e27e5E() unnamed_addr #20 {
  %1 = load atomic i64, ptr @_ZN3std3sys12thread_local5guard3key6enable5DTORS17h7430097ae0e77bf4E acquire, align 8
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = tail call noundef i64 @_ZN3std3sys12thread_local3key4racy7LazyKey9lazy_init17h1e6433018713260bE(ptr noundef nonnull align 8 @_ZN3std3sys12thread_local5guard3key6enable5DTORS17h7430097ae0e77bf4E)
  br label %5

5:                                                ; preds = %3, %0
  %6 = phi i64 [ %4, %3 ], [ %1, %0 ]
  %7 = trunc i64 %6 to i32
  %8 = tail call noundef i32 @pthread_setspecific(i32 noundef %7, ptr noundef nonnull inttoptr (i64 1 to ptr)) #37
  ret void
}

; Function Attrs: nounwind nonlazybind uwtable
declare noundef i32 @pthread_setspecific(i32 noundef, ptr noundef) unnamed_addr #22

; Function Attrs: nounwind nonlazybind uwtable
define hidden { i64, i64 } @_ZN3std3sys6thread4unix13current_os_id17h8f8ca4bd9475d4f6E() unnamed_addr #22 {
  %1 = icmp eq i64 ptrtoint (ptr @gettid to i64), 0
  br i1 %1, label %5, label %2

2:                                                ; preds = %0
  %3 = icmp ne ptr @gettid, null
  tail call void @llvm.assume(i1 %3)
  %4 = tail call noundef i32 @gettid() #37
  br label %8

5:                                                ; preds = %0
  %6 = tail call noundef i64 (i64, ...) @syscall(i64 noundef 186) #37
  %7 = trunc i64 %6 to i32
  br label %8

8:                                                ; preds = %5, %2
  %9 = phi i32 [ %4, %2 ], [ %7, %5 ]
  %10 = sext i32 %9 to i64
  %11 = insertvalue { i64, i64 } { i64 1, i64 poison }, i64 %10, 1
  ret { i64, i64 } %11
}

; Function Attrs: nounwind nonlazybind
declare extern_weak noundef i32 @gettid() unnamed_addr #51

; Function Attrs: nounwind nonlazybind uwtable
declare noundef range(i32 2, 9) i32 @rust_eh_personality(i32 noundef, i32 noundef, i64, ptr noundef, ptr noundef) unnamed_addr #22

; Function Attrs: nonlazybind uwtable
define hidden noundef i64 @_ZN3std3sys12thread_local3key4racy7LazyKey9lazy_init17h1e6433018713260bE(ptr noundef nonnull align 8 captures(none) %0) unnamed_addr #20 {
  %2 = alloca [0 x i8], align 1
  %3 = alloca [48 x i8], align 8
  %4 = alloca [4 x i8], align 4
  %5 = alloca [48 x i8], align 8
  %6 = alloca [4 x i8], align 4
  %7 = alloca [48 x i8], align 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = load ptr, ptr %8, align 8, !noundef !29
  call void @llvm.lifetime.start.p0(i64 0, ptr nonnull %2)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %6)
  store i32 0, ptr %6, align 4
  %10 = call noundef i32 @pthread_key_create(ptr noundef nonnull %6, ptr noundef %9) #37
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %18, label %12, !prof !361

12:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5)
  store ptr @anon.0bc126ce095015e3131941c1eb180bc1.20, ptr %5, align 8
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 1, ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %14, align 8
  %15 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %15, align 8
  %16 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 0, ptr %16, align 8
  %17 = call noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %2, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %5)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %5)
  call fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E.1161"(ptr %17)
  call void @_ZN3std7process5abort17hea5afc4016cb7294E() #53
  unreachable

18:                                               ; preds = %1
  %19 = load i32, ptr %6, align 4, !noundef !29
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %6)
  call void @llvm.lifetime.end.p0(i64 0, ptr nonnull %2)
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %21, label %41

21:                                               ; preds = %18
  %22 = load ptr, ptr %8, align 8, !noundef !29
  call void @llvm.lifetime.start.p0(i64 0, ptr nonnull %2)
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %4)
  store i32 0, ptr %4, align 4
  %23 = call noundef i32 @pthread_key_create(ptr noundef nonnull %4, ptr noundef %22) #37
  %24 = icmp eq i32 %23, 0
  br i1 %24, label %31, label %25, !prof !361

25:                                               ; preds = %21
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %3)
  store ptr @anon.0bc126ce095015e3131941c1eb180bc1.20, ptr %3, align 8
  %26 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 1, ptr %26, align 8
  %27 = getelementptr inbounds nuw i8, ptr %3, i64 32
  store ptr null, ptr %27, align 8
  %28 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %28, align 8
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i64 0, ptr %29, align 8
  %30 = call noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %2, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %3)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %3)
  call fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E.1161"(ptr %30)
  call void @_ZN3std7process5abort17hea5afc4016cb7294E() #53
  unreachable

31:                                               ; preds = %21
  %32 = load i32, ptr %4, align 4, !noundef !29
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %4)
  call void @llvm.lifetime.end.p0(i64 0, ptr nonnull %2)
  %33 = call noundef i32 @pthread_key_delete(i32 noundef 0) #37
  %34 = icmp eq i32 %32, 0
  br i1 %34, label %35, label %41, !prof !589

35:                                               ; preds = %31
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %7)
  store ptr @anon.0bc126ce095015e3131941c1eb180bc1.18, ptr %7, align 8
  %36 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i64 1, ptr %36, align 8
  %37 = getelementptr inbounds nuw i8, ptr %7, i64 32
  store ptr null, ptr %37, align 8
  %38 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %38, align 8
  %39 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store i64 0, ptr %39, align 8
  %40 = call noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %2, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %7)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %7)
  call fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E.1161"(ptr %40)
  call void @_ZN3std7process5abort17hea5afc4016cb7294E() #53
  unreachable

41:                                               ; preds = %31, %18
  %42 = phi i32 [ %32, %31 ], [ %19, %18 ]
  %43 = zext i32 %42 to i64
  %44 = cmpxchg ptr %0, i64 0, i64 %43 release acquire, align 8
  %45 = extractvalue { i64, i1 } %44, 1
  br i1 %45, label %49, label %46

46:                                               ; preds = %41
  %47 = extractvalue { i64, i1 } %44, 0
  %48 = call noundef i32 @pthread_key_delete(i32 noundef %42) #37
  br label %49

49:                                               ; preds = %46, %41
  %50 = phi i64 [ %47, %46 ], [ %43, %41 ]
  ret i64 %50
}

; Function Attrs: nounwind nonlazybind uwtable
declare noundef i32 @pthread_key_create(ptr noundef, ptr noundef) unnamed_addr #22

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E.1161"(ptr %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = ptrtoint ptr %0 to i64
  %3 = and i64 %2, 3
  %4 = icmp eq i64 %3, 1
  br i1 %4, label %6, label %5, !prof !477

5:                                                ; preds = %37, %1
  ret void

6:                                                ; preds = %1
  %7 = getelementptr i8, ptr %0, i64 -1
  %8 = icmp ne ptr %7, null
  tail call void @llvm.assume(i1 %8)
  %9 = load ptr, ptr %7, align 8
  %10 = getelementptr i8, ptr %0, i64 7
  %11 = load ptr, ptr %10, align 8, !nonnull !29, !align !410, !noundef !29
  %12 = load ptr, ptr %11, align 8, !invariant.load !29
  %13 = icmp eq ptr %12, null
  br i1 %13, label %16, label %14

14:                                               ; preds = %6
  %15 = icmp ne ptr %9, null
  tail call void @llvm.assume(i1 %15)
  invoke void %12(ptr noundef nonnull %9)
          to label %16 unwind label %26

16:                                               ; preds = %14, %6
  %17 = icmp ne ptr %9, null
  tail call void @llvm.assume(i1 %17)
  %18 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %19 = load i64, ptr %18, align 8, !range !354, !invariant.load !29
  %20 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %21 = load i64, ptr %20, align 8, !range !476, !invariant.load !29
  %22 = add i64 %21, -1
  %23 = icmp sgt i64 %22, -1
  tail call void @llvm.assume(i1 %23)
  %24 = icmp eq i64 %19, 0
  br i1 %24, label %37, label %25

25:                                               ; preds = %16
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %9, i64 noundef range(i64 1, 0) %19, i64 noundef range(i64 1, -9223372036854775807) %21) #37
  br label %37

26:                                               ; preds = %14
  %27 = landingpad { ptr, i32 }
          cleanup
  %28 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %29 = load i64, ptr %28, align 8, !range !354, !invariant.load !29
  %30 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %31 = load i64, ptr %30, align 8, !range !476, !invariant.load !29
  %32 = add i64 %31, -1
  %33 = icmp sgt i64 %32, -1
  tail call void @llvm.assume(i1 %33)
  %34 = icmp eq i64 %29, 0
  br i1 %34, label %36, label %35

35:                                               ; preds = %26
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %9, i64 noundef range(i64 1, 0) %29, i64 noundef range(i64 1, -9223372036854775807) %31) #37
  br label %36

36:                                               ; preds = %35, %26
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %7, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %27

37:                                               ; preds = %25, %16
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %7, i64 noundef 24, i64 noundef 8) #37
  br label %5
}

; Function Attrs: nounwind nonlazybind uwtable
declare noundef i32 @pthread_key_delete(i32 noundef) unnamed_addr #22

; Function Attrs: nonlazybind uwtable
declare hidden noundef zeroext i1 @"_ZN52_$LT$$RF$mut$u20$T$u20$as$u20$core..fmt..Display$GT$3fmt17h343b7c4136dd9f77E"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(16), ptr noalias noundef align 8 dereferenceable(24)) unnamed_addr #20

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN3std3sys12thread_local11destructors10linux_like8register17hebc998dae8cbee2eE(ptr noundef %0, ptr noundef nonnull %1) unnamed_addr #20 {
  %3 = icmp eq i64 ptrtoint (ptr @__cxa_thread_atexit_impl to i64), 0
  br i1 %3, label %7, label %4

4:                                                ; preds = %2
  %5 = icmp ne ptr @__cxa_thread_atexit_impl, null
  tail call void @llvm.assume(i1 %5)
  %6 = tail call noundef i32 @__cxa_thread_atexit_impl(ptr noundef nonnull %1, ptr noundef %0, ptr noundef nonnull @_rust_extern_with_linkage_c2be051742d3c8de___dso_handle) #37
  br label %8

7:                                                ; preds = %2
  tail call void @_ZN3std3sys12thread_local11destructors4list8register17h02d270f8dbbaaa35E(ptr noundef %0, ptr noundef nonnull %1)
  br label %8

8:                                                ; preds = %7, %4
  ret void
}

; Function Attrs: nounwind nonlazybind
declare extern_weak noundef i32 @__cxa_thread_atexit_impl(ptr noundef nonnull, ptr noundef, ptr noundef) unnamed_addr #51

; Function Attrs: nonlazybind uwtable
define hidden noundef zeroext i1 @_ZN3std3sys3pal4unix5futex10futex_wait17h26dd62589d2120e7E(ptr noundef nonnull align 4 %0, i32 noundef %1, i64 %2, i32 noundef range(i32 0, 1000000001) %3) unnamed_addr #20 personality ptr @rust_eh_personality {
  %5 = alloca [24 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %5)
  %6 = icmp eq i32 %3, 1000000000
  br i1 %6, label %33, label %7

7:                                                ; preds = %4
  %8 = tail call fastcc { i64, i32 } @_ZN3std3sys3pal4unix4time8Timespec3now17hf90ac21ba9802729E(i32 noundef 1)
  %9 = extractvalue { i64, i32 } %8, 0
  %10 = tail call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %9, i64 %2)
  %11 = extractvalue { i64, i1 } %10, 0
  %12 = extractvalue { i64, i1 } %10, 1
  %13 = icmp slt i64 %2, 0
  %14 = xor i1 %13, %12
  br i1 %14, label %33, label %15, !prof !353

15:                                               ; preds = %7
  %16 = extractvalue { i64, i32 } %8, 1
  %17 = add nuw nsw i32 %16, %3
  %18 = icmp samesign ugt i32 %17, 999999999
  br i1 %18, label %19, label %26

19:                                               ; preds = %15
  %20 = tail call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %11, i64 1)
  %21 = extractvalue { i64, i1 } %20, 1
  %22 = extractvalue { i64, i1 } %20, 0
  %23 = add nsw i32 %17, -1000000000
  %24 = icmp eq i32 %23, 1000000000
  %25 = select i1 %21, i1 true, i1 %24
  br i1 %25, label %33, label %26, !prof !355

26:                                               ; preds = %19, %15
  %27 = phi i64 [ %11, %15 ], [ %22, %19 ]
  %28 = phi i32 [ %17, %15 ], [ %23, %19 ]
  %29 = icmp ult i32 %28, 1000000000
  tail call void @llvm.assume(i1 %29)
  %30 = zext nneg i32 %28 to i64
  %31 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 %27, ptr %31, align 8
  %32 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i64 %30, ptr %32, align 8
  br label %33

33:                                               ; preds = %26, %19, %7, %4
  %34 = phi i64 [ 1, %26 ], [ 0, %4 ], [ 0, %7 ], [ 0, %19 ]
  store i64 %34, ptr %5, align 8
  %35 = getelementptr inbounds nuw i8, ptr %5, i64 8
  br label %36

36:                                               ; preds = %45, %33
  %37 = load atomic i32, ptr %0 monotonic, align 4
  %38 = icmp eq i32 %37, %1
  br i1 %38, label %39, label %48

39:                                               ; preds = %36
  %40 = load i64, ptr %5, align 8, !range !359, !noundef !29
  %41 = trunc nuw i64 %40 to i1
  %42 = select i1 %41, ptr %35, ptr null
  %43 = call noundef i64 (i64, ...) @syscall(i64 noundef 202, ptr noundef nonnull %0, i32 noundef 137, i32 noundef %1, ptr noundef %42, ptr noundef null, i32 noundef -1) #37
  %44 = icmp slt i64 %43, 0
  br i1 %44, label %45, label %48

45:                                               ; preds = %39
  %46 = tail call noundef ptr @__errno_location() #37
  %47 = load i32, ptr %46, align 4, !noundef !29
  switch i32 %47, label %48 [
    i32 110, label %49
    i32 4, label %36
  ]

48:                                               ; preds = %45, %39, %36
  br label %49

49:                                               ; preds = %48, %45
  %50 = phi i1 [ true, %48 ], [ false, %45 ]
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %5)
  ret i1 %50
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc { i64, i32 } @_ZN3std3sys3pal4unix4time8Timespec3now17hf90ac21ba9802729E(i32 noundef range(i32 0, 2) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = alloca [8 x i8], align 8
  %3 = alloca [8 x i8], align 8
  %4 = alloca [16 x i8], align 8
  %5 = alloca [16 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %5)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4)
  %6 = call noundef i32 @clock_gettime(i32 noundef %0, ptr noundef nonnull %5) #37
  call void @_ZN3std3sys3pal4unix3cvt17h4ef7b5dd8aa3bbb4E(ptr noalias noundef nonnull sret([16 x i8]) align 8 captures(address) dereferenceable(16) %4, i32 noundef %6)
  call void @llvm.experimental.noalias.scope.decl(metadata !590)
  %7 = load i32, ptr %4, align 8, !range !593, !alias.scope !590, !noundef !29
  %8 = trunc nuw i32 %7 to i1
  br i1 %8, label %9, label %19, !prof !353

9:                                                ; preds = %1
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2), !noalias !590
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %11 = load ptr, ptr %10, align 8, !alias.scope !590, !nonnull !29, !noundef !29
  store ptr %11, ptr %2, align 8, !noalias !590
  invoke void @_ZN4core6result13unwrap_failed17hf9aa201d733198edE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.20a33724a9b558c90ab53ce9f9d617b9.73, i64 noundef 43, ptr noundef nonnull align 1 %2, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(32) @anon.20a33724a9b558c90ab53ce9f9d617b9.72, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.20a33724a9b558c90ab53ce9f9d617b9.48) #53
          to label %14 unwind label %12, !noalias !590

12:                                               ; preds = %9
  %13 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17h6bf9f0fdc2ad5447E.1266"(ptr noalias noundef nonnull align 8 dereferenceable(8) %2) #56
          to label %17 unwind label %15, !noalias !590

14:                                               ; preds = %9
  unreachable

15:                                               ; preds = %12
  %16 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55, !noalias !590
  unreachable

17:                                               ; preds = %24, %12
  %18 = phi { ptr, i32 } [ %13, %12 ], [ %25, %24 ]
  resume { ptr, i32 } %18

19:                                               ; preds = %1
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %4)
  %20 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %21 = load i64, ptr %20, align 8
  %22 = icmp ugt i64 %21, 999999999
  br i1 %22, label %23, label %29, !prof !353

23:                                               ; preds = %19
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %3), !noalias !594
  store ptr @anon.20a33724a9b558c90ab53ce9f9d617b9.46, ptr %3, align 8, !noalias !594
  invoke void @_ZN4core6result13unwrap_failed17hf9aa201d733198edE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.20a33724a9b558c90ab53ce9f9d617b9.73, i64 noundef 43, ptr noundef nonnull align 1 %3, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(32) @anon.20a33724a9b558c90ab53ce9f9d617b9.72, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.20a33724a9b558c90ab53ce9f9d617b9.49) #53
          to label %26 unwind label %24, !noalias !594

24:                                               ; preds = %23
  %25 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17h6bf9f0fdc2ad5447E.1266"(ptr noalias noundef nonnull align 8 dereferenceable(8) %3) #56
          to label %17 unwind label %27, !noalias !594

26:                                               ; preds = %23
  unreachable

27:                                               ; preds = %24
  %28 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55, !noalias !594
  unreachable

29:                                               ; preds = %19
  %30 = load i64, ptr %5, align 8
  %31 = trunc nuw nsw i64 %21 to i32
  %32 = insertvalue { i64, i32 } poison, i64 %30, 0
  %33 = insertvalue { i64, i32 } %32, i32 %31, 1
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %5)
  ret { i64, i32 } %33
}

; Function Attrs: nounwind nonlazybind uwtable
declare noundef i32 @clock_gettime(i32 noundef, ptr noundef) unnamed_addr #22

; Function Attrs: nonlazybind uwtable
define hidden void @"_ZN4core3ptr42drop_in_place$LT$std..io..error..Error$GT$17h6bf9f0fdc2ad5447E.1266"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(8) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = load ptr, ptr %0, align 8, !nonnull !29, !noundef !29
  %3 = ptrtoint ptr %2 to i64
  %4 = and i64 %3, 3
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %38, !prof !559

6:                                                ; preds = %1
  %7 = getelementptr i8, ptr %2, i64 -1
  %8 = icmp ne ptr %7, null
  tail call void @llvm.assume(i1 %8)
  %9 = load ptr, ptr %7, align 8
  %10 = getelementptr i8, ptr %2, i64 7
  %11 = load ptr, ptr %10, align 8, !nonnull !29, !align !410, !noundef !29
  %12 = load ptr, ptr %11, align 8, !invariant.load !29
  %13 = icmp eq ptr %12, null
  br i1 %13, label %16, label %14

14:                                               ; preds = %6
  %15 = icmp ne ptr %9, null
  tail call void @llvm.assume(i1 %15)
  invoke void %12(ptr noundef nonnull %9)
          to label %16 unwind label %26

16:                                               ; preds = %14, %6
  %17 = icmp ne ptr %9, null
  tail call void @llvm.assume(i1 %17)
  %18 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %19 = load i64, ptr %18, align 8, !range !354, !invariant.load !29
  %20 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %21 = load i64, ptr %20, align 8, !range !476, !invariant.load !29
  %22 = add i64 %21, -1
  %23 = icmp sgt i64 %22, -1
  tail call void @llvm.assume(i1 %23)
  %24 = icmp eq i64 %19, 0
  br i1 %24, label %37, label %25

25:                                               ; preds = %16
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %9, i64 noundef range(i64 1, -9223372036854775808) %19, i64 noundef range(i64 1, -9223372036854775807) %21) #37
  br label %37

26:                                               ; preds = %14
  %27 = landingpad { ptr, i32 }
          cleanup
  %28 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %29 = load i64, ptr %28, align 8, !range !354, !invariant.load !29
  %30 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %31 = load i64, ptr %30, align 8, !range !476, !invariant.load !29
  %32 = add i64 %31, -1
  %33 = icmp sgt i64 %32, -1
  tail call void @llvm.assume(i1 %33)
  %34 = icmp eq i64 %29, 0
  br i1 %34, label %36, label %35

35:                                               ; preds = %26
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %9, i64 noundef range(i64 1, -9223372036854775808) %29, i64 noundef range(i64 1, -9223372036854775807) %31) #37
  br label %36

36:                                               ; preds = %35, %26
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %7, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %27

37:                                               ; preds = %25, %16
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %7, i64 noundef 24, i64 noundef 8) #37
  br label %38

38:                                               ; preds = %37, %1
  ret void
}

; Function Attrs: nounwind nonlazybind uwtable
define hidden noundef zeroext i1 @_ZN3std3sys3pal4unix5futex10futex_wake17hc3e243608aa5a740E(ptr noundef nonnull align 4 %0) unnamed_addr #22 {
  %2 = tail call noundef i64 (i64, ...) @syscall(i64 noundef 202, ptr noundef nonnull %0, i32 noundef 129, i32 noundef 1) #37
  %3 = icmp sgt i64 %2, 0
  ret i1 %3
}

; Function Attrs: nounwind nonlazybind uwtable
define hidden void @_ZN3std3sys3pal4unix5futex14futex_wake_all17h0970ed6199f67cb9E(ptr noundef nonnull align 4 %0) unnamed_addr #22 {
  %2 = tail call noundef i64 (i64, ...) @syscall(i64 noundef 202, ptr noundef nonnull %0, i32 noundef 129, i32 noundef 2147483647) #37
  ret void
}

; Function Attrs: cold nounwind nonlazybind uwtable
define hidden void @_ZN3std3sys4sync5mutex5futex5Mutex14lock_contended17hd6d171911655b783E(ptr noundef nonnull align 4 %0) unnamed_addr #17 personality ptr @rust_eh_personality {
  %2 = alloca [24 x i8], align 8
  %3 = load atomic i32, ptr %0 monotonic, align 4
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %5, label %12

5:                                                ; preds = %5, %1
  %6 = phi i32 [ %7, %5 ], [ 100, %1 ]
  tail call void @llvm.x86.sse2.pause() #37
  %7 = add nsw i32 %6, -1
  %8 = load atomic i32, ptr %0 monotonic, align 4
  %9 = icmp ne i32 %8, 1
  %10 = icmp eq i32 %7, 0
  %11 = select i1 %9, i1 true, i1 %10
  br i1 %11, label %12, label %5

12:                                               ; preds = %5, %1
  %13 = phi i32 [ %3, %1 ], [ %8, %5 ]
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %15, label %19

15:                                               ; preds = %12
  %16 = cmpxchg ptr %0, i32 0, i32 1 acquire monotonic, align 4
  %17 = extractvalue { i32, i1 } %16, 1
  %18 = extractvalue { i32, i1 } %16, 0
  br i1 %17, label %22, label %19

19:                                               ; preds = %15, %12
  %20 = phi i32 [ %13, %12 ], [ %18, %15 ]
  %21 = getelementptr inbounds nuw i8, ptr %2, i64 8
  br label %23

22:                                               ; preds = %26, %15
  ret void

23:                                               ; preds = %53, %19
  %24 = phi i32 [ %20, %19 ], [ %54, %53 ]
  %25 = icmp eq i32 %24, 2
  br i1 %25, label %29, label %26

26:                                               ; preds = %23
  %27 = atomicrmw xchg ptr %0, i32 2 acquire, align 4
  %28 = icmp eq i32 %27, 0
  br i1 %28, label %22, label %29

29:                                               ; preds = %26, %23
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %2)
  store i64 0, ptr %2, align 8
  br label %30

30:                                               ; preds = %39, %29
  %31 = load atomic i32, ptr %0 monotonic, align 4
  %32 = icmp eq i32 %31, 2
  br i1 %32, label %33, label %43

33:                                               ; preds = %30
  %34 = load i64, ptr %2, align 8, !range !359, !noundef !29
  %35 = trunc nuw i64 %34 to i1
  %36 = select i1 %35, ptr %21, ptr null
  %37 = call noundef i64 (i64, ...) @syscall(i64 noundef 202, ptr noundef nonnull align 4 %0, i32 noundef 137, i32 noundef 2, ptr noundef %36, ptr noundef null, i32 noundef -1) #37
  %38 = icmp slt i64 %37, 0
  br i1 %38, label %39, label %43

39:                                               ; preds = %33
  %40 = tail call noundef ptr @__errno_location() #37
  %41 = load i32, ptr %40, align 4, !noundef !29
  %42 = icmp eq i32 %41, 4
  br i1 %42, label %30, label %43

43:                                               ; preds = %39, %33, %30
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %2)
  %44 = load atomic i32, ptr %0 monotonic, align 4
  %45 = icmp eq i32 %44, 1
  br i1 %45, label %46, label %53

46:                                               ; preds = %46, %43
  %47 = phi i32 [ %48, %46 ], [ 100, %43 ]
  call void @llvm.x86.sse2.pause() #37
  %48 = add nsw i32 %47, -1
  %49 = load atomic i32, ptr %0 monotonic, align 4
  %50 = icmp ne i32 %49, 1
  %51 = icmp eq i32 %48, 0
  %52 = select i1 %50, i1 true, i1 %51
  br i1 %52, label %53, label %46

53:                                               ; preds = %46, %43
  %54 = phi i32 [ %44, %43 ], [ %49, %46 ]
  br label %23
}

; Function Attrs: cold nounwind nonlazybind uwtable
define hidden void @_ZN3std3sys4sync5mutex5futex5Mutex4wake17h98256de9aeb3cee5E(ptr noundef nonnull align 4 %0) unnamed_addr #17 {
  %2 = tail call noundef i64 (i64, ...) @syscall(i64 noundef 202, ptr noundef nonnull align 4 %0, i32 noundef 129, i32 noundef 1) #37
  ret void
}

; Function Attrs: cold noreturn nonlazybind uwtable
define hidden void @_ZN3std7process5abort17hea5afc4016cb7294E() unnamed_addr #25 {
  tail call void @_ZN3std3sys3pal4unix14abort_internal17hb7e3b71be536cdfcE() #53
  unreachable
}

; Function Attrs: nonlazybind uwtable
define hidden noundef ptr @_ZN3std2io17default_write_fmt17hb29cce0e5403eb27E(ptr noalias noundef nonnull align 1 %0, ptr dead_on_return noalias noundef readonly align 8 captures(address) dereferenceable(48) %1) unnamed_addr #20 personality ptr @rust_eh_personality {
  %3 = alloca [48 x i8], align 8
  %4 = alloca [16 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4)
  store ptr %0, ptr %4, align 8
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr null, ptr %5, align 8
  %6 = invoke noundef zeroext i1 @_ZN4core3fmt5write17h919175a03bb9497fE(ptr noundef nonnull align 1 %4, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.17, ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %1)
          to label %9 unwind label %7

7:                                                ; preds = %53, %2
  %8 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN4core3ptr93drop_in_place$LT$std..io..default_write_fmt..Adapter$LT$std..sys..stdio..unix..Stderr$GT$$GT$17h973b9303513bdf4bE"(ptr noalias noundef nonnull align 8 dereferenceable(16) %4) #56
          to label %47 unwind label %59

9:                                                ; preds = %2
  %10 = load ptr, ptr %5, align 8, !noundef !29
  br i1 %6, label %11, label %13

11:                                               ; preds = %9
  %12 = icmp eq ptr %10, null
  br i1 %12, label %53, label %51, !prof !353

13:                                               ; preds = %9
  %14 = ptrtoint ptr %10 to i64
  %15 = and i64 %14, 3
  %16 = icmp eq i64 %15, 1
  br i1 %16, label %17, label %51, !prof !477

17:                                               ; preds = %13
  %18 = getelementptr i8, ptr %10, i64 -1
  %19 = icmp ne ptr %18, null
  call void @llvm.assume(i1 %19)
  %20 = load ptr, ptr %18, align 8
  %21 = getelementptr i8, ptr %10, i64 7
  %22 = load ptr, ptr %21, align 8, !nonnull !29, !align !410, !noundef !29
  %23 = load ptr, ptr %22, align 8, !invariant.load !29
  %24 = icmp eq ptr %23, null
  br i1 %24, label %27, label %25

25:                                               ; preds = %17
  %26 = icmp ne ptr %20, null
  call void @llvm.assume(i1 %26)
  invoke void %23(ptr noundef nonnull %20)
          to label %27 unwind label %37

27:                                               ; preds = %25, %17
  %28 = icmp ne ptr %20, null
  call void @llvm.assume(i1 %28)
  %29 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %30 = load i64, ptr %29, align 8, !range !354, !invariant.load !29
  %31 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %32 = load i64, ptr %31, align 8, !range !476, !invariant.load !29
  %33 = add i64 %32, -1
  %34 = icmp sgt i64 %33, -1
  call void @llvm.assume(i1 %34)
  %35 = icmp eq i64 %30, 0
  br i1 %35, label %50, label %36

36:                                               ; preds = %27
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %20, i64 noundef range(i64 1, -9223372036854775808) %30, i64 noundef range(i64 1, -9223372036854775807) %32) #37
  br label %50

37:                                               ; preds = %25
  %38 = landingpad { ptr, i32 }
          cleanup
  %39 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %40 = load i64, ptr %39, align 8, !range !354, !invariant.load !29
  %41 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %42 = load i64, ptr %41, align 8, !range !476, !invariant.load !29
  %43 = add i64 %42, -1
  %44 = icmp sgt i64 %43, -1
  call void @llvm.assume(i1 %44)
  %45 = icmp eq i64 %40, 0
  br i1 %45, label %49, label %46

46:                                               ; preds = %37
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %20, i64 noundef range(i64 1, -9223372036854775808) %40, i64 noundef range(i64 1, -9223372036854775807) %42) #37
  br label %49

47:                                               ; preds = %49, %7
  %48 = phi { ptr, i32 } [ %38, %49 ], [ %8, %7 ]
  resume { ptr, i32 } %48

49:                                               ; preds = %46, %37
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %18, i64 noundef 24, i64 noundef 8) #37
  br label %47

50:                                               ; preds = %36, %27
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %18, i64 noundef 24, i64 noundef 8) #37
  br label %51

51:                                               ; preds = %50, %13, %11
  %52 = phi ptr [ %10, %11 ], [ null, %13 ], [ null, %50 ]
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %4)
  ret ptr %52

53:                                               ; preds = %11
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %3)
  store ptr @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.13, ptr %3, align 8
  %54 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 1, ptr %54, align 8
  %55 = getelementptr inbounds nuw i8, ptr %3, i64 32
  store ptr null, ptr %55, align 8
  %56 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %56, align 8
  %57 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i64 0, ptr %57, align 8
  invoke void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %3, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.14) #53
          to label %58 unwind label %7

58:                                               ; preds = %53
  unreachable

59:                                               ; preds = %7
  %60 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable
}

; Function Attrs: nonlazybind uwtable
define hidden void @"_ZN4core3ptr93drop_in_place$LT$std..io..default_write_fmt..Adapter$LT$std..sys..stdio..unix..Stderr$GT$$GT$17h973b9303513bdf4bE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(16) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8, !noundef !29
  %4 = ptrtoint ptr %3 to i64
  %5 = and i64 %4, 3
  %6 = icmp eq i64 %5, 1
  br i1 %6, label %7, label %39, !prof !477

7:                                                ; preds = %1
  %8 = getelementptr i8, ptr %3, i64 -1
  %9 = icmp ne ptr %8, null
  tail call void @llvm.assume(i1 %9)
  %10 = load ptr, ptr %8, align 8
  %11 = getelementptr i8, ptr %3, i64 7
  %12 = load ptr, ptr %11, align 8, !nonnull !29, !align !410, !noundef !29
  %13 = load ptr, ptr %12, align 8, !invariant.load !29
  %14 = icmp eq ptr %13, null
  br i1 %14, label %17, label %15

15:                                               ; preds = %7
  %16 = icmp ne ptr %10, null
  tail call void @llvm.assume(i1 %16)
  invoke void %13(ptr noundef nonnull %10)
          to label %17 unwind label %27

17:                                               ; preds = %15, %7
  %18 = icmp ne ptr %10, null
  tail call void @llvm.assume(i1 %18)
  %19 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %20 = load i64, ptr %19, align 8, !range !354, !invariant.load !29
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %22 = load i64, ptr %21, align 8, !range !476, !invariant.load !29
  %23 = add i64 %22, -1
  %24 = icmp sgt i64 %23, -1
  tail call void @llvm.assume(i1 %24)
  %25 = icmp eq i64 %20, 0
  br i1 %25, label %38, label %26

26:                                               ; preds = %17
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %10, i64 noundef range(i64 1, -9223372036854775808) %20, i64 noundef range(i64 1, -9223372036854775807) %22) #37
  br label %38

27:                                               ; preds = %15
  %28 = landingpad { ptr, i32 }
          cleanup
  %29 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %30 = load i64, ptr %29, align 8, !range !354, !invariant.load !29
  %31 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %32 = load i64, ptr %31, align 8, !range !476, !invariant.load !29
  %33 = add i64 %32, -1
  %34 = icmp sgt i64 %33, -1
  tail call void @llvm.assume(i1 %34)
  %35 = icmp eq i64 %30, 0
  br i1 %35, label %37, label %36

36:                                               ; preds = %27
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %10, i64 noundef range(i64 1, -9223372036854775808) %30, i64 noundef range(i64 1, -9223372036854775807) %32) #37
  br label %37

37:                                               ; preds = %36, %27
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %8, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %28

38:                                               ; preds = %26, %17
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %8, i64 noundef 24, i64 noundef 8) #37
  br label %39

39:                                               ; preds = %38, %1
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden noundef ptr @_ZN3std2io17default_write_fmt17hc990884f3953a138E(ptr noalias noundef align 8 dereferenceable(24) %0, ptr dead_on_return noalias noundef readonly align 8 captures(address) dereferenceable(48) %1) unnamed_addr #20 personality ptr @rust_eh_personality {
  %3 = alloca [48 x i8], align 8
  %4 = alloca [16 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %4)
  store ptr %0, ptr %4, align 8
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr null, ptr %5, align 8
  %6 = invoke noundef zeroext i1 @_ZN4core3fmt5write17h919175a03bb9497fE(ptr noundef nonnull align 1 %4, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.18, ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %1)
          to label %9 unwind label %7

7:                                                ; preds = %53, %2
  %8 = landingpad { ptr, i32 }
          cleanup
  invoke void @"_ZN4core3ptr119drop_in_place$LT$std..io..default_write_fmt..Adapter$LT$std..io..cursor..Cursor$LT$$RF$mut$u20$$u5b$u8$u5d$$GT$$GT$$GT$17h32bace49555b534cE"(ptr noalias noundef nonnull align 8 dereferenceable(16) %4) #56
          to label %47 unwind label %59

9:                                                ; preds = %2
  %10 = load ptr, ptr %5, align 8, !noundef !29
  br i1 %6, label %11, label %13

11:                                               ; preds = %9
  %12 = icmp eq ptr %10, null
  br i1 %12, label %53, label %51, !prof !353

13:                                               ; preds = %9
  %14 = ptrtoint ptr %10 to i64
  %15 = and i64 %14, 3
  %16 = icmp eq i64 %15, 1
  br i1 %16, label %17, label %51, !prof !477

17:                                               ; preds = %13
  %18 = getelementptr i8, ptr %10, i64 -1
  %19 = icmp ne ptr %18, null
  call void @llvm.assume(i1 %19)
  %20 = load ptr, ptr %18, align 8
  %21 = getelementptr i8, ptr %10, i64 7
  %22 = load ptr, ptr %21, align 8, !nonnull !29, !align !410, !noundef !29
  %23 = load ptr, ptr %22, align 8, !invariant.load !29
  %24 = icmp eq ptr %23, null
  br i1 %24, label %27, label %25

25:                                               ; preds = %17
  %26 = icmp ne ptr %20, null
  call void @llvm.assume(i1 %26)
  invoke void %23(ptr noundef nonnull %20)
          to label %27 unwind label %37

27:                                               ; preds = %25, %17
  %28 = icmp ne ptr %20, null
  call void @llvm.assume(i1 %28)
  %29 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %30 = load i64, ptr %29, align 8, !range !354, !invariant.load !29
  %31 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %32 = load i64, ptr %31, align 8, !range !476, !invariant.load !29
  %33 = add i64 %32, -1
  %34 = icmp sgt i64 %33, -1
  call void @llvm.assume(i1 %34)
  %35 = icmp eq i64 %30, 0
  br i1 %35, label %50, label %36

36:                                               ; preds = %27
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %20, i64 noundef range(i64 1, -9223372036854775808) %30, i64 noundef range(i64 1, -9223372036854775807) %32) #37
  br label %50

37:                                               ; preds = %25
  %38 = landingpad { ptr, i32 }
          cleanup
  %39 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %40 = load i64, ptr %39, align 8, !range !354, !invariant.load !29
  %41 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %42 = load i64, ptr %41, align 8, !range !476, !invariant.load !29
  %43 = add i64 %42, -1
  %44 = icmp sgt i64 %43, -1
  call void @llvm.assume(i1 %44)
  %45 = icmp eq i64 %40, 0
  br i1 %45, label %49, label %46

46:                                               ; preds = %37
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %20, i64 noundef range(i64 1, -9223372036854775808) %40, i64 noundef range(i64 1, -9223372036854775807) %42) #37
  br label %49

47:                                               ; preds = %49, %7
  %48 = phi { ptr, i32 } [ %38, %49 ], [ %8, %7 ]
  resume { ptr, i32 } %48

49:                                               ; preds = %46, %37
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %18, i64 noundef 24, i64 noundef 8) #37
  br label %47

50:                                               ; preds = %36, %27
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %18, i64 noundef 24, i64 noundef 8) #37
  br label %51

51:                                               ; preds = %50, %13, %11
  %52 = phi ptr [ %10, %11 ], [ null, %13 ], [ null, %50 ]
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %4)
  ret ptr %52

53:                                               ; preds = %11
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %3)
  store ptr @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.13, ptr %3, align 8
  %54 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 1, ptr %54, align 8
  %55 = getelementptr inbounds nuw i8, ptr %3, i64 32
  store ptr null, ptr %55, align 8
  %56 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %56, align 8
  %57 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i64 0, ptr %57, align 8
  invoke void @_ZN4core9panicking9panic_fmt17h3aea49fc48b5f252E(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(48) %3, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.14) #53
          to label %58 unwind label %7

58:                                               ; preds = %53
  unreachable

59:                                               ; preds = %7
  %60 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  call void @_ZN4core9panicking16panic_in_cleanup17ha70af596b440a548E() #55
  unreachable
}

; Function Attrs: nonlazybind uwtable
define hidden void @"_ZN4core3ptr119drop_in_place$LT$std..io..default_write_fmt..Adapter$LT$std..io..cursor..Cursor$LT$$RF$mut$u20$$u5b$u8$u5d$$GT$$GT$$GT$17h32bace49555b534cE"(ptr noalias noundef readonly align 8 captures(none) dereferenceable(16) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load ptr, ptr %2, align 8, !noundef !29
  %4 = ptrtoint ptr %3 to i64
  %5 = and i64 %4, 3
  %6 = icmp eq i64 %5, 1
  br i1 %6, label %7, label %39, !prof !477

7:                                                ; preds = %1
  %8 = getelementptr i8, ptr %3, i64 -1
  %9 = icmp ne ptr %8, null
  tail call void @llvm.assume(i1 %9)
  %10 = load ptr, ptr %8, align 8
  %11 = getelementptr i8, ptr %3, i64 7
  %12 = load ptr, ptr %11, align 8, !nonnull !29, !align !410, !noundef !29
  %13 = load ptr, ptr %12, align 8, !invariant.load !29
  %14 = icmp eq ptr %13, null
  br i1 %14, label %17, label %15

15:                                               ; preds = %7
  %16 = icmp ne ptr %10, null
  tail call void @llvm.assume(i1 %16)
  invoke void %13(ptr noundef nonnull %10)
          to label %17 unwind label %27

17:                                               ; preds = %15, %7
  %18 = icmp ne ptr %10, null
  tail call void @llvm.assume(i1 %18)
  %19 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %20 = load i64, ptr %19, align 8, !range !354, !invariant.load !29
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %22 = load i64, ptr %21, align 8, !range !476, !invariant.load !29
  %23 = add i64 %22, -1
  %24 = icmp sgt i64 %23, -1
  tail call void @llvm.assume(i1 %24)
  %25 = icmp eq i64 %20, 0
  br i1 %25, label %38, label %26

26:                                               ; preds = %17
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %10, i64 noundef range(i64 1, -9223372036854775808) %20, i64 noundef range(i64 1, -9223372036854775807) %22) #37
  br label %38

27:                                               ; preds = %15
  %28 = landingpad { ptr, i32 }
          cleanup
  %29 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %30 = load i64, ptr %29, align 8, !range !354, !invariant.load !29
  %31 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %32 = load i64, ptr %31, align 8, !range !476, !invariant.load !29
  %33 = add i64 %32, -1
  %34 = icmp sgt i64 %33, -1
  tail call void @llvm.assume(i1 %34)
  %35 = icmp eq i64 %30, 0
  br i1 %35, label %37, label %36

36:                                               ; preds = %27
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %10, i64 noundef range(i64 1, -9223372036854775808) %30, i64 noundef range(i64 1, -9223372036854775807) %32) #37
  br label %37

37:                                               ; preds = %36, %27
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %8, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %28

38:                                               ; preds = %26, %17
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %8, i64 noundef 24, i64 noundef 8) #37
  br label %39

39:                                               ; preds = %38, %1
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E.1432"(ptr %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = ptrtoint ptr %0 to i64
  %3 = and i64 %2, 3
  %4 = icmp eq i64 %3, 1
  br i1 %4, label %6, label %5, !prof !477

5:                                                ; preds = %37, %1
  ret void

6:                                                ; preds = %1
  %7 = getelementptr i8, ptr %0, i64 -1
  %8 = icmp ne ptr %7, null
  tail call void @llvm.assume(i1 %8)
  %9 = load ptr, ptr %7, align 8
  %10 = getelementptr i8, ptr %0, i64 7
  %11 = load ptr, ptr %10, align 8, !nonnull !29, !align !410, !noundef !29
  %12 = load ptr, ptr %11, align 8, !invariant.load !29
  %13 = icmp eq ptr %12, null
  br i1 %13, label %16, label %14

14:                                               ; preds = %6
  %15 = icmp ne ptr %9, null
  tail call void @llvm.assume(i1 %15)
  invoke void %12(ptr noundef nonnull %9)
          to label %16 unwind label %26

16:                                               ; preds = %14, %6
  %17 = icmp ne ptr %9, null
  tail call void @llvm.assume(i1 %17)
  %18 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %19 = load i64, ptr %18, align 8, !range !354, !invariant.load !29
  %20 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %21 = load i64, ptr %20, align 8, !range !476, !invariant.load !29
  %22 = add i64 %21, -1
  %23 = icmp sgt i64 %22, -1
  tail call void @llvm.assume(i1 %23)
  %24 = icmp eq i64 %19, 0
  br i1 %24, label %37, label %25

25:                                               ; preds = %16
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %9, i64 noundef range(i64 1, -9223372036854775808) %19, i64 noundef range(i64 1, -9223372036854775807) %21) #37
  br label %37

26:                                               ; preds = %14
  %27 = landingpad { ptr, i32 }
          cleanup
  %28 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %29 = load i64, ptr %28, align 8, !range !354, !invariant.load !29
  %30 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %31 = load i64, ptr %30, align 8, !range !476, !invariant.load !29
  %32 = add i64 %31, -1
  %33 = icmp sgt i64 %32, -1
  tail call void @llvm.assume(i1 %33)
  %34 = icmp eq i64 %29, 0
  br i1 %34, label %36, label %35

35:                                               ; preds = %26
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %9, i64 noundef range(i64 1, -9223372036854775808) %29, i64 noundef range(i64 1, -9223372036854775807) %31) #37
  br label %36

36:                                               ; preds = %35, %26
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %7, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %27

37:                                               ; preds = %25, %16
  tail call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %7, i64 noundef 24, i64 noundef 8) #37
  br label %5
}

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN3std3sys12thread_local11destructors4list8register17h02d270f8dbbaaa35E(ptr noundef %0, ptr noundef nonnull %1) unnamed_addr #20 personality ptr @rust_eh_personality {
  %3 = alloca [0 x i8], align 1
  %4 = alloca [48 x i8], align 8
  %5 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr @_ZN3std3sys12thread_local11destructors4list5DTORS17he82e25e5cbceeca8E)
  %6 = load i64, ptr %5, align 8, !noundef !29
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %8, label %10, !prof !361

8:                                                ; preds = %2
  store i64 -1, ptr %5, align 8
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 8
  invoke void @_ZN3std3sys12thread_local5guard3key6enable17h932fe352ce3e27e5E()
          to label %16 unwind label %30

10:                                               ; preds = %2
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %4)
  store ptr @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.39, ptr %4, align 8
  %11 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i64 1, ptr %11, align 8
  %12 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store ptr null, ptr %12, align 8
  %13 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr inttoptr (i64 8 to ptr), ptr %13, align 8
  %14 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i64 0, ptr %14, align 8
  %15 = call noundef ptr @_ZN3std2io5Write9write_fmt17h364726d8be2bd1deE(ptr noalias noundef nonnull align 1 %3, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %4)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %4)
  call fastcc void @"_ZN4core3ptr81drop_in_place$LT$core..result..Result$LT$$LP$$RP$$C$std..io..error..Error$GT$$GT$17h304d6db73f9e32f2E.1432"(ptr %15)
  call void @_ZN3std7process5abort17hea5afc4016cb7294E() #53
  unreachable

16:                                               ; preds = %8
  %17 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %18 = load i64, ptr %17, align 8, !alias.scope !597, !noundef !29
  %19 = load i64, ptr %9, align 8, !range !354, !alias.scope !597, !noundef !29
  %20 = icmp eq i64 %18, %19
  br i1 %20, label %21, label %22

21:                                               ; preds = %16
  invoke void @"_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17h48aa27d81eb31477E"(ptr noalias noundef nonnull align 8 dereferenceable(24) %9)
          to label %22 unwind label %30

22:                                               ; preds = %21, %16
  %23 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %24 = load ptr, ptr %23, align 8, !alias.scope !597, !nonnull !29, !noundef !29
  %25 = getelementptr inbounds nuw { ptr, ptr }, ptr %24, i64 %18
  store ptr %0, ptr %25, align 8
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 8
  store ptr %1, ptr %26, align 8
  %27 = add i64 %18, 1
  store i64 %27, ptr %17, align 8, !alias.scope !597
  %28 = load i64, ptr %5, align 8, !noundef !29
  %29 = add i64 %28, 1
  store i64 %29, ptr %5, align 8
  ret void

30:                                               ; preds = %21, %8
  %31 = landingpad { ptr, i32 }
          cleanup
  %32 = load i64, ptr %5, align 8, !noundef !29
  %33 = add i64 %32, 1
  store i64 %33, ptr %5, align 8
  resume { ptr, i32 } %31
}

; Function Attrs: nonlazybind uwtable
define hidden { i64, ptr } @_ZN3std3sys2fd4unix8FileDesc5write17h8d5efdf88bb3ba3aE(ptr noalias noundef readonly align 4 captures(none) dereferenceable(4) %0, ptr noalias noundef nonnull readonly align 1 captures(none) %1, i64 noundef %2) unnamed_addr #20 personality ptr @rust_eh_personality {
  %4 = load i32, ptr %0, align 4, !range !600, !noundef !29
  %5 = tail call noundef i64 @llvm.umin.i64(i64 %2, i64 9223372036854775807)
  %6 = tail call noundef i64 @write(i32 noundef %4, ptr noundef nonnull %1, i64 noundef %5) #37
  %7 = tail call { i64, ptr } @_ZN3std3sys3pal4unix3cvt17h2967415d3ce1c61bE(i64 noundef %6)
  %8 = extractvalue { i64, ptr } %7, 0
  %9 = and i64 %8, 1
  %10 = extractvalue { i64, ptr } %7, 1
  %11 = insertvalue { i64, ptr } poison, i64 %9, 0
  %12 = insertvalue { i64, ptr } %11, ptr %10, 1
  ret { i64, ptr } %12
}

; Function Attrs: nofree nounwind nonlazybind uwtable
declare noundef i64 @write(i32 noundef, ptr noundef readonly captures(none), i64 noundef) unnamed_addr #46

; Function Attrs: nonlazybind uwtable
define hidden void @_ZN3std6thread7current16try_with_current17h16830d05b1d122d7E(ptr dead_on_return noalias noundef readonly align 8 captures(none) dereferenceable(32) %0) unnamed_addr #20 personality ptr @rust_eh_personality {
  %2 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr @_ZN3std6thread7current7CURRENT17h856398c56bd7c7a7E)
  %3 = load ptr, ptr %2, align 8, !noundef !29
  %4 = icmp ugt ptr %3, inttoptr (i64 2 to ptr)
  br i1 %4, label %14, label %5

5:                                                ; preds = %1
  %6 = tail call noundef i64 @_ZN3std6thread11main_thread3get17h422e7de94df43145E(), !noalias !601
  %7 = icmp ne i64 %6, 0
  %8 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr @_ZN3std6thread7current2id2ID17hf976129d244513b0E)
  %9 = load i64, ptr %8, align 8, !noalias !601
  %10 = icmp eq i64 %9, %6
  %11 = select i1 %7, i1 %10, i1 false
  br i1 %11, label %13, label %12

12:                                               ; preds = %5
  tail call fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17hd1d43f5324e942b5E"(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(32) %0, ptr noalias noundef readonly align 1 captures(address, read_provenance) null, i64 undef), !noalias !605
  br label %30

13:                                               ; preds = %5
  tail call fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17hd1d43f5324e942b5E"(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(32) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.50, i64 4), !noalias !605
  br label %30

14:                                               ; preds = %1
  %15 = tail call noundef nonnull ptr @_ZN3std6thread6Thread8from_raw17h7c7925a72024cc22E(ptr noundef nonnull %3)
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 24
  %17 = load ptr, ptr %16, align 8, !noalias !606, !align !411, !noundef !29
  %18 = icmp eq ptr %17, null
  br i1 %18, label %23, label %19

19:                                               ; preds = %14
  %20 = getelementptr inbounds nuw i8, ptr %15, i64 32
  %21 = load i64, ptr %20, align 8, !alias.scope !610, !noalias !606, !noundef !29
  %22 = add i64 %21, -1
  tail call fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17hd1d43f5324e942b5E"(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(32) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %17, i64 %22), !noalias !613
  br label %30

23:                                               ; preds = %14
  %24 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %25 = load i64, ptr %24, align 8, !range !476, !noalias !606, !noundef !29
  %26 = tail call noundef i64 @_ZN3std6thread11main_thread3get17h422e7de94df43145E(), !noalias !606
  %27 = icmp eq i64 %25, %26
  br i1 %27, label %28, label %29

28:                                               ; preds = %23
  tail call fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17hd1d43f5324e942b5E"(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(32) %0, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.50, i64 4), !noalias !613
  br label %30

29:                                               ; preds = %23
  tail call fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17hd1d43f5324e942b5E"(ptr noalias noundef nonnull readonly align 8 captures(address) dereferenceable(32) %0, ptr noalias noundef readonly align 1 captures(address, read_provenance) null, i64 undef), !noalias !613
  br label %30

30:                                               ; preds = %29, %28, %19, %13, %12
  ret void
}

; Function Attrs: inlinehint nonlazybind uwtable
define hidden fastcc void @"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17hd1d43f5324e942b5E"(ptr dead_on_return noalias noundef nonnull readonly align 8 captures(none) dereferenceable(32) %0, ptr noalias noundef readonly align 1 captures(address, read_provenance) %1, i64 %2) unnamed_addr #52 personality ptr @rust_eh_personality {
  %4 = alloca [64 x i8], align 8
  %5 = alloca [48 x i8], align 8
  %6 = alloca [64 x i8], align 8
  %7 = alloca [48 x i8], align 8
  %8 = alloca [24 x i8], align 8
  %9 = alloca [512 x i8], align 1
  %10 = alloca [8 x i8], align 8
  %11 = alloca [16 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %11)
  %12 = icmp eq ptr %1, null
  %13 = select i1 %12, ptr @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.61, ptr %1
  %14 = select i1 %12, i64 9, i64 %2
  store ptr %13, ptr %11, align 8
  %15 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store i64 %14, ptr %15, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %10)
  %16 = tail call { i64, i64 } @_ZN3std3sys6thread4unix13current_os_id17h8f8ca4bd9475d4f6E()
  %17 = extractvalue { i64, i64 } %16, 0
  %18 = trunc nuw i64 %17 to i1
  br i1 %18, label %19, label %21

19:                                               ; preds = %3
  %20 = extractvalue { i64, i64 } %16, 1
  br label %27

21:                                               ; preds = %3
  %22 = tail call align 8 ptr @llvm.threadlocal.address.p0(ptr @_ZN3std6thread7current2id2ID17hf976129d244513b0E)
  %23 = load i64, ptr %22, align 8, !noundef !29
  %24 = icmp eq i64 %23, 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %21
  %26 = tail call noundef range(i64 1, 0) i64 @_ZN3std6thread8ThreadId3new17hba8a0cb27b9eece5E()
  store i64 %26, ptr %22, align 8
  br label %27

27:                                               ; preds = %25, %21, %19
  %28 = phi i64 [ %20, %19 ], [ %26, %25 ], [ %23, %21 ]
  store i64 %28, ptr %10, align 8
  call void @llvm.lifetime.start.p0(i64 512, ptr nonnull %9)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(512) %9, i8 0, i64 512, i1 false)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %8)
  store ptr %9, ptr %8, align 8
  %29 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store i64 512, ptr %29, align 8
  %30 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store i64 0, ptr %30, align 8
  %31 = load ptr, ptr %0, align 8, !nonnull !29, !align !410, !noundef !29
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %33 = load ptr, ptr %32, align 8, !nonnull !29, !align !410, !noundef !29
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %7), !noalias !614
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %6), !noalias !614
  store ptr %11, ptr %6, align 8, !noalias !614
  %34 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17hbbba634b9e74e25cE", ptr %34, align 8, !noalias !614
  %35 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store ptr %10, ptr %35, align 8, !noalias !614
  %36 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store ptr @"_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u64$GT$3fmt17hb8c6372145dbb21aE", ptr %36, align 8, !noalias !614
  %37 = getelementptr inbounds nuw i8, ptr %6, i64 32
  store ptr %31, ptr %37, align 8, !noalias !614
  %38 = getelementptr inbounds nuw i8, ptr %6, i64 40
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h05ab7ecd738f8002E", ptr %38, align 8, !noalias !614
  %39 = getelementptr inbounds nuw i8, ptr %6, i64 48
  store ptr %33, ptr %39, align 8, !noalias !614
  %40 = getelementptr inbounds nuw i8, ptr %6, i64 56
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17hbbba634b9e74e25cE", ptr %40, align 8, !noalias !614
  store ptr @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.69, ptr %7, align 8, !noalias !614
  %41 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i64 5, ptr %41, align 8, !noalias !614
  %42 = getelementptr inbounds nuw i8, ptr %7, i64 32
  store ptr null, ptr %42, align 8, !noalias !614
  %43 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store ptr %6, ptr %43, align 8, !noalias !614
  %44 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store i64 4, ptr %44, align 8, !noalias !614
  %45 = call noundef ptr @_ZN3std2io5Write9write_fmt17h7f43f95a9f0ae4faE(ptr noundef nonnull align 1 %8, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %7), !noalias !614
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %6), !noalias !614
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %7), !noalias !614
  %46 = icmp eq ptr %45, null
  br i1 %46, label %50, label %53

47:                                               ; preds = %181, %172, %139, %130, %86, %77
  %48 = phi ptr [ %58, %86 ], [ %58, %77 ], [ %111, %139 ], [ %111, %130 ], [ %153, %181 ], [ %153, %172 ]
  %49 = phi { ptr, i32 } [ %78, %86 ], [ %78, %77 ], [ %131, %139 ], [ %131, %130 ], [ %173, %181 ], [ %173, %172 ]
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %48, i64 noundef 24, i64 noundef 8) #37
  resume { ptr, i32 } %49

50:                                               ; preds = %27
  %51 = load i64, ptr %30, align 8, !noundef !29
  %52 = icmp ult i64 %51, 513
  br i1 %52, label %141, label %140, !prof !617

53:                                               ; preds = %27
  %54 = ptrtoint ptr %45 to i64
  %55 = and i64 %54, 3
  %56 = icmp eq i64 %55, 1
  br i1 %56, label %57, label %88, !prof !477

57:                                               ; preds = %53
  %58 = getelementptr i8, ptr %45, i64 -1
  %59 = icmp ne ptr %58, null
  call void @llvm.assume(i1 %59)
  %60 = load ptr, ptr %58, align 8
  %61 = getelementptr i8, ptr %45, i64 7
  %62 = load ptr, ptr %61, align 8, !nonnull !29, !align !410, !noundef !29
  %63 = load ptr, ptr %62, align 8, !invariant.load !29
  %64 = icmp eq ptr %63, null
  br i1 %64, label %67, label %65

65:                                               ; preds = %57
  %66 = icmp ne ptr %60, null
  call void @llvm.assume(i1 %66)
  invoke void %63(ptr noundef nonnull %60)
          to label %67 unwind label %77

67:                                               ; preds = %65, %57
  %68 = icmp ne ptr %60, null
  call void @llvm.assume(i1 %68)
  %69 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %70 = load i64, ptr %69, align 8, !range !354, !invariant.load !29
  %71 = getelementptr inbounds nuw i8, ptr %62, i64 16
  %72 = load i64, ptr %71, align 8, !range !476, !invariant.load !29
  %73 = add i64 %72, -1
  %74 = icmp sgt i64 %73, -1
  call void @llvm.assume(i1 %74)
  %75 = icmp eq i64 %70, 0
  br i1 %75, label %87, label %76

76:                                               ; preds = %67
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %60, i64 noundef range(i64 1, -9223372036854775808) %70, i64 noundef range(i64 1, -9223372036854775807) %72) #37
  br label %87

77:                                               ; preds = %65
  %78 = landingpad { ptr, i32 }
          cleanup
  %79 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %80 = load i64, ptr %79, align 8, !range !354, !invariant.load !29
  %81 = getelementptr inbounds nuw i8, ptr %62, i64 16
  %82 = load i64, ptr %81, align 8, !range !476, !invariant.load !29
  %83 = add i64 %82, -1
  %84 = icmp sgt i64 %83, -1
  call void @llvm.assume(i1 %84)
  %85 = icmp eq i64 %80, 0
  br i1 %85, label %47, label %86

86:                                               ; preds = %77
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %60, i64 noundef range(i64 1, -9223372036854775808) %80, i64 noundef range(i64 1, -9223372036854775807) %82) #37
  br label %47

87:                                               ; preds = %76, %67
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %58, i64 noundef 24, i64 noundef 8) #37
  br label %88

88:                                               ; preds = %87, %53
  %89 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %90 = load ptr, ptr %89, align 8, !nonnull !29, !align !411, !noundef !29
  %91 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %92 = load ptr, ptr %91, align 8, !nonnull !29, !align !410, !noundef !29
  %93 = getelementptr i8, ptr %92, i64 72
  %94 = load ptr, ptr %93, align 8
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %5), !noalias !618
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %4), !noalias !618
  store ptr %11, ptr %4, align 8, !noalias !618
  %95 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17hbbba634b9e74e25cE", ptr %95, align 8, !noalias !618
  %96 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %10, ptr %96, align 8, !noalias !618
  %97 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store ptr @"_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u64$GT$3fmt17hb8c6372145dbb21aE", ptr %97, align 8, !noalias !618
  %98 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store ptr %31, ptr %98, align 8, !noalias !618
  %99 = getelementptr inbounds nuw i8, ptr %4, i64 40
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17h05ab7ecd738f8002E", ptr %99, align 8, !noalias !618
  %100 = getelementptr inbounds nuw i8, ptr %4, i64 48
  store ptr %33, ptr %100, align 8, !noalias !618
  %101 = getelementptr inbounds nuw i8, ptr %4, i64 56
  store ptr @"_ZN44_$LT$$RF$T$u20$as$u20$core..fmt..Display$GT$3fmt17hbbba634b9e74e25cE", ptr %101, align 8, !noalias !618
  store ptr @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.69, ptr %5, align 8, !noalias !618
  %102 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 5, ptr %102, align 8, !noalias !618
  %103 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store ptr null, ptr %103, align 8, !noalias !618
  %104 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %4, ptr %104, align 8, !noalias !618
  %105 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i64 4, ptr %105, align 8, !noalias !618
  %106 = call noundef ptr %94(ptr noundef nonnull align 1 %90, ptr noalias noundef nonnull align 8 captures(address) dereferenceable(48) %5), !noalias !618
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %4), !noalias !618
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %5), !noalias !618
  %107 = ptrtoint ptr %106 to i64
  %108 = and i64 %107, 3
  %109 = icmp eq i64 %108, 1
  br i1 %109, label %110, label %184, !prof !477

110:                                              ; preds = %88
  %111 = getelementptr i8, ptr %106, i64 -1
  %112 = icmp ne ptr %111, null
  call void @llvm.assume(i1 %112)
  %113 = load ptr, ptr %111, align 8
  %114 = getelementptr i8, ptr %106, i64 7
  %115 = load ptr, ptr %114, align 8, !nonnull !29, !align !410, !noundef !29
  %116 = load ptr, ptr %115, align 8, !invariant.load !29
  %117 = icmp eq ptr %116, null
  br i1 %117, label %120, label %118

118:                                              ; preds = %110
  %119 = icmp ne ptr %113, null
  call void @llvm.assume(i1 %119)
  invoke void %116(ptr noundef nonnull %113)
          to label %120 unwind label %130

120:                                              ; preds = %118, %110
  %121 = icmp ne ptr %113, null
  call void @llvm.assume(i1 %121)
  %122 = getelementptr inbounds nuw i8, ptr %115, i64 8
  %123 = load i64, ptr %122, align 8, !range !354, !invariant.load !29
  %124 = getelementptr inbounds nuw i8, ptr %115, i64 16
  %125 = load i64, ptr %124, align 8, !range !476, !invariant.load !29
  %126 = add i64 %125, -1
  %127 = icmp sgt i64 %126, -1
  call void @llvm.assume(i1 %127)
  %128 = icmp eq i64 %123, 0
  br i1 %128, label %182, label %129

129:                                              ; preds = %120
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %113, i64 noundef range(i64 1, -9223372036854775808) %123, i64 noundef range(i64 1, -9223372036854775807) %125) #37
  br label %182

130:                                              ; preds = %118
  %131 = landingpad { ptr, i32 }
          cleanup
  %132 = getelementptr inbounds nuw i8, ptr %115, i64 8
  %133 = load i64, ptr %132, align 8, !range !354, !invariant.load !29
  %134 = getelementptr inbounds nuw i8, ptr %115, i64 16
  %135 = load i64, ptr %134, align 8, !range !476, !invariant.load !29
  %136 = add i64 %135, -1
  %137 = icmp sgt i64 %136, -1
  call void @llvm.assume(i1 %137)
  %138 = icmp eq i64 %133, 0
  br i1 %138, label %47, label %139

139:                                              ; preds = %130
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %113, i64 noundef range(i64 1, -9223372036854775808) %133, i64 noundef range(i64 1, -9223372036854775807) %135) #37
  br label %47

140:                                              ; preds = %50
  call void @_ZN4core5slice5index16slice_index_fail17hf7a05389aea37f33E(i64 noundef 0, i64 noundef %51, i64 noundef 512, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.63) #53
  unreachable

141:                                              ; preds = %50
  %142 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %143 = load ptr, ptr %142, align 8, !nonnull !29, !align !410, !noundef !29
  %144 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %145 = load ptr, ptr %144, align 8, !nonnull !29, !align !411, !noundef !29
  %146 = getelementptr inbounds nuw i8, ptr %143, i64 56
  %147 = load ptr, ptr %146, align 8, !invariant.load !29, !nonnull !29
  %148 = call noundef ptr %147(ptr noundef nonnull align 1 %145, ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) %9, i64 noundef %51)
  %149 = ptrtoint ptr %148 to i64
  %150 = and i64 %149, 3
  %151 = icmp eq i64 %150, 1
  br i1 %151, label %152, label %184, !prof !477

152:                                              ; preds = %141
  %153 = getelementptr i8, ptr %148, i64 -1
  %154 = icmp ne ptr %153, null
  call void @llvm.assume(i1 %154)
  %155 = load ptr, ptr %153, align 8
  %156 = getelementptr i8, ptr %148, i64 7
  %157 = load ptr, ptr %156, align 8, !nonnull !29, !align !410, !noundef !29
  %158 = load ptr, ptr %157, align 8, !invariant.load !29
  %159 = icmp eq ptr %158, null
  br i1 %159, label %162, label %160

160:                                              ; preds = %152
  %161 = icmp ne ptr %155, null
  call void @llvm.assume(i1 %161)
  invoke void %158(ptr noundef nonnull %155)
          to label %162 unwind label %172

162:                                              ; preds = %160, %152
  %163 = icmp ne ptr %155, null
  call void @llvm.assume(i1 %163)
  %164 = getelementptr inbounds nuw i8, ptr %157, i64 8
  %165 = load i64, ptr %164, align 8, !range !354, !invariant.load !29
  %166 = getelementptr inbounds nuw i8, ptr %157, i64 16
  %167 = load i64, ptr %166, align 8, !range !476, !invariant.load !29
  %168 = add i64 %167, -1
  %169 = icmp sgt i64 %168, -1
  call void @llvm.assume(i1 %169)
  %170 = icmp eq i64 %165, 0
  br i1 %170, label %182, label %171

171:                                              ; preds = %162
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %155, i64 noundef range(i64 1, -9223372036854775808) %165, i64 noundef range(i64 1, -9223372036854775807) %167) #37
  br label %182

172:                                              ; preds = %160
  %173 = landingpad { ptr, i32 }
          cleanup
  %174 = getelementptr inbounds nuw i8, ptr %157, i64 8
  %175 = load i64, ptr %174, align 8, !range !354, !invariant.load !29
  %176 = getelementptr inbounds nuw i8, ptr %157, i64 16
  %177 = load i64, ptr %176, align 8, !range !476, !invariant.load !29
  %178 = add i64 %177, -1
  %179 = icmp sgt i64 %178, -1
  call void @llvm.assume(i1 %179)
  %180 = icmp eq i64 %175, 0
  br i1 %180, label %47, label %181

181:                                              ; preds = %172
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %155, i64 noundef range(i64 1, -9223372036854775808) %175, i64 noundef range(i64 1, -9223372036854775807) %177) #37
  br label %47

182:                                              ; preds = %171, %162, %129, %120
  %183 = phi ptr [ %111, %129 ], [ %111, %120 ], [ %153, %171 ], [ %153, %162 ]
  call void @_RNvCs1QLEhZ2QfLZ_7___rustc14___rust_dealloc(ptr noundef nonnull %183, i64 noundef 24, i64 noundef 8) #37
  br label %184

184:                                              ; preds = %182, %141, %88
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %8)
  call void @llvm.lifetime.end.p0(i64 512, ptr nonnull %9)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %10)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %11)
  ret void
}

; Function Attrs: nonlazybind uwtable
define hidden noundef range(i32 0, -1) i32 @"_ZN76_$LT$std..sys..fd..unix..FileDesc$u20$as$u20$std..os..fd..raw..FromRawFd$GT$11from_raw_fd17h3ef52e8d28c3d9ffE"(i32 noundef returned %0) unnamed_addr #20 {
  %2 = icmp eq i32 %0, -1
  br i1 %2, label %3, label %4, !prof !621

3:                                                ; preds = %1
  tail call void @_ZN4core6option13expect_failed17hb93f3b2511d2da6dE(ptr noalias noundef nonnull readonly align 1 captures(address, read_provenance) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.40, i64 noundef 8, ptr noalias noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @anon.fa0b702a440d3cf6c6d0f877f47e0aa3.93) #53
  unreachable

4:                                                ; preds = %1
  ret i32 %0
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #6 = { inlinehint nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #7 = { nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #8 = { allockind("alloc,uninitialized,aligned") allocsize(0) uwtable "alloc-family"="__rust_alloc" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #9 = { allockind("free") uwtable "alloc-family"="__rust_alloc" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #10 = { allockind("realloc,aligned") allocsize(3) uwtable "alloc-family"="__rust_alloc" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #11 = { allockind("alloc,zeroed,aligned") allocsize(0) uwtable "alloc-family"="__rust_alloc" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #12 = { noreturn uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #13 = { uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #14 = { nofree nounwind nonlazybind willreturn memory(argmem: read) }
attributes #15 = { inlinehint nounwind nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #16 = { cold nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #17 = { cold nounwind nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #18 = { cold minsize noreturn nonlazybind optsize uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #19 = { cold noinline noreturn nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #20 = { nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #21 = { noinline nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #22 = { nounwind nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #23 = { mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #24 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #25 = { cold noreturn nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #26 = { mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(inaccessiblemem: write) uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #27 = { cold noinline noreturn nounwind nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #28 = { cold minsize noinline noreturn nounwind nonlazybind optsize uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #29 = { cold minsize noinline noreturn nonlazybind optsize uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #30 = { noreturn nounwind nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #31 = { noreturn nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #32 = { noinline noreturn nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #33 = { inlinehint noreturn nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #34 = { minsize noreturn nonlazybind optsize uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #35 = { minsize nonlazybind optsize uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #36 = { inlinehint minsize nonlazybind optsize uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #37 = { nounwind }
attributes #38 = { nofree nosync nounwind nonlazybind memory(none) uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #39 = { mustprogress nounwind nonlazybind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable "alloc-family"="malloc" "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #40 = { cold nofree noreturn nounwind nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #41 = { nofree nosync nounwind nonlazybind memory(read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #42 = { nofree nosync nounwind nonlazybind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #43 = { mustprogress nofree norecurse nosync nounwind nonlazybind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #44 = { cold noinline nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #45 = { mustprogress nofree norecurse nounwind nonlazybind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #46 = { nofree nounwind nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #47 = { mustprogress nofree nounwind nonlazybind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) uwtable "alloc-family"="malloc" "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #48 = { mustprogress nounwind nonlazybind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #49 = { mustprogress nounwind nonlazybind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable "alloc-family"="malloc" "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #50 = { mustprogress nofree nounwind nonlazybind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) uwtable "alloc-family"="malloc" "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #51 = { nounwind nonlazybind }
attributes #52 = { inlinehint nonlazybind uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #53 = { noreturn }
attributes #54 = { noreturn nounwind }
attributes #55 = { cold noreturn nounwind }
attributes #56 = { cold }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4}
!llvm.dbg.cu = !{!5, !7, !9, !11}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"RtLibUseGOT", i32 1}
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"rustc version 1.92.0-nightly (af7587183 2025-10-05)"}
!5 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !6, producer: "clang LLVM (rustc version 1.92.0-nightly (af7587183 2025-10-05))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!6 = !DIFile(filename: "src/lib.rs/@/4810t8lr4rp55i6jcy0q6mzvw", directory: "/home/manuel/prog/Enzyme/enzyme/benchmarks/ReverseMode/gmm")
!7 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !8, producer: "clang LLVM (rustc version 1.92.0-nightly (af7587183 2025-10-05))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!8 = !DIFile(filename: "src/lib.rs/@/93ij7lcuq962vkzw7o7acawys", directory: "/home/manuel/prog/Enzyme/enzyme/benchmarks/ReverseMode/gmm")
!9 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !10, producer: "clang LLVM (rustc version 1.92.0-nightly (af7587183 2025-10-05))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!10 = !DIFile(filename: "src/lib.rs/@/akxoswfu8ge4rrwcz84cff1xb", directory: "/home/manuel/prog/Enzyme/enzyme/benchmarks/ReverseMode/gmm")
!11 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !12, producer: "clang LLVM (rustc version 1.92.0-nightly (af7587183 2025-10-05))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!12 = !DIFile(filename: "src/lib.rs/@/d997a5lvk99kg9l3nywk6xp1m", directory: "/home/manuel/prog/Enzyme/enzyme/benchmarks/ReverseMode/gmm")
!13 = distinct !DISubprogram(name: "next<usize>", linkageName: "_ZN4core4iter5range101_$LT$impl$u20$core..iter..traits..iterator..Iterator$u20$for$u20$core..ops..range..Range$LT$A$GT$$GT$4next17h566d9e10f4c48efbE", scope: !15, file: !14, line: 849, type: !19, scopeLine: 849, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, templateParams: !50, retainedNodes: !48)
!14 = !DIFile(filename: "/home/manuel/prog/rust2/library/core/src/iter/range.rs", directory: "", checksumkind: CSK_MD5, checksum: "bd5e2cda5ef8f5ce87ca9ba8425e7413")
!15 = !DINamespace(name: "{impl#6}", scope: !16)
!16 = !DINamespace(name: "range", scope: !17)
!17 = !DINamespace(name: "iter", scope: !18)
!18 = !DINamespace(name: "core", scope: null)
!19 = !DISubroutineType(types: !20)
!20 = !{!21, !39}
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Option<usize>", scope: !23, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !24, templateParams: !29, identifier: "9c67f04a5cd6b79a2807e8012810e193")
!22 = !DIFile(filename: "<unknown>", directory: "")
!23 = !DINamespace(name: "option", scope: !18)
!24 = !{!25}
!25 = distinct !DICompositeType(tag: DW_TAG_variant_part, scope: !21, file: !22, size: 128, align: 64, elements: !26, templateParams: !29, identifier: "e69f74061023f332c1ecace61157a401", discriminator: !37)
!26 = !{!27, !33}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "None", scope: !25, file: !22, baseType: !28, size: 128, align: 64, extraData: i64 0)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "None", scope: !21, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !29, templateParams: !30, identifier: "67d8c48743f2f1bfc0ab004e79f8ced6")
!29 = !{}
!30 = !{!31}
!31 = !DITemplateTypeParameter(name: "T", type: !32)
!32 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "Some", scope: !25, file: !22, baseType: !34, size: 128, align: 64, extraData: i64 1)
!34 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Some", scope: !21, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !35, templateParams: !30, identifier: "bb2a329052f6b4d5bd63e8b99b80c58")
!35 = !{!36}
!36 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !34, file: !22, baseType: !32, size: 64, align: 64, offset: 64, flags: DIFlagPublic)
!37 = !DIDerivedType(tag: DW_TAG_member, scope: !21, file: !22, baseType: !38, size: 64, align: 64, flags: DIFlagArtificial)
!38 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&mut core::ops::range::Range<usize>", baseType: !40, size: 64, align: 64, dwarfAddressSpace: 0)
!40 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Range<usize>", scope: !41, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !43, templateParams: !46, identifier: "c46b108006d628f8b319841afeb1307")
!41 = !DINamespace(name: "range", scope: !42)
!42 = !DINamespace(name: "ops", scope: !18)
!43 = !{!44, !45}
!44 = !DIDerivedType(tag: DW_TAG_member, name: "start", scope: !40, file: !22, baseType: !32, size: 64, align: 64, flags: DIFlagPublic)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "end", scope: !40, file: !22, baseType: !32, size: 64, align: 64, offset: 64, flags: DIFlagPublic)
!46 = !{!47}
!47 = !DITemplateTypeParameter(name: "Idx", type: !32)
!48 = !{!49}
!49 = !DILocalVariable(name: "self", arg: 1, scope: !13, file: !14, line: 849, type: !39)
!50 = !{!51}
!51 = !DITemplateTypeParameter(name: "A", type: !32)
!52 = !DILocation(line: 849, column: 13, scope: !13)
!53 = !DILocation(line: 850, column: 14, scope: !13)
!54 = !DILocation(line: 851, column: 6, scope: !13)
!55 = distinct !DISubprogram(name: "spec_next<usize>", linkageName: "_ZN89_$LT$core..ops..range..Range$LT$T$GT$$u20$as$u20$core..iter..range..RangeIteratorImpl$GT$9spec_next17h4dfe42ce4e69327cE", scope: !56, file: !14, line: 764, type: !19, scopeLine: 764, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, templateParams: !30, retainedNodes: !57)
!56 = !DINamespace(name: "{impl#5}", scope: !16)
!57 = !{!58, !59}
!58 = !DILocalVariable(name: "self", arg: 1, scope: !55, file: !14, line: 764, type: !39)
!59 = !DILocalVariable(name: "old", scope: !60, file: !14, line: 766, type: !32, align: 64)
!60 = distinct !DILexicalBlock(scope: !55, file: !14, line: 766, column: 13)
!61 = !DILocation(line: 764, column: 18, scope: !55)
!62 = !DILocation(line: 765, column: 25, scope: !55)
!63 = !DILocalVariable(name: "self", arg: 1, scope: !64, file: !65, line: 1903, type: !72)
!64 = distinct !DISubprogram(name: "lt", linkageName: "_ZN4core3cmp5impls57_$LT$impl$u20$core..cmp..PartialOrd$u20$for$u20$usize$GT$2lt17h62a90545b12adf42E", scope: !66, file: !65, line: 1903, type: !69, scopeLine: 1903, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, templateParams: !29, retainedNodes: !73)
!65 = !DIFile(filename: "/home/manuel/prog/rust2/library/core/src/cmp.rs", directory: "", checksumkind: CSK_MD5, checksum: "2ebed4d982e1934df4c432f70a016f34")
!66 = !DINamespace(name: "{impl#58}", scope: !67)
!67 = !DINamespace(name: "impls", scope: !68)
!68 = !DINamespace(name: "cmp", scope: !18)
!69 = !DISubroutineType(types: !70)
!70 = !{!71, !72, !72}
!71 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!72 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&usize", baseType: !32, size: 64, align: 64, dwarfAddressSpace: 0)
!73 = !{!63, !74}
!74 = !DILocalVariable(name: "other", arg: 2, scope: !64, file: !65, line: 1903, type: !72)
!75 = !DILocation(line: 1903, column: 19, scope: !64, inlinedAt: !76)
!76 = distinct !DILocation(line: 765, column: 12, scope: !55)
!77 = !DILocation(line: 1903, column: 26, scope: !64, inlinedAt: !76)
!78 = !DILocation(line: 1903, column: 50, scope: !64, inlinedAt: !76)
!79 = !DILocation(line: 1903, column: 59, scope: !64, inlinedAt: !76)
!80 = !DILocation(line: 765, column: 12, scope: !55)
!81 = !DILocation(line: 771, column: 13, scope: !55)
!82 = !DILocation(line: 765, column: 9, scope: !55)
!83 = !DILocation(line: 766, column: 23, scope: !55)
!84 = !DILocation(line: 766, column: 17, scope: !60)
!85 = !DILocation(line: 768, column: 35, scope: !60)
!86 = !DILocation(line: 768, column: 13, scope: !60)
!87 = !DILocation(line: 769, column: 13, scope: !60)
!88 = !DILocation(line: 773, column: 6, scope: !55)
!89 = distinct !DISubprogram(name: "into_iter<core::ops::range::Range<usize>>", linkageName: "_ZN63_$LT$I$u20$as$u20$core..iter..traits..collect..IntoIterator$GT$9into_iter17h35b862536a052749E", scope: !91, file: !90, line: 319, type: !94, scopeLine: 319, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, templateParams: !98, retainedNodes: !96)
!90 = !DIFile(filename: "/home/manuel/prog/rust2/library/core/src/iter/traits/collect.rs", directory: "", checksumkind: CSK_MD5, checksum: "65a297fadeea84104fff966a2f9a30dc")
!91 = !DINamespace(name: "{impl#0}", scope: !92)
!92 = !DINamespace(name: "collect", scope: !93)
!93 = !DINamespace(name: "traits", scope: !17)
!94 = !DISubroutineType(types: !95)
!95 = !{!40, !40}
!96 = !{!97}
!97 = !DILocalVariable(name: "self", arg: 1, scope: !89, file: !90, line: 319, type: !40)
!98 = !{!99}
!99 = !DITemplateTypeParameter(name: "I", type: !40)
!100 = !DILocation(line: 319, column: 18, scope: !89)
!101 = !DILocation(line: 321, column: 6, scope: !89)
!102 = distinct !DISubprogram(name: "dgmm_objective", linkageName: "_ZN5gmmrs4safe14dgmm_objective17hd5def2d68203fb76E", scope: !104, file: !103, line: 15, type: !106, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !7, templateParams: !29, retainedNodes: !114)
!103 = !DIFile(filename: "src/safe.rs", directory: "/home/manuel/prog/Enzyme/enzyme/benchmarks/ReverseMode/gmm", checksumkind: CSK_MD5, checksum: "cd35c3119ba218b21d956af10db5ce2e")
!104 = !DINamespace(name: "safe", scope: !105)
!105 = !DINamespace(name: "gmmrs", scope: null)
!106 = !DISubroutineType(types: !107)
!107 = !{null, !108, !108}
!108 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "&mut [f64]", file: !22, size: 128, align: 64, elements: !109, templateParams: !29, identifier: "20ddfcfce50bf865828153ff1ef46b96")
!109 = !{!110, !113}
!110 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !108, file: !22, baseType: !111, size: 64, align: 64)
!111 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !112, size: 64, align: 64, dwarfAddressSpace: 0)
!112 = !DIBasicType(name: "f64", size: 64, encoding: DW_ATE_float)
!113 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !108, file: !22, baseType: !32, size: 64, align: 64, offset: 64)
!114 = !{!115, !116}
!115 = !DILocalVariable(name: "main_term", arg: 1, scope: !102, file: !103, line: 20, type: !108)
!116 = !DILocalVariable(name: "dmain_term_0", arg: 2, scope: !102, file: !103, line: 20, type: !108)
!117 = !DILocation(line: 20, column: 5, scope: !102)
!118 = !DILocation(line: 15, column: 1, scope: !102)
!119 = !DILocation(line: 18, column: 3, scope: !102)
!120 = distinct !DISubprogram(name: "gmm_objective", linkageName: "_ZN5gmmrs4safe13gmm_objective17h849a4e1a48dbbe36E", scope: !104, file: !103, line: 19, type: !121, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !7, templateParams: !29, retainedNodes: !123)
!121 = !DISubroutineType(types: !122)
!122 = !{null, !108}
!123 = !{!124, !125, !127, !129}
!124 = !DILocalVariable(name: "main_term", arg: 1, scope: !120, file: !103, line: 20, type: !108)
!125 = !DILocalVariable(name: "slse", scope: !126, file: !103, line: 22, type: !112, align: 64)
!126 = distinct !DILexicalBlock(scope: !120, file: !103, line: 22, column: 5)
!127 = !DILocalVariable(name: "iter", scope: !128, file: !103, line: 23, type: !40, align: 64)
!128 = distinct !DILexicalBlock(scope: !126, file: !103, line: 23, column: 5)
!129 = !DILocalVariable(name: "ik", scope: !130, file: !103, line: 23, type: !32, align: 64)
!130 = distinct !DILexicalBlock(scope: !128, file: !103, line: 23, column: 26)
!131 = !DILocation(line: 20, column: 5, scope: !120)
!132 = !DILocation(line: 22, column: 9, scope: !126)
!133 = !DILocation(line: 23, column: 15, scope: !128)
!134 = !DILocation(line: 22, column: 20, scope: !120)
!135 = !DILocation(line: 23, column: 15, scope: !126)
!136 = !DILocation(line: 23, column: 5, scope: !128)
!137 = !DILocation(line: 23, column: 9, scope: !128)
!138 = !DILocation(line: 23, column: 9, scope: !130)
!139 = !DILocation(line: 24, column: 25, scope: !130)
!140 = !DILocation(line: 26, column: 12, scope: !126)
!141 = !DILocation(line: 26, column: 5, scope: !126)
!142 = !DILocation(line: 27, column: 2, scope: !120)
!143 = !DILocation(line: 24, column: 9, scope: !130)
!144 = distinct !DISubprogram(name: "precondition_check", linkageName: "_ZN4core3num23_$LT$impl$u20$usize$GT$13unchecked_add18precondition_check17h5021e0cd12831d11E", scope: !146, file: !145, line: 68, type: !149, scopeLine: 68, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !9, templateParams: !29, retainedNodes: !182)
!145 = !DIFile(filename: "/home/manuel/prog/rust2/library/core/src/ub_checks.rs", directory: "", checksumkind: CSK_MD5, checksum: "41b3943b2b7dc8c218ee37ead81b317d")
!146 = !DINamespace(name: "unchecked_add", scope: !147)
!147 = !DINamespace(name: "{impl#11}", scope: !148)
!148 = !DINamespace(name: "num", scope: !18)
!149 = !DISubroutineType(types: !150)
!150 = !{null, !32, !32, !151}
!151 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&core::panic::location::Location", baseType: !152, size: 64, align: 64, dwarfAddressSpace: 0)
!152 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Location", scope: !153, file: !22, size: 192, align: 64, flags: DIFlagPublic, elements: !155, templateParams: !29, identifier: "df5f75f9d369fff02c3726ce692452e1")
!153 = !DINamespace(name: "location", scope: !154)
!154 = !DINamespace(name: "panic", scope: !18)
!155 = !{!156, !170, !172, !173}
!156 = !DIDerivedType(tag: DW_TAG_member, name: "filename", scope: !152, file: !22, baseType: !157, size: 128, align: 64, flags: DIFlagPrivate)
!157 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "NonNull<str>", scope: !158, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !160, templateParams: !168, identifier: "f2cefde0fc0863a642bc959022a0fba")
!158 = !DINamespace(name: "non_null", scope: !159)
!159 = !DINamespace(name: "ptr", scope: !18)
!160 = !{!161}
!161 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !157, file: !22, baseType: !162, size: 128, align: 64, flags: DIFlagPrivate)
!162 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "*const str", file: !22, size: 128, align: 64, elements: !163, templateParams: !29, identifier: "238a44609877474087c05adf26cd41fa")
!163 = !{!164, !167}
!164 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !162, file: !22, baseType: !165, size: 64, align: 64)
!165 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !166, size: 64, align: 64, dwarfAddressSpace: 0)
!166 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!167 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !162, file: !22, baseType: !32, size: 64, align: 64, offset: 64)
!168 = !{!169}
!169 = !DITemplateTypeParameter(name: "T", type: !166)
!170 = !DIDerivedType(tag: DW_TAG_member, name: "line", scope: !152, file: !22, baseType: !171, size: 32, align: 32, offset: 128, flags: DIFlagPrivate)
!171 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!172 = !DIDerivedType(tag: DW_TAG_member, name: "col", scope: !152, file: !22, baseType: !171, size: 32, align: 32, offset: 160, flags: DIFlagPrivate)
!173 = !DIDerivedType(tag: DW_TAG_member, name: "_filename", scope: !152, file: !22, baseType: !174, align: 8, offset: 192, flags: DIFlagPrivate)
!174 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "PhantomData<&str>", scope: !175, file: !22, align: 8, flags: DIFlagPublic, elements: !29, templateParams: !176, identifier: "8ab19967ffc9d0b4dfd90e4c8ba5b673")
!175 = !DINamespace(name: "marker", scope: !18)
!176 = !{!177}
!177 = !DITemplateTypeParameter(name: "T", type: !178)
!178 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "&str", file: !22, size: 128, align: 64, elements: !179, templateParams: !29, identifier: "9277eecd40495f85161460476aacc992")
!179 = !{!180, !181}
!180 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !178, file: !22, baseType: !165, size: 64, align: 64)
!181 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !178, file: !22, baseType: !32, size: 64, align: 64, offset: 64)
!182 = !{!183, !184, !185}
!183 = !DILocalVariable(name: "lhs", arg: 1, scope: !144, file: !145, line: 68, type: !32)
!184 = !DILocalVariable(name: "rhs", arg: 2, scope: !144, file: !145, line: 68, type: !32)
!185 = !DILocalVariable(name: "msg", scope: !186, file: !145, line: 70, type: !178, align: 64)
!186 = distinct !DILexicalBlock(scope: !144, file: !145, line: 70, column: 21)
!187 = !DILocation(line: 68, column: 43, scope: !144)
!188 = !DILocalVariable(name: "self", arg: 1, scope: !189, file: !190, line: 2645, type: !32)
!189 = distinct !DISubprogram(name: "overflowing_add", linkageName: "_ZN4core3num23_$LT$impl$u20$usize$GT$15overflowing_add17h8fc014c6b1c2b038E", scope: !147, file: !190, line: 2645, type: !191, scopeLine: 2645, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !9, templateParams: !29, retainedNodes: !197)
!190 = !DIFile(filename: "/home/manuel/prog/rust2/library/core/src/num/uint_macros.rs", directory: "", checksumkind: CSK_MD5, checksum: "65b9bf3a9a50c9aa1f2509ce7b373fab")
!191 = !DISubroutineType(types: !192)
!192 = !{!193, !32, !32}
!193 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "(usize, bool)", file: !22, size: 128, align: 64, elements: !194, templateParams: !29, identifier: "d571287e27d8be874e95a2f698798cc6")
!194 = !{!195, !196}
!195 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !193, file: !22, baseType: !32, size: 64, align: 64)
!196 = !DIDerivedType(tag: DW_TAG_member, name: "__1", scope: !193, file: !22, baseType: !71, size: 8, align: 8, offset: 64)
!197 = !{!188, !198}
!198 = !DILocalVariable(name: "rhs", arg: 2, scope: !189, file: !190, line: 2645, type: !32)
!199 = !DILocation(line: 2645, column: 38, scope: !189, inlinedAt: !200)
!200 = !DILocation(line: 712, column: 27, scope: !201)
!201 = !DILexicalBlockFile(scope: !144, file: !190, discriminator: 0)
!202 = !DILocation(line: 2645, column: 44, scope: !189, inlinedAt: !200)
!203 = !DILocation(line: 70, column: 25, scope: !186)
!204 = !DILocation(line: 2646, column: 26, scope: !189, inlinedAt: !200)
!205 = !DILocation(line: 712, column: 23, scope: !201)
!206 = !DILocation(line: 75, column: 14, scope: !144)
!207 = !DILocation(line: 73, column: 94, scope: !186)
!208 = !DILocation(line: 73, column: 93, scope: !186)
!209 = !DILocalVariable(name: "pieces", arg: 1, scope: !210, file: !211, line: 194, type: !340)
!210 = distinct !DISubprogram(name: "new_const<1>", linkageName: "_ZN4core3fmt2rt38_$LT$impl$u20$core..fmt..Arguments$GT$9new_const17h03b905de218d8016E", scope: !212, file: !211, line: 194, type: !338, scopeLine: 194, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !9, templateParams: !29, declaration: !344, retainedNodes: !345)
!211 = !DIFile(filename: "/home/manuel/prog/rust2/library/core/src/fmt/rt.rs", directory: "", checksumkind: CSK_MD5, checksum: "03cba3c9b7eca44212bc16adf1d5b1bc")
!212 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Arguments", scope: !213, file: !22, size: 384, align: 64, flags: DIFlagPublic, elements: !214, templateParams: !29, identifier: "7a47f06667015a1ff7fff03ffaaee97c")
!213 = !DINamespace(name: "fmt", scope: !18)
!214 = !{!215, !221, !263}
!215 = !DIDerivedType(tag: DW_TAG_member, name: "pieces", scope: !212, file: !22, baseType: !216, size: 128, align: 64, flags: DIFlagPrivate)
!216 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "&[&str]", file: !22, size: 128, align: 64, elements: !217, templateParams: !29, identifier: "4e66b00a376d6af5b8765440fb2839f")
!217 = !{!218, !220}
!218 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !216, file: !22, baseType: !219, size: 64, align: 64)
!219 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !178, size: 64, align: 64, dwarfAddressSpace: 0)
!220 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !216, file: !22, baseType: !32, size: 64, align: 64, offset: 64)
!221 = !DIDerivedType(tag: DW_TAG_member, name: "fmt", scope: !212, file: !22, baseType: !222, size: 128, align: 64, offset: 256, flags: DIFlagPrivate)
!222 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Option<&[core::fmt::rt::Placeholder]>", scope: !23, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !223, templateParams: !29, identifier: "c6d3d6d504f5486b372bbeb9e0c91cbd")
!223 = !{!224}
!224 = distinct !DICompositeType(tag: DW_TAG_variant_part, scope: !222, file: !22, size: 128, align: 64, elements: !225, templateParams: !29, identifier: "430e935c56cbc6e0ed750a83f51f13b9", discriminator: !262)
!225 = !{!226, !258}
!226 = !DIDerivedType(tag: DW_TAG_member, name: "None", scope: !224, file: !22, baseType: !227, size: 128, align: 64, extraData: i64 0)
!227 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "None", scope: !222, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !29, templateParams: !228, identifier: "a6a4ad5e8eafb225c0abf74a67725a76")
!228 = !{!229}
!229 = !DITemplateTypeParameter(name: "T", type: !230)
!230 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "&[core::fmt::rt::Placeholder]", file: !22, size: 128, align: 64, elements: !231, templateParams: !29, identifier: "4dd27f4b868367bbc2c6e6195a6689aa")
!231 = !{!232, !257}
!232 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !230, file: !22, baseType: !233, size: 64, align: 64)
!233 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !234, size: 64, align: 64, dwarfAddressSpace: 0)
!234 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Placeholder", scope: !235, file: !22, size: 384, align: 64, flags: DIFlagPublic, elements: !236, templateParams: !29, identifier: "867cdc70f72690957be906c1595d18b")
!235 = !DINamespace(name: "rt", scope: !213)
!236 = !{!237, !238, !239, !256}
!237 = !DIDerivedType(tag: DW_TAG_member, name: "position", scope: !234, file: !22, baseType: !32, size: 64, align: 64, offset: 256, flags: DIFlagPublic)
!238 = !DIDerivedType(tag: DW_TAG_member, name: "flags", scope: !234, file: !22, baseType: !171, size: 32, align: 32, offset: 320, flags: DIFlagPublic)
!239 = !DIDerivedType(tag: DW_TAG_member, name: "precision", scope: !234, file: !22, baseType: !240, size: 128, align: 64, flags: DIFlagPublic)
!240 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Count", scope: !235, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !241, templateParams: !29, identifier: "1b5ff4df120c65ffc67edd5cf3393fec")
!241 = !{!242}
!242 = distinct !DICompositeType(tag: DW_TAG_variant_part, scope: !240, file: !22, size: 128, align: 64, elements: !243, templateParams: !29, identifier: "cd2436e128c22e79c984813bbb005cb6", discriminator: !255)
!243 = !{!244, !249, !253}
!244 = !DIDerivedType(tag: DW_TAG_member, name: "Is", scope: !242, file: !22, baseType: !245, size: 128, align: 64, extraData: i16 0)
!245 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Is", scope: !240, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !246, templateParams: !29, identifier: "3860b4c7a8e38cf292aa4fb0a6c795f2")
!246 = !{!247}
!247 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !245, file: !22, baseType: !248, size: 16, align: 16, offset: 16, flags: DIFlagPublic)
!248 = !DIBasicType(name: "u16", size: 16, encoding: DW_ATE_unsigned)
!249 = !DIDerivedType(tag: DW_TAG_member, name: "Param", scope: !242, file: !22, baseType: !250, size: 128, align: 64, extraData: i16 1)
!250 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Param", scope: !240, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !251, templateParams: !29, identifier: "40ae355ee4b5ca218177bca1e50700a")
!251 = !{!252}
!252 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !250, file: !22, baseType: !32, size: 64, align: 64, offset: 64, flags: DIFlagPublic)
!253 = !DIDerivedType(tag: DW_TAG_member, name: "Implied", scope: !242, file: !22, baseType: !254, size: 128, align: 64, extraData: i16 2)
!254 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Implied", scope: !240, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !29, identifier: "7592be2a8984ca9d9bf91d4786f6f2c8")
!255 = !DIDerivedType(tag: DW_TAG_member, scope: !240, file: !22, baseType: !248, size: 16, align: 16, flags: DIFlagArtificial)
!256 = !DIDerivedType(tag: DW_TAG_member, name: "width", scope: !234, file: !22, baseType: !240, size: 128, align: 64, offset: 128, flags: DIFlagPublic)
!257 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !230, file: !22, baseType: !32, size: 64, align: 64, offset: 64)
!258 = !DIDerivedType(tag: DW_TAG_member, name: "Some", scope: !224, file: !22, baseType: !259, size: 128, align: 64)
!259 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Some", scope: !222, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !260, templateParams: !228, identifier: "2c475c1db29e723791380fe51a915425")
!260 = !{!261}
!261 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !259, file: !22, baseType: !230, size: 128, align: 64, flags: DIFlagPublic)
!262 = !DIDerivedType(tag: DW_TAG_member, scope: !222, file: !22, baseType: !38, size: 64, align: 64, flags: DIFlagArtificial)
!263 = !DIDerivedType(tag: DW_TAG_member, name: "args", scope: !212, file: !22, baseType: !264, size: 128, align: 64, offset: 128, flags: DIFlagPrivate)
!264 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "&[core::fmt::rt::Argument]", file: !22, size: 128, align: 64, elements: !265, templateParams: !29, identifier: "b758283bdeaee867749f2c2cb86911de")
!265 = !{!266, !337}
!266 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !264, file: !22, baseType: !267, size: 64, align: 64)
!267 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !268, size: 64, align: 64, dwarfAddressSpace: 0)
!268 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Argument", scope: !235, file: !22, size: 128, align: 64, flags: DIFlagPublic, elements: !269, templateParams: !29, identifier: "7a17c71d7ab34d1f646196627b73bb28")
!269 = !{!270}
!270 = !DIDerivedType(tag: DW_TAG_member, name: "ty", scope: !268, file: !22, baseType: !271, size: 128, align: 64, flags: DIFlagPrivate)
!271 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ArgumentType", scope: !235, file: !22, size: 128, align: 64, flags: DIFlagPrivate, elements: !272, templateParams: !29, identifier: "c7ef6c2b97aa1b0898e0dbb3c903ca02")
!272 = !{!273}
!273 = distinct !DICompositeType(tag: DW_TAG_variant_part, scope: !271, file: !22, size: 128, align: 64, elements: !274, templateParams: !29, identifier: "a6059b0b035ce2d4c674d42e8be885b5", discriminator: !336)
!274 = !{!275, !332}
!275 = !DIDerivedType(tag: DW_TAG_member, name: "Placeholder", scope: !273, file: !22, baseType: !276, size: 128, align: 64)
!276 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Placeholder", scope: !271, file: !22, size: 128, align: 64, flags: DIFlagPrivate, elements: !277, templateParams: !29, identifier: "3d9193e469e151d06011b18f07baee69")
!277 = !{!278, !286, !327}
!278 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !276, file: !22, baseType: !279, size: 64, align: 64, flags: DIFlagPrivate)
!279 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "NonNull<()>", scope: !158, file: !22, size: 64, align: 64, flags: DIFlagPublic, elements: !280, templateParams: !284, identifier: "d724caf8b3d2575cfc2119eb7cd406ba")
!280 = !{!281}
!281 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !279, file: !22, baseType: !282, size: 64, align: 64, flags: DIFlagPrivate)
!282 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const ()", baseType: !283, size: 64, align: 64, dwarfAddressSpace: 0)
!283 = !DIBasicType(name: "()", encoding: DW_ATE_unsigned)
!284 = !{!285}
!285 = !DITemplateTypeParameter(name: "T", type: !283)
!286 = !DIDerivedType(tag: DW_TAG_member, name: "formatter", scope: !276, file: !22, baseType: !287, size: 64, align: 64, offset: 64, flags: DIFlagPrivate)
!287 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "unsafe fn(core::ptr::non_null::NonNull<()>, &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error>", baseType: !288, size: 64, align: 64, dwarfAddressSpace: 0)
!288 = !DISubroutineType(types: !289)
!289 = !{!290, !279, !307}
!290 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Result<(), core::fmt::Error>", scope: !291, file: !22, size: 8, align: 8, flags: DIFlagPublic, elements: !292, templateParams: !29, identifier: "acd70010668961733032a81677e3a1f6")
!291 = !DINamespace(name: "result", scope: !18)
!292 = !{!293}
!293 = distinct !DICompositeType(tag: DW_TAG_variant_part, scope: !290, file: !22, size: 8, align: 8, elements: !294, templateParams: !29, identifier: "134a126fb51d6e7bfe4ac5689cdeef83", discriminator: !306)
!294 = !{!295, !302}
!295 = !DIDerivedType(tag: DW_TAG_member, name: "Ok", scope: !293, file: !22, baseType: !296, size: 8, align: 8, extraData: i8 0)
!296 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Ok", scope: !290, file: !22, size: 8, align: 8, flags: DIFlagPublic, elements: !297, templateParams: !299, identifier: "d99948acdf64f45322eea7a5541ef0eb")
!297 = !{!298}
!298 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !296, file: !22, baseType: !283, align: 8, offset: 8, flags: DIFlagPublic)
!299 = !{!285, !300}
!300 = !DITemplateTypeParameter(name: "E", type: !301)
!301 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Error", scope: !213, file: !22, align: 8, flags: DIFlagPublic, elements: !29, identifier: "4652b324dc796ed82167a937d9695cdc")
!302 = !DIDerivedType(tag: DW_TAG_member, name: "Err", scope: !293, file: !22, baseType: !303, size: 8, align: 8, extraData: i8 1)
!303 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Err", scope: !290, file: !22, size: 8, align: 8, flags: DIFlagPublic, elements: !304, templateParams: !299, identifier: "8c9647fdf36c92caff8b2560cc299bc7")
!304 = !{!305}
!305 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !303, file: !22, baseType: !301, align: 8, offset: 8, flags: DIFlagPublic)
!306 = !DIDerivedType(tag: DW_TAG_member, scope: !290, file: !22, baseType: !166, size: 8, align: 8, flags: DIFlagArtificial)
!307 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&mut core::fmt::Formatter", baseType: !308, size: 64, align: 64, dwarfAddressSpace: 0)
!308 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Formatter", scope: !213, file: !22, size: 192, align: 64, flags: DIFlagPublic, elements: !309, templateParams: !29, identifier: "97ade6d90b9d899aeeda5c8143d9b7a5")
!309 = !{!310, !316}
!310 = !DIDerivedType(tag: DW_TAG_member, name: "options", scope: !308, file: !22, baseType: !311, size: 64, align: 32, offset: 128, flags: DIFlagPrivate)
!311 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "FormattingOptions", scope: !213, file: !22, size: 64, align: 32, flags: DIFlagPublic, elements: !312, templateParams: !29, identifier: "8b4ddd817c2ce67f9ceec8a0f8a7ceb7")
!312 = !{!313, !314, !315}
!313 = !DIDerivedType(tag: DW_TAG_member, name: "flags", scope: !311, file: !22, baseType: !171, size: 32, align: 32, flags: DIFlagPrivate)
!314 = !DIDerivedType(tag: DW_TAG_member, name: "width", scope: !311, file: !22, baseType: !248, size: 16, align: 16, offset: 32, flags: DIFlagPrivate)
!315 = !DIDerivedType(tag: DW_TAG_member, name: "precision", scope: !311, file: !22, baseType: !248, size: 16, align: 16, offset: 48, flags: DIFlagPrivate)
!316 = !DIDerivedType(tag: DW_TAG_member, name: "buf", scope: !308, file: !22, baseType: !317, size: 128, align: 64, flags: DIFlagPrivate)
!317 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "&mut dyn core::fmt::Write", file: !22, size: 128, align: 64, elements: !318, templateParams: !29, identifier: "5774209541d22b676b07bca66e5d4536")
!318 = !{!319, !322}
!319 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !317, file: !22, baseType: !320, size: 64, align: 64)
!320 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !321, size: 64, align: 64, dwarfAddressSpace: 0)
!321 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dyn core::fmt::Write", file: !22, align: 8, elements: !29, identifier: "5dfd794ece931e18d0ba3c6f661eb5f6")
!322 = !DIDerivedType(tag: DW_TAG_member, name: "vtable", scope: !317, file: !22, baseType: !323, size: 64, align: 64, offset: 64)
!323 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&[usize; 6]", baseType: !324, size: 64, align: 64, dwarfAddressSpace: 0)
!324 = !DICompositeType(tag: DW_TAG_array_type, baseType: !32, size: 384, align: 64, elements: !325)
!325 = !{!326}
!326 = !DISubrange(count: 6, lowerBound: 0)
!327 = !DIDerivedType(tag: DW_TAG_member, name: "_lifetime", scope: !276, file: !22, baseType: !328, align: 8, offset: 128, flags: DIFlagPrivate)
!328 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "PhantomData<&()>", scope: !175, file: !22, align: 8, flags: DIFlagPublic, elements: !29, templateParams: !329, identifier: "ba892b4c18db535076ae1e656d61832d")
!329 = !{!330}
!330 = !DITemplateTypeParameter(name: "T", type: !331)
!331 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&()", baseType: !283, size: 64, align: 64, dwarfAddressSpace: 0)
!332 = !DIDerivedType(tag: DW_TAG_member, name: "Count", scope: !273, file: !22, baseType: !333, size: 128, align: 64, extraData: i64 0)
!333 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Count", scope: !271, file: !22, size: 128, align: 64, flags: DIFlagPrivate, elements: !334, templateParams: !29, identifier: "33428427eb5c845177614220fcfef113")
!334 = !{!335}
!335 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !333, file: !22, baseType: !248, size: 16, align: 16, offset: 64, flags: DIFlagPrivate)
!336 = !DIDerivedType(tag: DW_TAG_member, scope: !271, file: !22, baseType: !38, size: 64, align: 64, flags: DIFlagArtificial)
!337 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !264, file: !22, baseType: !32, size: 64, align: 64, offset: 64)
!338 = !DISubroutineType(types: !339)
!339 = !{!212, !340}
!340 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&[&str; 1]", baseType: !341, size: 64, align: 64, dwarfAddressSpace: 0)
!341 = !DICompositeType(tag: DW_TAG_array_type, baseType: !178, size: 128, align: 64, elements: !342)
!342 = !{!343}
!343 = !DISubrange(count: 1, lowerBound: 0)
!344 = !DISubprogram(name: "new_const<1>", linkageName: "_ZN4core3fmt2rt38_$LT$impl$u20$core..fmt..Arguments$GT$9new_const17h03b905de218d8016E", scope: !212, file: !211, line: 194, type: !338, scopeLine: 194, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit, templateParams: !29)
!345 = !{!209}
!346 = !DILocation(line: 194, column: 44, scope: !210, inlinedAt: !347)
!347 = !DILocation(line: 73, column: 59, scope: !186)
!348 = !DILocation(line: 196, column: 9, scope: !210, inlinedAt: !347)
!349 = !DILocation(line: 73, column: 21, scope: !186)
!350 = !{!351}
!351 = distinct !{!351, !352, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14grow_amortized17hb1d395eb5c3a51a4E: argument 0"}
!352 = distinct !{!352, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14grow_amortized17hb1d395eb5c3a51a4E"}
!353 = !{!"branch_weights", !"expected", i32 1, i32 2000}
!354 = !{i64 0, i64 -9223372036854775808}
!355 = !{!"branch_weights", i32 2002, i32 2000}
!356 = !{!357}
!357 = distinct !{!357, !358, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14current_memory17h62a2dcaa88e8c12bE: argument 0"}
!358 = distinct !{!358, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14current_memory17h62a2dcaa88e8c12bE"}
!359 = !{i64 0, i64 2}
!360 = !{i64 0, i64 -9223372036854775807}
!361 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!362 = !{!363}
!363 = distinct !{!363, !364, !"_ZN5alloc3vec16Vec$LT$T$C$A$GT$15append_elements17hd132e62fa7803d84E: argument 0"}
!364 = distinct !{!364, !"_ZN5alloc3vec16Vec$LT$T$C$A$GT$15append_elements17hd132e62fa7803d84E"}
!365 = !{!366, !363}
!366 = distinct !{!366, !367, !"_ZN5alloc3vec16Vec$LT$T$C$A$GT$7reserve17h2d769b7b1ee34e7eE: argument 0"}
!367 = distinct !{!367, !"_ZN5alloc3vec16Vec$LT$T$C$A$GT$7reserve17h2d769b7b1ee34e7eE"}
!368 = !{!369}
!369 = distinct !{!369, !370, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$13shrink_to_fit17h85ecc1cd80b574c6E: argument 0"}
!370 = distinct !{!370, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$13shrink_to_fit17h85ecc1cd80b574c6E"}
!371 = !{!372}
!372 = distinct !{!372, !373, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$16shrink_unchecked17h6d548b35ef48c425E: argument 0"}
!373 = distinct !{!373, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$16shrink_unchecked17h6d548b35ef48c425E"}
!374 = !{!372, !369}
!375 = !{!376}
!376 = distinct !{!376, !377, !"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17hdf7f45732cc816a4E: argument 0"}
!377 = distinct !{!377, !"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17hdf7f45732cc816a4E"}
!378 = !{!379}
!379 = distinct !{!379, !380, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14grow_amortized17hb1d395eb5c3a51a4E: argument 0"}
!380 = distinct !{!380, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14grow_amortized17hb1d395eb5c3a51a4E"}
!381 = !{!382}
!382 = distinct !{!382, !383, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14current_memory17h62a2dcaa88e8c12bE: argument 0"}
!383 = distinct !{!383, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14current_memory17h62a2dcaa88e8c12bE"}
!384 = !{!385}
!385 = distinct !{!385, !386, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$17try_reserve_exact17hbfa71e3764eae68cE: argument 0"}
!386 = distinct !{!386, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$17try_reserve_exact17hbfa71e3764eae68cE"}
!387 = !{!388}
!388 = distinct !{!388, !389, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$10grow_exact17hf83aab6f421c3530E: argument 0"}
!389 = distinct !{!389, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$10grow_exact17hf83aab6f421c3530E"}
!390 = !{!388, !385}
!391 = !{!392}
!392 = distinct !{!392, !393, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14current_memory17h62a2dcaa88e8c12bE: argument 0"}
!393 = distinct !{!393, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14current_memory17h62a2dcaa88e8c12bE"}
!394 = !{!395}
!395 = distinct !{!395, !396, !"_ZN5alloc3ffi5c_str7CString3new19spec_new_impl_bytes17h8cadcd938f57d2a2E: argument 0"}
!396 = distinct !{!396, !"_ZN5alloc3ffi5c_str7CString3new19spec_new_impl_bytes17h8cadcd938f57d2a2E"}
!397 = !{!395, !398}
!398 = distinct !{!398, !396, !"_ZN5alloc3ffi5c_str7CString3new19spec_new_impl_bytes17h8cadcd938f57d2a2E: argument 1"}
!399 = !{!400}
!400 = distinct !{!400, !401, !"_ZN4core5slice6memchr6memchr17h7a8fef6b3a6004e9E: argument 0"}
!401 = distinct !{!401, !"_ZN4core5slice6memchr6memchr17h7a8fef6b3a6004e9E"}
!402 = !{!398}
!403 = !{!404}
!404 = distinct !{!404, !405, !"_ZN5alloc3ffi5c_str7CString19_from_vec_unchecked17hab60afa2994aa81aE: argument 0"}
!405 = distinct !{!405, !"_ZN5alloc3ffi5c_str7CString19_from_vec_unchecked17hab60afa2994aa81aE"}
!406 = !{!407, !404}
!407 = distinct !{!407, !408, !"_ZN5alloc3vec16Vec$LT$T$C$A$GT$8push_mut17h5ec2591c1f1af24aE: argument 0"}
!408 = distinct !{!408, !"_ZN5alloc3vec16Vec$LT$T$C$A$GT$8push_mut17h5ec2591c1f1af24aE"}
!409 = !{!404, !395}
!410 = !{i64 8}
!411 = !{i64 1}
!412 = !{!413}
!413 = distinct !{!413, !414, !"_ZN4core3fmt3run17hf294645269a8e5ebE: argument 0"}
!414 = distinct !{!414, !"_ZN4core3fmt3run17hf294645269a8e5ebE"}
!415 = !{!416}
!416 = distinct !{!416, !414, !"_ZN4core3fmt3run17hf294645269a8e5ebE: argument 1"}
!417 = !{!418}
!418 = distinct !{!418, !414, !"_ZN4core3fmt3run17hf294645269a8e5ebE: argument 2"}
!419 = !{!420}
!420 = distinct !{!420, !421, !"_ZN4core3fmt8getcount17hc8769f67454d1608E: argument 0"}
!421 = distinct !{!421, !"_ZN4core3fmt8getcount17hc8769f67454d1608E"}
!422 = !{!423}
!423 = distinct !{!423, !421, !"_ZN4core3fmt8getcount17hc8769f67454d1608E: argument 1"}
!424 = !{i16 0, i16 3}
!425 = !{!423, !416}
!426 = !{!420, !413, !418}
!427 = !{!420, !418}
!428 = !{!423, !413, !416}
!429 = !{!430}
!430 = distinct !{!430, !431, !"_ZN4core3fmt8getcount17hc8769f67454d1608E: argument 0"}
!431 = distinct !{!431, !"_ZN4core3fmt8getcount17hc8769f67454d1608E"}
!432 = !{!433}
!433 = distinct !{!433, !431, !"_ZN4core3fmt8getcount17hc8769f67454d1608E: argument 1"}
!434 = !{!433, !416}
!435 = !{!430, !413, !418}
!436 = !{!430, !418}
!437 = !{!433, !413, !416}
!438 = !{!413, !418}
!439 = !{!416, !418}
!440 = !{!413, !416}
!441 = !{!442}
!442 = distinct !{!442, !443, !"_ZN4core5slice6memchr14memchr_aligned7runtime17h4813b5b4b7e38829E: argument 0"}
!443 = distinct !{!443, !"_ZN4core5slice6memchr14memchr_aligned7runtime17h4813b5b4b7e38829E"}
!444 = !{!445}
!445 = distinct !{!445, !446, !"_ZN4core5slice6memchr6memchr17h7a8fef6b3a6004e9E: argument 0"}
!446 = distinct !{!446, !"_ZN4core5slice6memchr6memchr17h7a8fef6b3a6004e9E"}
!447 = distinct !DISubprogram(name: "forward_unchecked", linkageName: "_ZN49_$LT$usize$u20$as$u20$core..iter..range..Step$GT$17forward_unchecked17h4a46ddd3bf136cddE", scope: !448, file: !14, line: 204, type: !449, scopeLine: 204, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !11, templateParams: !29, retainedNodes: !451)
!448 = !DINamespace(name: "{impl#43}", scope: !16)
!449 = !DISubroutineType(types: !450)
!450 = !{!32, !32, !32}
!451 = !{!452, !453}
!452 = !DILocalVariable(name: "start", arg: 1, scope: !447, file: !14, line: 204, type: !32)
!453 = !DILocalVariable(name: "n", arg: 2, scope: !447, file: !14, line: 204, type: !32)
!454 = !DILocation(line: 204, column: 37, scope: !447)
!455 = !DILocalVariable(name: "self", arg: 1, scope: !456, file: !190, line: 705, type: !32)
!456 = distinct !DISubprogram(name: "unchecked_add", linkageName: "_ZN4core3num23_$LT$impl$u20$usize$GT$13unchecked_add17hfedfbef8259bd54bE", scope: !147, file: !190, line: 705, type: !457, scopeLine: 705, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !11, templateParams: !29, retainedNodes: !459)
!457 = !DISubroutineType(types: !458)
!458 = !{!32, !32, !32, !151}
!459 = !{!455, !460}
!460 = !DILocalVariable(name: "rhs", arg: 2, scope: !456, file: !190, line: 705, type: !32)
!461 = !DILocation(line: 705, column: 43, scope: !456, inlinedAt: !462)
!462 = !DILocation(line: 206, column: 28, scope: !447)
!463 = !DILocation(line: 204, column: 50, scope: !447)
!464 = !DILocation(line: 705, column: 49, scope: !456, inlinedAt: !462)
!465 = !DILocation(line: 77, column: 35, scope: !466, inlinedAt: !462)
!466 = !DILexicalBlockFile(scope: !456, file: !145, discriminator: 0)
!467 = !DILocation(line: 78, column: 17, scope: !466, inlinedAt: !462)
!468 = !DILocation(line: 717, column: 17, scope: !456, inlinedAt: !462)
!469 = !DILocation(line: 207, column: 10, scope: !447)
!470 = !{!471, !473}
!471 = distinct !{!471, !472, !"_ZN5alloc5boxed12Box$LT$T$GT$3new17h6600c742345a444cE: argument 0"}
!472 = distinct !{!472, !"_ZN5alloc5boxed12Box$LT$T$GT$3new17h6600c742345a444cE"}
!473 = distinct !{!473, !474, !"_ZN12panic_unwind3imp5panic17hfaff90219a6368a7E: argument 0"}
!474 = distinct !{!474, !"_ZN12panic_unwind3imp5panic17hfaff90219a6368a7E"}
!475 = !{!473}
!476 = !{i64 1, i64 0}
!477 = !{!"branch_weights", i32 2000, i32 14002}
!478 = !{i8 0, i8 2}
!479 = !{!480, !482}
!480 = distinct !{!480, !481, !"_ZN4core3ptr70drop_in_place$LT$core..option..Option$LT$alloc..string..String$GT$$GT$17h90ef220e3085c8cbE: argument 0"}
!481 = distinct !{!481, !"_ZN4core3ptr70drop_in_place$LT$core..option..Option$LT$alloc..string..String$GT$$GT$17h90ef220e3085c8cbE"}
!482 = distinct !{!482, !483, !"_ZN4core3ptr71drop_in_place$LT$std..panicking..panic_handler..FormatStringPayload$GT$17h1e88e0c52d8f6bccE: argument 0"}
!483 = distinct !{!483, !"_ZN4core3ptr71drop_in_place$LT$std..panicking..panic_handler..FormatStringPayload$GT$17h1e88e0c52d8f6bccE"}
!484 = !{!"branch_weights", i32 -294967296, i32 6003000}
!485 = !{!486}
!486 = distinct !{!486, !487, !"_ZN3std4sync6poison6rwlock24RwLockReadGuard$LT$T$GT$3new17h978e83deb7c36b6bE: argument 0"}
!487 = distinct !{!487, !"_ZN3std4sync6poison6rwlock24RwLockReadGuard$LT$T$GT$3new17h978e83deb7c36b6bE"}
!488 = !{i64 4}
!489 = !{!"branch_weights", i32 1, i32 4001}
!490 = !{!491, !493, !495, !497}
!491 = distinct !{!491, !492, !"_ZN71_$LT$alloc..sync..Arc$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h2565a0d01e8ca8c1E: argument 0"}
!492 = distinct !{!492, !"_ZN71_$LT$alloc..sync..Arc$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h2565a0d01e8ca8c1E"}
!493 = distinct !{!493, !494, !"_ZN4core3ptr109drop_in_place$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$17hdc41514f2fb1efeeE: argument 0"}
!494 = distinct !{!494, !"_ZN4core3ptr109drop_in_place$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$17hdc41514f2fb1efeeE"}
!495 = distinct !{!495, !496, !"_ZN4core3ptr137drop_in_place$LT$core..option..Option$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$$GT$17h1e6d5c58f0995081E: argument 0"}
!496 = distinct !{!496, !"_ZN4core3ptr137drop_in_place$LT$core..option..Option$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$$GT$17h1e6d5c58f0995081E"}
!497 = distinct !{!497, !498, !"_ZN4core3ptr165drop_in_place$LT$core..option..Option$LT$core..option..Option$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$$GT$$GT$17h73c201329a2f2d93E: argument 0"}
!498 = distinct !{!498, !"_ZN4core3ptr165drop_in_place$LT$core..option..Option$LT$core..option..Option$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$$GT$$GT$17h73c201329a2f2d93E"}
!499 = !{!500, !502}
!500 = distinct !{!500, !501, !"_ZN71_$LT$alloc..sync..Arc$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h2565a0d01e8ca8c1E: argument 0"}
!501 = distinct !{!501, !"_ZN71_$LT$alloc..sync..Arc$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h2565a0d01e8ca8c1E"}
!502 = distinct !{!502, !503, !"_ZN4core3ptr109drop_in_place$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$17hdc41514f2fb1efeeE: argument 0"}
!503 = distinct !{!503, !"_ZN4core3ptr109drop_in_place$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$17hdc41514f2fb1efeeE"}
!504 = !{i8 0, i8 4}
!505 = !{!506}
!506 = distinct !{!506, !507, !"_ZN3std4sync6poison5mutex14Mutex$LT$T$GT$4lock17hc0bd6981bf8efe2fE: argument 0"}
!507 = distinct !{!507, !"_ZN3std4sync6poison5mutex14Mutex$LT$T$GT$4lock17hc0bd6981bf8efe2fE"}
!508 = !{!509}
!509 = distinct !{!509, !510, !"_ZN64_$LT$std..sys..stdio..unix..Stderr$u20$as$u20$std..io..Write$GT$5write17hfde181961ad61673E: argument 0"}
!510 = distinct !{!510, !"_ZN64_$LT$std..sys..stdio..unix..Stderr$u20$as$u20$std..io..Write$GT$5write17hfde181961ad61673E"}
!511 = !{!"branch_weights", i32 1, i32 2000, i32 2000, i32 2000, i32 2000}
!512 = !{i8 0, i8 42}
!513 = !{!514, !516, !518, !520}
!514 = distinct !{!514, !515, !"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E: argument 0"}
!515 = distinct !{!515, !"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"}
!516 = distinct !{!516, !517, !"_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17hf85b16a76ee5b37dE: argument 0"}
!517 = distinct !{!517, !"_ZN4core3ptr46drop_in_place$LT$alloc..vec..Vec$LT$u8$GT$$GT$17hf85b16a76ee5b37dE"}
!518 = distinct !{!518, !519, !"_ZN4core3ptr76drop_in_place$LT$core..cell..UnsafeCell$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$17h96e12135cfd86ca1E: argument 0"}
!519 = distinct !{!519, !"_ZN4core3ptr76drop_in_place$LT$core..cell..UnsafeCell$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$17h96e12135cfd86ca1E"}
!520 = distinct !{!520, !521, !"_ZN4core3ptr85drop_in_place$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$17h99a57e512481b25eE: argument 0"}
!521 = distinct !{!521, !"_ZN4core3ptr85drop_in_place$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$17h99a57e512481b25eE"}
!522 = !{!516, !518, !520}
!523 = !{!524}
!524 = distinct !{!524, !525, !"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E: argument 0"}
!525 = distinct !{!525, !"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"}
!526 = !{!527}
!527 = distinct !{!527, !528, !"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E: argument 0"}
!528 = distinct !{!528, !"_ZN77_$LT$alloc..raw_vec..RawVec$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h442d43ed120749f9E"}
!529 = !{!530}
!530 = distinct !{!530, !531, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14grow_amortized17h73ecfe77cf9c2c84E: argument 0"}
!531 = distinct !{!531, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14grow_amortized17h73ecfe77cf9c2c84E"}
!532 = !{!533}
!533 = distinct !{!533, !534, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14current_memory17he70b04e6cebede6bE: argument 0"}
!534 = distinct !{!534, !"_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$14current_memory17he70b04e6cebede6bE"}
!535 = !{i8 0, i8 3}
!536 = !{!"branch_weights", i32 1, i32 1, i32 2000, i32 2000}
!537 = !{!538, !540, !542, !544}
!538 = distinct !{!538, !539, !"_ZN71_$LT$alloc..sync..Arc$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h2565a0d01e8ca8c1E: argument 0"}
!539 = distinct !{!539, !"_ZN71_$LT$alloc..sync..Arc$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h2565a0d01e8ca8c1E"}
!540 = distinct !{!540, !541, !"_ZN4core3ptr109drop_in_place$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$17hdc41514f2fb1efeeE: argument 0"}
!541 = distinct !{!541, !"_ZN4core3ptr109drop_in_place$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$17hdc41514f2fb1efeeE"}
!542 = distinct !{!542, !543, !"_ZN4core3ptr137drop_in_place$LT$core..option..Option$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$$GT$17h1e6d5c58f0995081E: argument 0"}
!543 = distinct !{!543, !"_ZN4core3ptr137drop_in_place$LT$core..option..Option$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$$GT$17h1e6d5c58f0995081E"}
!544 = distinct !{!544, !545, !"_ZN4core3ptr88drop_in_place$LT$std..io..stdio..try_set_output_capture..$u7b$$u7b$closure$u7d$$u7d$$GT$17he5a7141c9fa4de6aE: argument 0"}
!545 = distinct !{!545, !"_ZN4core3ptr88drop_in_place$LT$std..io..stdio..try_set_output_capture..$u7b$$u7b$closure$u7d$$u7d$$GT$17he5a7141c9fa4de6aE"}
!546 = !{!547}
!547 = distinct !{!547, !548, !"_ZN4core3ptr88drop_in_place$LT$std..io..stdio..try_set_output_capture..$u7b$$u7b$closure$u7d$$u7d$$GT$17he5a7141c9fa4de6aE: argument 0"}
!548 = distinct !{!548, !"_ZN4core3ptr88drop_in_place$LT$std..io..stdio..try_set_output_capture..$u7b$$u7b$closure$u7d$$u7d$$GT$17he5a7141c9fa4de6aE"}
!549 = !{!550}
!550 = distinct !{!550, !551, !"_ZN4core3ptr137drop_in_place$LT$core..option..Option$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$$GT$17h1e6d5c58f0995081E: argument 0"}
!551 = distinct !{!551, !"_ZN4core3ptr137drop_in_place$LT$core..option..Option$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$$GT$17h1e6d5c58f0995081E"}
!552 = !{!550, !547}
!553 = !{!554, !556, !550, !547}
!554 = distinct !{!554, !555, !"_ZN71_$LT$alloc..sync..Arc$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h2565a0d01e8ca8c1E: argument 0"}
!555 = distinct !{!555, !"_ZN71_$LT$alloc..sync..Arc$LT$T$C$A$GT$$u20$as$u20$core..ops..drop..Drop$GT$4drop17h2565a0d01e8ca8c1E"}
!556 = distinct !{!556, !557, !"_ZN4core3ptr109drop_in_place$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$17hdc41514f2fb1efeeE: argument 0"}
!557 = distinct !{!557, !"_ZN4core3ptr109drop_in_place$LT$alloc..sync..Arc$LT$std..sync..poison..mutex..Mutex$LT$alloc..vec..Vec$LT$u8$GT$$GT$$GT$$GT$17hdc41514f2fb1efeeE"}
!558 = !{i64 0, i64 -9223372036854775806}
!559 = !{!"branch_weights", i32 2000, i32 6001}
!560 = !{!561}
!561 = distinct !{!561, !562, !"_ZN90_$LT$std..io..cursor..Cursor$LT$$RF$mut$u20$$u5b$u8$u5d$$GT$$u20$as$u20$std..io..Write$GT$9write_all17hdb2d9c88d10ffdfdE: argument 0"}
!562 = distinct !{!562, !"_ZN90_$LT$std..io..cursor..Cursor$LT$$RF$mut$u20$$u5b$u8$u5d$$GT$$u20$as$u20$std..io..Write$GT$9write_all17hdb2d9c88d10ffdfdE"}
!563 = !{!564}
!564 = distinct !{!564, !562, !"_ZN90_$LT$std..io..cursor..Cursor$LT$$RF$mut$u20$$u5b$u8$u5d$$GT$$u20$as$u20$std..io..Write$GT$9write_all17hdb2d9c88d10ffdfdE: argument 1"}
!565 = !{!566}
!566 = distinct !{!566, !567, !"_ZN3std2io6cursor15slice_write_all17h410739cd02662a0eE: argument 0"}
!567 = distinct !{!567, !"_ZN3std2io6cursor15slice_write_all17h410739cd02662a0eE"}
!568 = !{!569}
!569 = distinct !{!569, !567, !"_ZN3std2io6cursor15slice_write_all17h410739cd02662a0eE: argument 1"}
!570 = !{!571}
!571 = distinct !{!571, !567, !"_ZN3std2io6cursor15slice_write_all17h410739cd02662a0eE: argument 2"}
!572 = !{!573}
!573 = distinct !{!573, !574, !"_ZN3std2io6cursor11slice_write17he89095d9eb96856dE: argument 0"}
!574 = distinct !{!574, !"_ZN3std2io6cursor11slice_write17he89095d9eb96856dE"}
!575 = !{!576}
!576 = distinct !{!576, !574, !"_ZN3std2io6cursor11slice_write17he89095d9eb96856dE: argument 1"}
!577 = !{!578}
!578 = distinct !{!578, !574, !"_ZN3std2io6cursor11slice_write17he89095d9eb96856dE: argument 2"}
!579 = !{!573, !566, !561}
!580 = !{!576, !578, !569, !571, !564}
!581 = !{!582, !584, !576, !578, !569, !571}
!582 = distinct !{!582, !583, !"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$15copy_from_slice17hec1f0011fbe91152E: argument 0"}
!583 = distinct !{!583, !"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$15copy_from_slice17hec1f0011fbe91152E"}
!584 = distinct !{!584, !583, !"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$15copy_from_slice17hec1f0011fbe91152E: argument 1"}
!585 = !{!586, !587, !573, !566, !561}
!586 = distinct !{!586, !583, !"_ZN4core5slice29_$LT$impl$u20$$u5b$T$u5d$$GT$15copy_from_slice17hec1f0011fbe91152E: argument 2"}
!587 = distinct !{!587, !588, !"_ZN3std2io5impls69_$LT$impl$u20$std..io..Write$u20$for$u20$$RF$mut$u20$$u5b$u8$u5d$$GT$5write17h68a35991ff0310a8E: argument 0"}
!588 = distinct !{!588, !"_ZN3std2io5impls69_$LT$impl$u20$std..io..Write$u20$for$u20$$RF$mut$u20$$u5b$u8$u5d$$GT$5write17h68a35991ff0310a8E"}
!589 = !{!"branch_weights", !"expected", i32 2862774, i32 2144620874}
!590 = !{!591}
!591 = distinct !{!591, !592, !"_ZN4core6result19Result$LT$T$C$E$GT$6unwrap17he7b52a82e17b6b8dE: argument 0"}
!592 = distinct !{!592, !"_ZN4core6result19Result$LT$T$C$E$GT$6unwrap17he7b52a82e17b6b8dE"}
!593 = !{i32 0, i32 2}
!594 = !{!595}
!595 = distinct !{!595, !596, !"_ZN4core6result19Result$LT$T$C$E$GT$6unwrap17h3bdbc6dbe93f785fE: argument 0"}
!596 = distinct !{!596, !"_ZN4core6result19Result$LT$T$C$E$GT$6unwrap17h3bdbc6dbe93f785fE"}
!597 = !{!598}
!598 = distinct !{!598, !599, !"_ZN5alloc3vec16Vec$LT$T$C$A$GT$8push_mut17hf8f249dc510df47fE: argument 0"}
!599 = distinct !{!599, !"_ZN5alloc3vec16Vec$LT$T$C$A$GT$8push_mut17hf8f249dc510df47fE"}
!600 = !{i32 0, i32 -1}
!601 = !{!602, !604}
!602 = distinct !{!602, !603, !"_ZN3std6thread17with_current_name28_$u7b$$u7b$closure$u7d$$u7d$17h297334d44d113746E: argument 0"}
!603 = distinct !{!603, !"_ZN3std6thread17with_current_name28_$u7b$$u7b$closure$u7d$$u7d$17h297334d44d113746E"}
!604 = distinct !{!604, !603, !"_ZN3std6thread17with_current_name28_$u7b$$u7b$closure$u7d$$u7d$17h297334d44d113746E: argument 1"}
!605 = !{!604}
!606 = !{!607, !609}
!607 = distinct !{!607, !608, !"_ZN3std6thread17with_current_name28_$u7b$$u7b$closure$u7d$$u7d$17h297334d44d113746E: argument 0"}
!608 = distinct !{!608, !"_ZN3std6thread17with_current_name28_$u7b$$u7b$closure$u7d$$u7d$17h297334d44d113746E"}
!609 = distinct !{!609, !608, !"_ZN3std6thread17with_current_name28_$u7b$$u7b$closure$u7d$$u7d$17h297334d44d113746E: argument 1"}
!610 = !{!611}
!611 = distinct !{!611, !612, !"_ZN3std6thread18thread_name_string16ThreadNameString6as_str17hb92a42e365d9fb27E: argument 0"}
!612 = distinct !{!612, !"_ZN3std6thread18thread_name_string16ThreadNameString6as_str17hb92a42e365d9fb27E"}
!613 = !{!609}
!614 = !{!615}
!615 = distinct !{!615, !616, !"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17h73c1f0f8cb8d1e3cE: argument 0"}
!616 = distinct !{!616, !"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17h73c1f0f8cb8d1e3cE"}
!617 = !{!"branch_weights", i32 4000000, i32 4001}
!618 = !{!619}
!619 = distinct !{!619, !620, !"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17h73c1f0f8cb8d1e3cE: argument 0"}
!620 = distinct !{!620, !"_ZN3std9panicking12default_hook28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$28_$u7b$$u7b$closure$u7d$$u7d$17h73c1f0f8cb8d1e3cE"}
!621 = !{!"branch_weights", i32 4001, i32 4000000}

; CHECK: define internal void @diffe
