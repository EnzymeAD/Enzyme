; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=foo -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=foo -S -o /dev/null | FileCheck %s

; Test that GEP with null pointer does not cause "Illegal updateAnalysis" error
; This test is based on the issue: https://fwd.gymni.ch/IpbFyC
; The fix prevents propagating pointer type to GEP result when input is null

declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare ptr @rust_alloc_zeroed(i64, i64)
declare ptr @rust_alloc(i64, i64)

define void @foo(ptr %0, i64 %1, i1 %2, i64 %3, i64 %4) {
entry:
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
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %19 = icmp eq i64 %11, 0
  %21 = getelementptr i8, ptr null, i64 %3
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 16
  br i1 %2, label %24, label %25

24:
  br i1 %15, label %34, label %26

25:
  br i1 %15, label %37, label %26

26:
  br i1 %19, label %27, label %29

27:
  br i1 %2, label %28, label %31

28:
  %28_val = tail call noundef ptr @rust_alloc_zeroed(i64 noundef range(i64 1, 0) %11, i64 noundef range(i64 1, -9223372036854775807) %3)
  br label %32

29:
  br i1 %2, label %30, label %31

30:
  %30_val = tail call noundef ptr @rust_alloc(i64 noundef %11, i64 noundef range(i64 1, -9223372036854775807) %3)
  br label %32

31:
  ret void

32:
  %32_phi = phi ptr [ %28_val, %28 ], [ %30_val, %30 ]
  %33 = icmp eq ptr %32_phi, null
  br i1 %33, label %34, label %35

34:
  %35_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  ret void

35:
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 16
  ret void

37:
  %38 = icmp sgt i64 %1, -1
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 16
  ret void
}

; CHECK: foo
; Verify that the analysis completes without "Illegal updateAnalysis" error
; CHECK: %21 = getelementptr i8, ptr null, i64 %3: {[-1]:Pointer, [-1,-1]:Anything}
