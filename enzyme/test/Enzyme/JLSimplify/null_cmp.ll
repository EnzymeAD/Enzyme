; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="jl-inst-simplify" -S | FileCheck %s; fi

declare ptr @memchr(ptr, i32, i64)

define i1 @test_memchr_null(ptr %p, i64 %n) {
entry:
  %call = call ptr @memchr(ptr %p, i32 0, i64 %n)
  %cmp = icmp eq ptr %call, null
  ret i1 %cmp
}

define i1 @test_gep_null(ptr %p, i64 %off) {
entry:
  %gep = getelementptr i8, ptr %p, i64 %off
  %cmp = icmp eq ptr %gep, null
  ret i1 %cmp
}

define i1 @test_gep_from_null(ptr %p, i64 %off) {
entry:
  %gep = getelementptr i8, ptr null, i64 %off
  %cmp = icmp eq ptr %gep, %p
  ret i1 %cmp
}

; CHECK-LABEL: define i1 @test_memchr_null(
; CHECK: %cmp = icmp eq ptr %call, null
; CHECK-NEXT: ret i1 %cmp

; CHECK-LABEL: define i1 @test_gep_null(
; CHECK: %cmp = icmp eq ptr %gep, null
; CHECK-NEXT: ret i1 %cmp

; CHECK-LABEL: define i1 @test_gep_from_null(
; CHECK: %cmp = icmp eq ptr %gep, %p
; CHECK-NEXT: ret i1 %cmp
