; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli | FileCheck %s --check-prefix=EXEC; fi

declare ptr @__enzyme_todense(...)
declare i32 @printf(ptr, ...)

@.fmt = private unnamed_addr constant [12 x i8] c"%s:off=%ld\0A\00"
@.s_neg32 = private unnamed_addr constant [6 x i8] c"neg32\00"
@.s_neg64 = private unnamed_addr constant [6 x i8] c"neg64\00"
@.s_cgep = private unnamed_addr constant [5 x i8] c"cgep\00"
@.s_i24 = private unnamed_addr constant [4 x i8] c"i24\00"
@.s_f80 = private unnamed_addr constant [4 x i8] c"f80\00"
@.s_sel = private unnamed_addr constant [4 x i8] c"sel\00"
@.s_mgep = private unnamed_addr constant [5 x i8] c"mgep\00"

define internal double @rec_load(i64 %off, ptr %tag) #1 {
  %r = call i32 (ptr, ...) @printf(ptr @.fmt, ptr %tag, i64 %off)
  ret double 0.0
}

define internal void @nostore(double %v, i64 %off, ptr %tag) #1 {
  ret void
}

; 1. Single-index GEP with negative i32 index must sign-extend, not zero-extend,
;    and must not attach nuw to a potentially negative multiply.
define void @test_neg_gep(i64 %base, i32 %neg32, i64 %neg64) {
  %p1 = call ptr (...) @__enzyme_todense(ptr @rec_load, ptr @nostore, ptr @.s_neg32)
  %p1a = getelementptr inbounds double, ptr %p1, i64 %base
  %p1b = getelementptr inbounds double, ptr %p1a, i32 %neg32
  %v1 = load double, ptr %p1b, align 8

  %p2 = call ptr (...) @__enzyme_todense(ptr @rec_load, ptr @nostore, ptr @.s_neg64)
  %p2a = getelementptr inbounds double, ptr %p2, i64 %base
  %p2b = getelementptr inbounds double, ptr %p2a, i64 %neg64
  %v2 = load double, ptr %p2b, align 8
  ret void
}

; 2. Multi-index constant GEP chained after a runtime GEP must preserve the
;    base offset instead of replacing it with the constant offset alone.
define void @test_const_multi_gep(i64 %base) {
  %p = call ptr (...) @__enzyme_todense(ptr @rec_load, ptr @nostore, ptr @.s_cgep)
  %pa = getelementptr [2 x double], ptr %p, i64 %base
  %pb = getelementptr [2 x double], ptr %pa, i64 0, i64 1
  %v = load double, ptr %pb, align 8
  ret void
}

; 3. Single-index GEP stride must use TypeAllocSize (i24 -> 4, x86_fp80 -> 16),
;    not (TypeSizeInBits + 7) / 8 (3 and 10).
define void @test_alloc_size_stride(i64 %idx) {
  %p1 = call ptr (...) @__enzyme_todense(ptr @rec_load, ptr @nostore, ptr @.s_i24)
  %p1a = getelementptr i24, ptr %p1, i64 %idx
  %v1 = load double, ptr %p1a, align 4

  %p2 = call ptr (...) @__enzyme_todense(ptr @rec_load, ptr @nostore, ptr @.s_f80)
  %p2a = getelementptr x86_fp80, ptr %p2, i64 %idx
  %v2 = load double, ptr %p2a, align 16
  ret void
}

; 4. Select between two todense-derived pointers must replace both operands
;    before lowering loads/stores.
define void @test_select_two_arms(i1 %c, i64 %i, i64 %j) {
  %p = call ptr (...) @__enzyme_todense(ptr @rec_load, ptr @nostore, ptr @.s_sel)
  %g1 = getelementptr inbounds double, ptr %p, i64 %i
  %g2 = getelementptr inbounds double, ptr %p, i64 %j
  %s = select i1 %c, ptr %g1, ptr %g2
  %v = load double, ptr %s, align 8
  ret void
}

; 5. Multi-index non-constant GEP derived from null replacement must not copy
;    inbounds (a non-zero inbounds GEP from null is poison).
define void @test_multi_gep_no_inbounds(i64 %i) {
  %p = call ptr (...) @__enzyme_todense(ptr @rec_load, ptr @nostore, ptr @.s_mgep)
  %g = getelementptr inbounds [2 x double], ptr %p, i64 %i, i64 1
  %v = load double, ptr %g, align 8
  ret void
}

define i32 @main() {
  call void @test_neg_gep(i64 3, i32 -1, i64 -1)
  call void @test_const_multi_gep(i64 3)
  call void @test_alloc_size_stride(i64 1)
  call void @test_select_two_arms(i1 true, i64 2, i64 5)
  call void @test_select_two_arms(i1 false, i64 2, i64 5)
  call void @test_multi_gep_no_inbounds(i64 3)
  ret i32 0
}

attributes #1 = { alwaysinline }

; CHECK-LABEL: define void @test_neg_gep(i64 %base, i32 %neg32, i64 %neg64)
; CHECK: sext i32 %neg32 to i64
; CHECK-NOT: mul nuw
; CHECK: mul nsw i64 {{.*}}, 8
; CHECK-NOT: mul nuw
; CHECK: mul nsw i64 %neg64, 8

; CHECK-LABEL: define void @test_alloc_size_stride(i64 %idx)
; CHECK: mul i64 %idx, 4
; CHECK: mul i64 %idx, 16

; CHECK-LABEL: define void @test_select_two_arms(i1 %c, i64 %i, i64 %j)
; CHECK-NOT: __enzyme_todense
; CHECK: select i1 %c, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}

; CHECK-LABEL: define void @test_multi_gep_no_inbounds(i64 %i)
; CHECK: [[GEP:%[0-9]+]] = getelementptr [2 x double], ptr null, i64 %i, i64 1
; CHECK: ptrtoint ptr [[GEP]] to i64

; EXEC: neg32:off=16
; EXEC-NEXT: neg64:off=16
; EXEC-NEXT: cgep:off=56
; EXEC-NEXT: i24:off=4
; EXEC-NEXT: f80:off=16
; EXEC-NEXT: sel:off=16
; EXEC-NEXT: sel:off=40
; EXEC-NEXT: mgep:off=56
