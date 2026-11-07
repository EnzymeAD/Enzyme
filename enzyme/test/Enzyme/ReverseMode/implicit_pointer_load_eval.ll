; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

%struct.Matrix = type { [4 x double] }
%struct.Product = type { ptr, ptr }
%struct.MixedPair = type { double, ptr }

@.str = private unnamed_addr constant [26 x i8] c"%.6f %.6f %.6f %.6f %.6f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare void @__enzyme_autodiff(...)

define internal ptr @product_lhs(ptr %this) noinline {
entry:
  %slot = alloca ptr, align 8
  store ptr %this, ptr %slot, align 8
  %loaded_this = load ptr, ptr %slot, align 8
  %m_lhs = getelementptr inbounds %struct.Product, ptr %loaded_this, i32 0, i32 0
  %lhs = load ptr, ptr %m_lhs, align 8
  ret ptr %lhs
}

define internal ptr @product_rhs(ptr %this) noinline {
entry:
  %slot = alloca ptr, align 8
  store ptr %this, ptr %slot, align 8
  %loaded_this = load ptr, ptr %slot, align 8
  %m_rhs = getelementptr inbounds %struct.Product, ptr %loaded_this, i32 0, i32 1
  %rhs = load ptr, ptr %m_rhs, align 8
  ret ptr %rhs
}

define internal void @eval_product(ptr %dst, ptr %prod_ref) noinline {
entry:
  %prod_slot = alloca ptr, align 8
  store ptr %prod_ref, ptr %prod_slot, align 8
  %p0 = load ptr, ptr %prod_slot, align 8
  %lhs = call ptr @product_lhs(ptr %p0)
  %p1 = load ptr, ptr %prod_slot, align 8
  %rhs = call ptr @product_rhs(ptr %p1)
  %l0 = load <2 x double>, ptr %lhs, align 8
  %lhs_col1 = getelementptr inbounds double, ptr %lhs, i64 2
  %l1 = load <2 x double>, ptr %lhs_col1, align 8
  br label %loop

loop:
  %col = phi i64 [ 0, %entry ], [ %next_col, %loop ]
  %base = mul i64 %col, 2
  %r0_ptr = getelementptr inbounds double, ptr %rhs, i64 %base
  %r0 = load double, ptr %r0_ptr, align 8
  %idx1 = add i64 %base, 1
  %r1_ptr = getelementptr inbounds double, ptr %rhs, i64 %idx1
  %r1 = load double, ptr %r1_ptr, align 8
  %splat0 = insertelement <2 x double> poison, double %r0, i32 0
  %vec0 = shufflevector <2 x double> %splat0, <2 x double> poison, <2 x i32> zeroinitializer
  %splat1 = insertelement <2 x double> poison, double %r1, i32 0
  %vec1 = shufflevector <2 x double> %splat1, <2 x double> poison, <2 x i32> zeroinitializer
  %mul0 = fmul <2 x double> %l0, %vec0
  %mul1 = fmul <2 x double> %l1, %vec1
  %sum = fadd <2 x double> %mul0, %mul1
  %dst_ptr = getelementptr inbounds double, ptr %dst, i64 %base
  store <2 x double> %sum, ptr %dst_ptr, align 8
  %next_col = add i64 %col, 1
  %done = icmp eq i64 %next_col, 2
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define internal void @eval_add(ptr %dst, ptr %lhs, ptr %rhs) noinline {
entry:
  %a0 = load <2 x double>, ptr %lhs, align 8
  %b0 = load <2 x double>, ptr %rhs, align 8
  %s0 = fadd <2 x double> %a0, %b0
  store <2 x double> %s0, ptr %dst, align 8
  %lhs1 = getelementptr inbounds double, ptr %lhs, i64 2
  %rhs1 = getelementptr inbounds double, ptr %rhs, i64 2
  %dst1 = getelementptr inbounds double, ptr %dst, i64 2
  %a1 = load <2 x double>, ptr %lhs1, align 8
  %b1 = load <2 x double>, ptr %rhs1, align 8
  %s1 = fadd <2 x double> %a1, %b1
  store <2 x double> %s1, ptr %dst1, align 8
  ret void
}

define internal double @load_mixed_pair(ptr "enzyme_type"="{[-1,0]:Float@double, [-1,8,0]:Float@double}" %pair_ptr) noinline {
entry:
  %pair = load %struct.MixedPair, ptr %pair_ptr, align 8
  %scalar = extractvalue %struct.MixedPair %pair, 0
  %ptr_field = extractvalue %struct.MixedPair %pair, 1
  %pointed = load double, ptr %ptr_field, align 8
  %res = fadd double %scalar, %pointed
  ret double %res
}

define void @matrix_objective(ptr %mats, ptr %sol) {
entry:
  %m0 = alloca %struct.Matrix, align 8
  %m1 = alloca %struct.Matrix, align 8
  %m2 = alloca %struct.Matrix, align 8
  %m4 = alloca %struct.Matrix, align 8
  %p0 = alloca %struct.Product, align 8
  %p4 = alloca %struct.Product, align 8
  %pair = alloca %struct.MixedPair, align 8

  %mat0 = getelementptr inbounds %struct.Matrix, ptr %mats, i64 0
  %mat1 = getelementptr inbounds %struct.Matrix, ptr %mats, i64 1
  %mat2 = getelementptr inbounds %struct.Matrix, ptr %mats, i64 2

  %p0_lhs = getelementptr inbounds %struct.Product, ptr %p0, i32 0, i32 0
  store ptr %mat0, ptr %p0_lhs, align 8
  %p0_rhs = getelementptr inbounds %struct.Product, ptr %p0, i32 0, i32 1
  store ptr %mat1, ptr %p0_rhs, align 8
  call void @eval_product(ptr %m0, ptr %p0)

  call void @eval_add(ptr %m1, ptr %m0, ptr %mat1)
  call void @eval_add(ptr %m2, ptr %mat2, ptr %m1)

  ; Unused matrix product reading active M2 through Product::rhs()
  %p4_lhs = getelementptr inbounds %struct.Product, ptr %p4, i32 0, i32 0
  store ptr %m1, ptr %p4_lhs, align 8
  %p4_rhs = getelementptr inbounds %struct.Product, ptr %p4, i32 0, i32 1
  store ptr %m2, ptr %p4_rhs, align 8
  call void @eval_product(ptr %m4, ptr %p4)

  %e0_ptr = getelementptr inbounds double, ptr %m2, i64 0
  %e1_ptr = getelementptr inbounds double, ptr %m2, i64 1
  %e2_ptr = getelementptr inbounds double, ptr %m2, i64 2
  %e3_ptr = getelementptr inbounds double, ptr %m2, i64 3
  %e0 = load double, ptr %e0_ptr, align 8
  %e1 = load double, ptr %e1_ptr, align 8
  %e2 = load double, ptr %e2_ptr, align 8
  %s01 = fadd double %e0, %e1
  %pair_s = getelementptr inbounds %struct.MixedPair, ptr %pair, i32 0, i32 0
  store double %e2, ptr %pair_s, align 8
  %pair_p = getelementptr inbounds %struct.MixedPair, ptr %pair, i32 0, i32 1
  store ptr %e3_ptr, ptr %pair_p, align 8
  %s23 = call double @load_mixed_pair(ptr %pair)
  %total = fadd double %s01, %s23
  store double %total, ptr %sol, align 8
  ret void
}

define i32 @main() {
entry:
  %mats = alloca [3 x %struct.Matrix], align 16
  %dmats = alloca [3 x %struct.Matrix], align 16
  %y = alloca double, align 8
  %dy = alloca double, align 8

  ; Initialize mats[0] = [1, 3, 2, 4] (col-major: A00=1, A10=3, A01=2, A11=4)
  ; mats[1] = [5, 7, 6, 8] (col-major: B00=5, B10=7, B01=6, B11=8)
  ; mats[2] = [9, 11, 10, 12]
  %vals = alloca [12 x double], align 16
  store [12 x double] [double 1.0, double 3.0, double 2.0, double 4.0, double 5.0, double 7.0, double 6.0, double 8.0, double 9.0, double 11.0, double 10.0, double 12.0], ptr %vals, align 16
  %raw_vals = load [12 x double], ptr %vals, align 16
  store [12 x double] %raw_vals, ptr %mats, align 16
  store [12 x double] zeroinitializer, ptr %dmats, align 16
  store double 0.0, ptr %y, align 8
  store double 1.0, ptr %dy, align 8

  call void (...) @__enzyme_autodiff(ptr @matrix_objective, ptr %mats, ptr %dmats, ptr %y, ptr %dy)

  ; For sol = sum(C + A * B + B):
  ; dsol/dA00 = B00 + B01 = 5 + 6 = 11
  ; dsol/dA01 = B10 + B11 = 7 + 8 = 15
  ; dsol/dB00 = 1 + A00 + A10 = 1 + 1 + 3 = 5
  ; dsol/dB10 = 1 + A01 + A11 = 1 + 2 + 4 = 7
  ; dsol/dC00 = 1
  %da00_ptr = getelementptr inbounds double, ptr %dmats, i64 0
  %da01_ptr = getelementptr inbounds double, ptr %dmats, i64 2
  %db00_ptr = getelementptr inbounds double, ptr %dmats, i64 4
  %db10_ptr = getelementptr inbounds double, ptr %dmats, i64 5
  %dc00_ptr = getelementptr inbounds double, ptr %dmats, i64 8
  %da00 = load double, ptr %da00_ptr, align 8
  %da01 = load double, ptr %da01_ptr, align 8
  %db00 = load double, ptr %db00_ptr, align 8
  %db10 = load double, ptr %db10_ptr, align 8
  %dc00 = load double, ptr %dc00_ptr, align 8

  %fmt = getelementptr inbounds [26 x i8], ptr @.str, i64 0, i64 0
  call i32 (ptr, ...) @printf(ptr %fmt, double %da00, double %da01, double %db00, double %db10, double %dc00)
  ret i32 0
}

; CHECK-LABEL: define internal void @diffematrix_objective(
; CHECK: ret void

; EVAL: 11.000000 15.000000 5.000000 7.000000 1.000000
