; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli -

; Regression test for https://github.com/EnzymeAD/Enzyme/issues/3128:
; inverting a global ConstantDataArray/ConstantDataVector initializer whose
; element slices have Unknown or partially-float TypeTrees must fold to a
; Constant rather than emitting runtime alloca/load instructions.

; CHECK: @tab_shadow = global [16 x i8] zeroinitializer
@tab = global [16 x i8] c"UUUUUU\D5?\00\00\00\00\00\00\D0?", align 8

; Bytes 0..7 are accessed as double (0.5), bytes 8..11 as i32 (42), and bytes
; 12..15 are unaccessed (Unknown), forcing a module-level shadow global whose
; float and Unknown bytes are zeroed while Integer bytes are preserved.
; CHECK: @mixed_tab_shadow = global [16 x i8] c"\00\00\00\00\00\00\00\00*\00\00\00\00\00\00\00"
@mixed_tab = global [16 x i8] c"\00\00\00\00\00\00\E0?*\00\00\00\7F\7F\7F\7F", align 8

; ConstantDataVector with float bytes in elements 0..1 (double 0.5), Integer in
; element 2 (7), and Unknown in element 3 (99).
; CHECK: @vec_tab_shadow = global <4 x i32> <i32 0, i32 0, i32 7, i32 0>
@vec_tab = global <4 x i32> <i32 0, i32 1071644672, i32 7, i32 99>, align 16

@enzyme_const = external global i32
@enzyme_dup = external global i32

declare {} @__enzyme_autodiff(...)
declare double @__enzyme_autodiff_val(...)

; CHECK-LABEL: define void @repro_3128(
define void @repro_3128(ptr %dst, i64 %n, ptr %src, i64 %m, i32 %flag) {
entry:
  store i32 0, ptr @tab, align 4
  ret void
}

define double @mixed_kernel(double %x) {
entry:
  %weight = load double, ptr @mixed_tab, align 8
  %scale_ptr = getelementptr inbounds [16 x i8], ptr @mixed_tab, i64 0, i64 8
  %scale_i32 = load i32, ptr %scale_ptr, align 4
  %scale = sitofp i32 %scale_i32 to double
  %prod1 = fmul double %x, %weight
  %prod2 = fmul double %prod1, %scale
  store double %x, ptr @mixed_tab, align 8

  %vweight = load double, ptr @vec_tab, align 16
  %vscale_ptr = getelementptr inbounds i32, ptr @vec_tab, i64 2
  %vscale_i32 = load i32, ptr %vscale_ptr, align 8
  %vscale = sitofp i32 %vscale_i32 to double
  %vprod1 = fmul double %x, %vweight
  %vprod2 = fmul double %vprod1, %vscale
  store double %x, ptr @vec_tab, align 16

  %sum = fadd double %prod2, %vprod2
  ret double %sum
}

define i32 @main() {
entry:
  %unused = call {} (...) @__enzyme_autodiff(ptr @repro_3128, ptr @enzyme_dup, ptr null, ptr null, ptr @enzyme_const, i64 0, ptr @enzyme_dup, ptr null, ptr null, ptr @enzyme_const, i64 0, ptr @enzyme_const, i32 0)

  ; sum = x * 0.5 * 42 + x * 0.5 * 7 = 24.5 * x => d/dx = 24.5
  %dx = call double (...) @__enzyme_autodiff_val(ptr @mixed_kernel, double 3.000000e+00)
  %err = fsub double %dx, 2.450000e+01
  %err_sq = fmul double %err, %err
  %bad = fcmp ugt double %err_sq, 1.000000e-18
  %rc = select i1 %bad, i32 1, i32 0
  ret i32 %rc
}
