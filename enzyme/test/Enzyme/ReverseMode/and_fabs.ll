; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: -2.000000 3.000000 -1.000000

@.fmt = private unnamed_addr constant [13 x i8] c"%f %f %f\0A\00\00\00\00", align 1

declare i32 @printf(ptr, ...)
declare void @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)

define i32 @main() {
entry:
  %vec = alloca [2 x double], align 16
  %dvec = alloca [2 x double], align 16
  store double -4.000000e+00, ptr %vec, align 16
  %vec1 = getelementptr inbounds double, ptr %vec, i64 1
  store double 5.000000e+00, ptr %vec1, align 8
  store double 0.000000e+00, ptr %dvec, align 16
  %dvec1 = getelementptr inbounds double, ptr %dvec, i64 1
  store double 0.000000e+00, ptr %dvec1, align 8
  call void (...) @__enzyme_autodiff(ptr @simd_abs_weighted, metadata !"enzyme_dup", ptr %vec, ptr %dvec)
  %g0 = load double, ptr %dvec, align 16
  %g1 = load double, ptr %dvec1, align 8
  %f0 = call double (...) @__enzyme_fwddiff(ptr @scalar_abs, double -7.000000e+00, double 1.000000e+00)
  %printed = call i32 (ptr, ...) @printf(ptr nonnull @.fmt, double %g0, double %g1, double %f0)
  ret i32 0
}

; CHECK-LABEL: define internal void @diffesimd_abs_weighted(
define double @simd_abs_weighted(ptr nocapture readonly %input) {
entry:
  %loaded = load <2 x double>, ptr %input, align 16
  %bits = bitcast <2 x double> %loaded to <2 x i64>
  ; CHECK: %[[ISNEG:.+]] = fcmp fast olt <2 x double> %loaded, zeroinitializer
  ; CHECK: %[[NEG:.+]] = fneg fast <2 x double> %{{.+}}
  ; CHECK: %[[SEL:.+]] = select fast <2 x i1> %[[ISNEG]], <2 x double> %[[NEG]], <2 x double> %{{.+}}
  %cleared = and <2 x i64> %bits, <i64 9223372036854775807, i64 9223372036854775807>
  %abs_vec = bitcast <2 x i64> %cleared to <2 x double>
  %scaled = fmul <2 x double> %abs_vec, <double 2.000000e+00, double 3.000000e+00>
  %lane0 = extractelement <2 x double> %scaled, i64 0
  %lane1 = extractelement <2 x double> %scaled, i64 1
  %sum = fadd double %lane0, %lane1
  ret double %sum
}

; CHECK-LABEL: define internal double @fwddiffescalar_abs(
define double @scalar_abs(double %x) {
entry:
  %bits = bitcast double %x to i64
  ; CHECK: %[[FWD_ISNEG:.+]] = fcmp fast olt double %x, 0.000000e+00
  ; CHECK: %[[FWD_NEG:.+]] = fneg fast double %"x'"
  ; CHECK: %[[FWD_SEL:.+]] = select fast i1 %[[FWD_ISNEG]], double %[[FWD_NEG]], double %"x'"
  %cleared = and i64 %bits, 9223372036854775807
  %abs_val = bitcast i64 %cleared to double
  ret double %abs_val
}
