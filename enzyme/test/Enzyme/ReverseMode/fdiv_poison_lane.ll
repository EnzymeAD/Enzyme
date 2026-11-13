; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: -0.250000 1.250000

@.fmt = private unnamed_addr constant [11 x i8] c"%.6f %.6f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare void @__enzyme_autodiff(...)

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  %in = alloca <2 x double>, align 16
  %din = alloca <2 x double>, align 16
  %other = alloca <2 x double>, align 16
  %dother = alloca <2 x double>, align 16
  store <2 x double> <double 4.000000e+00, double 2.000000e+00>, ptr %in, align 16
  store <2 x double> zeroinitializer, ptr %din, align 16
  store <2 x double> <double 7.000000e+00, double 9.000000e+00>, ptr %other, align 16
  store <2 x double> zeroinitializer, ptr %dother, align 16
  ; CHECK: call void @diffesimd_recip_lane0(
  call void (...) @__enzyme_autodiff(ptr @simd_recip_lane0, ptr %in, ptr %din, ptr %other, ptr %dother)
  %grad = load <2 x double>, ptr %din, align 16
  %g0 = extractelement <2 x double> %grad, i64 0
  %g1 = extractelement <2 x double> %grad, i64 1
  call i32 (ptr, ...) @printf(ptr @.fmt, double %g0, double %g1)
  ret i32 0
}

; CHECK-LABEL: define internal void @diffesimd_recip_lane0(
define double @simd_recip_lane0(ptr nocapture readonly %in, ptr nocapture readonly %other) {
entry:
  %vec = load <2 x double>, ptr %in, align 16
  %other_vec = load <2 x double>, ptr %other, align 16
  %hi = shufflevector <2 x double> %vec, <2 x double> %other_vec, <2 x i32> <i32 1, i32 poison>
  %denom = fsub <2 x double> %vec, %hi
  ; CHECK: shufflevector <2 x double> %{{.+}}, <2 x double> zeroinitializer, <2 x i32> <i32 0, i32 3>
  %recip = fdiv <2 x double> <double 1.000000e+00, double poison>, %denom
  %lane0 = extractelement <2 x double> %recip, i64 0
  %lane1 = extractelement <2 x double> %vec, i64 1
  %sum = fadd double %lane0, %lane1
  ret double %sum
}
