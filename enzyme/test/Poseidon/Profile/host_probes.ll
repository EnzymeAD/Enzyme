; REQUIRES: enzyme
; RUN: %opt < %s %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-generate -S | FileCheck %s

%struct.Gradients = type { double, double }

define double @test_maxnum_zero(double %x) {
entry:
  %cmp = fcmp ogt double %x, 0.0
  %result = select i1 %cmp, double %x, double 0.0
  ret double %result
}

define double @test_maxnum_zero_reversed(double %x) {
entry:
  %cmp = fcmp olt double %x, 0.0
  %result = select i1 %cmp, double 0.0, double %x
  ret double %result
}

define double @test_maxnum_general(double %x, double %y) {
entry:
  %cmp = fcmp ogt double %x, %y
  %result = select i1 %cmp, double %x, double %y
  ret double %result
}

define double @test_minnum_general(double %x, double %y) {
entry:
  %cmp = fcmp olt double %x, %y
  %result = select i1 %cmp, double %x, double %y
  ret double %result
}

define double @test_combined(double %x, double %y, double %z) {
entry:
  %cmp1 = fcmp ogt double %x, 0.0
  %max_x = select i1 %cmp1, double %x, double 0.0
  %cmp2 = fcmp olt double %max_x, %y
  %min_xy = select i1 %cmp2, double %max_x, double %y
  %result = fadd fast double %min_xy, %z
  ret double %result
}

define double @test_fadd(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

define double @test_fma(double %x, double %y, double %z) {
entry:
  %0 = fmul fast double %x, %y
  %1 = fadd fast double %0, %z
  ret double %1
}

define double @muldd(double %x, double %y) {
entry:
  %mul = fmul fast double %x, %y
  ret double %mul
}

define double @test_profile_maxnum_zero(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__poseidon_fp_optimize(double (double)* nonnull @test_maxnum_zero, double %x)
  ret double %0
}

define double @test_profile_maxnum_zero_reversed(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__poseidon_fp_optimize(double (double)* nonnull @test_maxnum_zero_reversed, double %x)
  ret double %0
}

define double @test_profile_maxnum(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @test_maxnum_general, double %x, double %y)
  ret double %0
}

define double @test_profile_minnum(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @test_minnum_general, double %x, double %y)
  ret double %0
}

define double @test_profile_combined(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @test_combined, double %x, double %y, double %z)
  ret double %0
}

define double @test_profile_fadd(double %x, double %y) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @test_fadd, double %x, double %y, metadata !"poseidon_tau", double 1.0e-6)
  ret double %0
}

define double @test_profile_fma(double %x, double %y, double %z) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @test_fma, double %x, double %y, double %z)
  ret double %0
}

; The marker returns the gradient struct, not the type of muldd: the request
; returns no primal (adapted from enzyme/test/Enzyme/ReverseMode/gradient-struct-ret.ll).
define %struct.Gradients @test_profile_muldd(double %x, double %y) {
entry:
  %call = call %struct.Gradients (ptr, ...) @__poseidon_fp_optimize(ptr @muldd, double %x, double %y)
  ret %struct.Gradients %call
}

declare double @__poseidon_fp_optimize(...)

; CHECK: @__poseidon_profile_runtime = external global i32
; CHECK: @poseidon.site.preprocess_test_maxnum_zero = private unnamed_addr constant [17 x i8] c"test_maxnum_zero\00", align 1
; CHECK: @poseidon.site.preprocess_test_maxnum_zero_reversed = private unnamed_addr constant [26 x i8] c"test_maxnum_zero_reversed\00", align 1
; CHECK: @poseidon.site.preprocess_test_maxnum_general = private unnamed_addr constant [20 x i8] c"test_maxnum_general\00", align 1
; CHECK: @poseidon.site.preprocess_test_minnum_general = private unnamed_addr constant [20 x i8] c"test_minnum_general\00", align 1
; CHECK: @poseidon.site.preprocess_test_combined = private unnamed_addr constant [14 x i8] c"test_combined\00", align 1
; CHECK: @poseidon.site.preprocess_test_fadd = private unnamed_addr constant [10 x i8] c"test_fadd\00", align 1
; CHECK: @poseidon.site.preprocess_test_fma = private unnamed_addr constant [9 x i8] c"test_fma\00", align 1
; CHECK: @poseidon.site.preprocess_muldd = private unnamed_addr constant [6 x i8] c"muldd\00", align 1

; One probe per profiled value, and its custom gradient carries that value's
; slot into the gradient record.
; CHECK: define internal double @__poseidon_probe_preprocess_test_combined_0_rev(double %[[v0:.+]], double %[[d0:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @__poseidon_log_grad(ptr @poseidon.site.preprocess_test_combined, i64 0, double %[[v0]], double %[[d0]])
; CHECK-NEXT:   ret double %[[d0]]

; CHECK: define internal double @__poseidon_probe_preprocess_test_combined_1_rev(double %[[v1:.+]], double %[[d1:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @__poseidon_log_grad(ptr @poseidon.site.preprocess_test_combined, i64 1, double %[[v1]], double %[[d1]])

; CHECK: define internal double @__poseidon_probe_preprocess_test_combined_2_rev(double %[[v2:.+]], double %[[d2:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @__poseidon_log_grad(ptr @poseidon.site.preprocess_test_combined, i64 2, double %[[v2]], double %[[d2]])

; The probe leaves the reverse pass through its custom gradient.
; CHECK: define internal double @__poseidon_probe_preprocess_test_fadd_0(double %{{.+}}) #{{[0-9]+}} !enzyme_augment !{{[0-9]+}} !enzyme_gradient !{{[0-9]+}} {

; fcmp+select becomes maxnum/minnum, whichever way the comparison is written,
; and a constant operand is recorded as such.
; CHECK: define internal { double, double } @diffepreprocess_test_maxnum_zero(double %x, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double 0.000000e+00){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx0:[0-9]+]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_test_maxnum_zero_0_aug(double %[[maxnum]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr {{\[2 x double\]|i8}}, ptr %[[alloca]], {{i32 0, i32 1|i64 8}}
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @__poseidon_log_value(ptr @poseidon.site.preprocess_test_maxnum_zero, i64 0, double %[[probed]], i64 2, ptr %[[alloca]])

; CHECK: define internal { double, double } @diffepreprocess_test_maxnum_zero_reversed(
; CHECK-NEXT: entry:
; CHECK-NEXT:   alloca [2 x double], align 8
; CHECK-NEXT:   call double @llvm.maxnum.f64(double %x, double 0.000000e+00){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx0]]

; CHECK: define internal { double, double, double } @diffepreprocess_test_maxnum_general(
; CHECK-NEXT: entry:
; CHECK-NEXT:   alloca [2 x double], align 8
; CHECK-NEXT:   call double @llvm.maxnum.f64(double %x, double %y){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx0]]

; CHECK: define internal { double, double, double } @diffepreprocess_test_minnum_general(
; CHECK-NEXT: entry:
; CHECK-NEXT:   alloca [2 x double], align 8
; CHECK-NEXT:   call double @llvm.minnum.f64(double %x, double %y){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx0]]

; Three profiled values in one site: each is logged under its own slot and each
; operand array records the probed (not the raw) value of its producer.
; CHECK: define internal { double, double, double, double } @diffepreprocess_test_combined(double %x, double %y, double %z, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca0:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[alloca1:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[alloca2:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double 0.000000e+00){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx0]]
; CHECK-NEXT:   %[[probed0:.+]] = call fast double @__poseidon_probe_preprocess_test_combined_0_aug(double %[[maxnum]])
; CHECK-NEXT:   store double %x, ptr %[[alloca0]], align 8
; CHECK-NEXT:   %[[gep0:.+]] = getelementptr {{\[2 x double\]|i8}}, ptr %[[alloca0]], {{i32 0, i32 1|i64 8}}
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[gep0]], align 8
; CHECK-NEXT:   call void @__poseidon_log_value(ptr @poseidon.site.preprocess_test_combined, i64 0, double %[[probed0]], i64 2, ptr %[[alloca0]])
; CHECK-NEXT:   %[[minnum:.+]] = call double @llvm.minnum.f64(double %[[probed0]], double %y){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx1:[0-9]+]]
; CHECK-NEXT:   %[[probed1:.+]] = call fast double @__poseidon_probe_preprocess_test_combined_1_aug(double %[[minnum]])
; CHECK-NEXT:   store double %[[probed0]], ptr %[[alloca1]], align 8
; CHECK-NEXT:   %[[gep1:.+]] = getelementptr {{\[2 x double\]|i8}}, ptr %[[alloca1]], {{i32 0, i32 1|i64 8}}
; CHECK-NEXT:   store double %y, ptr %[[gep1]], align 8
; CHECK-NEXT:   call void @__poseidon_log_value(ptr @poseidon.site.preprocess_test_combined, i64 1, double %[[probed1]], i64 2, ptr %[[alloca1]])
; CHECK-NEXT:   %result = fadd fast double %[[probed1]], %z, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx2:[0-9]+]]
; CHECK-NEXT:   %[[probed2:.+]] = call fast double @__poseidon_probe_preprocess_test_combined_2_aug(double %result)
; CHECK-NEXT:   store double %[[probed1]], ptr %[[alloca2]], align 8
; CHECK-NEXT:   %[[gep2:.+]] = getelementptr {{\[2 x double\]|i8}}, ptr %[[alloca2]], {{i32 0, i32 1|i64 8}}
; CHECK-NEXT:   store double %z, ptr %[[gep2]], align 8
; CHECK-NEXT:   call void @__poseidon_log_value(ptr @poseidon.site.preprocess_test_combined, i64 2, double %[[probed2]], i64 2, ptr %[[alloca2]])
; CHECK-NEXT:   %[[gc2:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_combined_2(double %result, double %[[differet]])
; CHECK-NEXT:   %[[g2:.+]] = extractvalue { double } %[[gc2]], 0
; CHECK-NEXT:   %[[gc1:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_combined_1(double %[[minnum]], double %[[g2]])
; CHECK-NEXT:   %[[g1:.+]] = extractvalue { double } %[[gc1]], 0

; CHECK: define internal { double, double, double } @diffepreprocess_test_fadd(
; CHECK-NEXT: entry:
; CHECK-NEXT:   alloca [2 x double], align 8
; CHECK-NEXT:   fadd fast double %x, %y, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx0]]
; CHECK:        call void @__poseidon_log_value(ptr @poseidon.site.preprocess_test_fadd, i64 0, double %{{.+}}, i64 2, ptr %{{.+}})

; An fmuladd logs its three operands.
; CHECK: define internal { double, double, double, double } @diffepreprocess_test_fma(double %x, double %y, double %z, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [3 x double], align 8
; CHECK-NEXT:   %[[fmuladd:.+]] = call fast double @llvm.fmuladd.f64(double %x, double %y, double %z){{.*}}, !enzyme_active !{{[0-9]+}}, !poseidon.slot ![[idx0]]
; CHECK-NEXT:   %[[probed:.+]] = call fast double @__poseidon_probe_preprocess_test_fma_0_aug(double %[[fmuladd]])
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep1:.+]] = getelementptr {{\[3 x double\]|i8}}, ptr %[[alloca]], {{i32 0, i32 1|i64 8}}
; CHECK-NEXT:   store double %y, ptr %[[gep1]], align 8
; CHECK-NEXT:   %[[gep2:.+]] = getelementptr {{\[3 x double\]|i8}}, ptr %[[alloca]], {{i32 0, i32 2|i64 16}}
; CHECK-NEXT:   store double %z, ptr %[[gep2]], align 8
; CHECK-NEXT:   call void @__poseidon_log_value(ptr @poseidon.site.preprocess_test_fma, i64 0, double %[[probed]], i64 3, ptr %[[alloca]])
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_test_fma_0(double %[[fmuladd]], double %[[differet]])
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[grad1:.+]] = fmul fast double %[[grad]], %y
; CHECK-NEXT:   %[[grad2:.+]] = fmul fast double %[[grad]], %x
; CHECK-NEXT:   %[[primal:.+]] = insertvalue { double, double, double, double } undef, double %[[probed]], 0
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double, double, double } %[[primal]], double %[[grad1]], 1
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double, double, double } %[[ins1]], double %[[grad2]], 2
; CHECK-NEXT:   %[[ins3:.+]] = insertvalue { double, double, double, double } %[[ins2]], double %[[grad]], 3
; CHECK-NEXT:   ret { double, double, double, double } %[[ins3]]
; CHECK-NEXT: }

; CHECK: define internal { double, double } @diffepreprocess_muldd(double %x, double %y, double %{{.+}}) #{{[0-9]+}} {
; CHECK:        call void @__poseidon_log_value(ptr @poseidon.site.preprocess_muldd, i64 0, double %{{.+}}, i64 2, ptr %{{.+}})
; CHECK-NEXT:   %[[gradcall:.+]] = call fast { double } @fixgradient___poseidon_probe_preprocess_muldd_0(
; CHECK-NEXT:   %[[grad:.+]] = extractvalue { double } %[[gradcall]], 0
; CHECK-NEXT:   %[[grad1:.+]] = fmul fast double %[[grad]], %y
; CHECK-NEXT:   %[[grad2:.+]] = fmul fast double %[[grad]], %x
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[grad1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[grad2]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]

; CHECK: ![[idx0]] = !{i64 0}
; CHECK: ![[idx1]] = !{i64 1}
; CHECK: ![[idx2]] = !{i64 2}
