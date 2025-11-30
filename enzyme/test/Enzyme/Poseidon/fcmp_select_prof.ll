; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -fpprofile-generate -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -fpprofile-generate -S | FileCheck %s

; CHECK: @ENZYME_FPPROFILE_RUNTIME_VAR = external global i32
; CHECK: @fpprofiled_preprocess_test_maxnum_zero = private unnamed_addr constant [28 x i8] c"preprocess_test_maxnum_zero\00", align 1
; CHECK: @fpprofiled_preprocess_test_maxnum_zero_reversed = private unnamed_addr constant [37 x i8] c"preprocess_test_maxnum_zero_reversed\00", align 1
; CHECK: @fpprofiled_preprocess_test_maxnum_general = private unnamed_addr constant [31 x i8] c"preprocess_test_maxnum_general\00", align 1
; CHECK: @fpprofiled_preprocess_test_minnum_general = private unnamed_addr constant [31 x i8] c"preprocess_test_minnum_general\00", align 1
; CHECK: @fpprofiled_preprocess_test_combined = private unnamed_addr constant [25 x i8] c"preprocess_test_combined\00", align 1

define double @test_maxnum_zero(double %x) {
entry:
  %cmp = fcmp ogt double %x, 0.0
  %result = select i1 %cmp, double %x, double 0.0
  ret double %result
}

; CHECK: define internal { double } @instrtest_maxnum_zero(double %x, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double 0.000000e+00), !enzyme_active !{{[0-9]+}}, !enzyme_fpprofile_idx !{{[0-9]+}}
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_test_maxnum_zero, i64 0, double %[[maxnum]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_test_maxnum_zero, i64 0, double %[[maxnum]], double %[[differet]])
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[sel:.+]] = select fast i1 %[[cmp]], double 0.000000e+00, double %[[differet]]
; CHECK-NEXT:   %[[ins:.+]] = insertvalue { double } undef, double %[[sel]], 0
; CHECK-NEXT:   ret { double } %[[ins]]
; CHECK-NEXT: }

define double @test_maxnum_zero_reversed(double %x) {
entry:
  %cmp = fcmp olt double %x, 0.0
  %result = select i1 %cmp, double 0.0, double %x
  ret double %result
}

; CHECK: define internal { double } @instrtest_maxnum_zero_reversed(double %x, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double 0.000000e+00), !enzyme_active !{{[0-9]+}}, !enzyme_fpprofile_idx !{{[0-9]+}}
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_test_maxnum_zero_reversed, i64 0, double %[[maxnum]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_test_maxnum_zero_reversed, i64 0, double %[[maxnum]], double %[[differet]])
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[sel:.+]] = select fast i1 %[[cmp]], double 0.000000e+00, double %[[differet]]
; CHECK-NEXT:   %[[ins:.+]] = insertvalue { double } undef, double %[[sel]], 0
; CHECK-NEXT:   ret { double } %[[ins]]
; CHECK-NEXT: }

define double @test_maxnum_general(double %x, double %y) {
entry:
  %cmp = fcmp ogt double %x, %y
  %result = select i1 %cmp, double %x, double %y
  ret double %result
}

; CHECK: define internal { double, double } @instrtest_maxnum_general(double %x, double %y, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double %y), !enzyme_active !{{[0-9]+}}, !enzyme_fpprofile_idx !{{[0-9]+}}
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_test_maxnum_general, i64 0, double %[[maxnum]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_test_maxnum_general, i64 0, double %[[maxnum]], double %[[differet]])
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[sel1:.+]] = select fast i1 %[[cmp]], double 0.000000e+00, double %[[differet]]
; CHECK-NEXT:   %[[cmp2:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[sel2:.+]] = select fast i1 %[[cmp2]], double %[[differet]], double 0.000000e+00
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[sel1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[sel2]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]
; CHECK-NEXT: }

define double @test_minnum_general(double %x, double %y) {
entry:
  %cmp = fcmp olt double %x, %y
  %result = select i1 %cmp, double %x, double %y
  ret double %result
}

; CHECK: define internal { double, double } @instrtest_minnum_general(double %x, double %y, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[minnum:.+]] = call double @llvm.minnum.f64(double %x, double %y), !enzyme_active !{{[0-9]+}}, !enzyme_fpprofile_idx !{{[0-9]+}}
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_test_minnum_general, i64 0, double %[[minnum]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_test_minnum_general, i64 0, double %[[minnum]], double %[[differet]])
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[sel1:.+]] = select fast i1 %[[cmp]], double %[[differet]], double 0.000000e+00
; CHECK-NEXT:   %[[cmp2:.+]] = fcmp fast olt double %x, %y
; CHECK-NEXT:   %[[sel2:.+]] = select fast i1 %[[cmp2]], double 0.000000e+00
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[sel1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[sel2]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]
; CHECK-NEXT: }

define double @test_combined(double %x, double %y, double %z) {
entry:
  %cmp1 = fcmp ogt double %x, 0.0
  %max_x = select i1 %cmp1, double %x, double 0.0
  %cmp2 = fcmp olt double %max_x, %y
  %min_xy = select i1 %cmp2, double %max_x, double %y
  %result = fadd fast double %min_xy, %z
  ret double %result
}

; CHECK: define internal { double, double, double } @instrtest_combined(double %x, double %y, double %z, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca0:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[alloca1:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[alloca2:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[maxnum:.+]] = call double @llvm.maxnum.f64(double %x, double 0.000000e+00), !enzyme_active !{{[0-9]+}}, !enzyme_fpprofile_idx !{{[0-9]+}}
; CHECK-NEXT:   store double %x, ptr %[[alloca2]], align 8
; CHECK-NEXT:   %[[gep2:.+]] = getelementptr [2 x double], ptr %[[alloca2]], i32 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, ptr %[[gep2]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_test_combined, i64 0, double %[[maxnum]], i32 2, ptr %[[alloca2]])
; CHECK-NEXT:   %[[minnum:.+]] = call double @llvm.minnum.f64(double %[[maxnum]], double %y), !enzyme_active !{{[0-9]+}}, !enzyme_fpprofile_idx !{{[0-9]+}}
; CHECK-NEXT:   store double %[[maxnum]], ptr %[[alloca1]], align 8
; CHECK-NEXT:   %[[gep1:.+]] = getelementptr [2 x double], ptr %[[alloca1]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep1]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_test_combined, i64 1, double %[[minnum]], i32 2, ptr %[[alloca1]])
; CHECK-NEXT:   %result = fadd fast double %[[minnum]], %z, !enzyme_active !{{[0-9]+}}, !enzyme_fpprofile_idx !{{[0-9]+}}
; CHECK-NEXT:   store double %[[minnum]], ptr %[[alloca0]], align 8
; CHECK-NEXT:   %[[gep0:.+]] = getelementptr [2 x double], ptr %[[alloca0]], i32 0, i32 1
; CHECK-NEXT:   store double %z, ptr %[[gep0]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_test_combined, i64 2, double %result, i32 2, ptr %[[alloca0]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_test_combined, i64 2, double %result, double %[[differet]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_test_combined, i64 1, double %[[minnum]], double %[[differet]])
; CHECK-NEXT:   %[[cmp1:.+]] = fcmp fast olt double %[[maxnum]], %y
; CHECK-NEXT:   %[[sel1:.+]] = select fast i1 %[[cmp1]], double %[[differet]], double 0.000000e+00
; CHECK-NEXT:   %[[cmp2:.+]] = fcmp fast olt double %[[maxnum]], %y
; CHECK-NEXT:   %[[sel2:.+]] = select fast i1 %[[cmp2]], double 0.000000e+00, double %[[differet]]
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_test_combined, i64 0, double %[[maxnum]], double %[[sel1]])
; CHECK-NEXT:   %[[cmp3:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[sel3:.+]] = select fast i1 %[[cmp3]], double 0.000000e+00, double %[[sel1]]
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double, double } undef, double %[[sel3]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double, double } %[[ins1]], double %[[sel2]], 1
; CHECK-NEXT:   %[[ins3:.+]] = insertvalue { double, double, double } %[[ins2]], double %[[differet]], 2
; CHECK-NEXT:   ret { double, double, double } %[[ins3]]
; CHECK-NEXT: }

define double @test_profile_maxnum_zero(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fp_optimize(double (double)* nonnull @test_maxnum_zero, double %x)
  ret double %0
}

define double @test_profile_maxnum_zero_reversed(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fp_optimize(double (double)* nonnull @test_maxnum_zero_reversed, double %x)
  ret double %0
}

define double @test_profile_maxnum(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fp_optimize(double (double, double)* nonnull @test_maxnum_general, double %x, double %y)
  ret double %0
}


define double @test_profile_minnum(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fp_optimize(double (double, double)* nonnull @test_minnum_general, double %x, double %y)
  ret double %0
}

define double @test_profile_combined(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__enzyme_fp_optimize(double (double, double, double)* nonnull @test_combined, double %x, double %y, double %z)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fp_optimize(...)

; CHECK: !0 = !{}
; CHECK: !1 = !{i64 0}
; CHECK: !2 = !{i64 1}
; CHECK: !3 = !{i64 2}