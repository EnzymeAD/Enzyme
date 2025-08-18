; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -fpprofile-generate -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -fpprofile-generate -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y, double %z) {
entry:
  %0 = fmul fast double %x, %y
  %1 = fadd fast double %0, %z
  ret double %1
}

define double @test_profile(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__enzyme_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fp_optimize(double (double, double, double)*, ...)

; CHECK: @ENZYME_FPPROFILE_RUNTIME_VAR = external global i32
; CHECK: @fpprofiled_preprocess_tester = private unnamed_addr constant [18 x i8] c"preprocess_tester\00", align 1

; CHECK: define internal { double, double, double } @instrtester(double %x, double %y, double %z, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [3 x double], align 8
; CHECK-NEXT:   %[[fmuladd:.+]] = call fast double @llvm.fmuladd.f64(double %x, double %y, double %z), !enzyme_active !0, !enzyme_fpprofile_idx !1
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep1:.+]] = getelementptr [3 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep1]], align 8
; CHECK-NEXT:   %[[gep2:.+]] = getelementptr [3 x double], ptr %[[alloca]], i32 0, i32 2
; CHECK-NEXT:   store double %z, ptr %[[gep2]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_tester, i64 0, double %[[fmuladd]], i32 3, ptr %[[alloca]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_tester, i64 0, double %[[fmuladd]], double %[[differet]])
; CHECK-NEXT:   %[[grad1:.+]] = fmul fast double %[[differet]], %y
; CHECK-NEXT:   %[[grad2:.+]] = fmul fast double %[[differet]], %x
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double, double } undef, double %[[grad1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double, double } %[[ins1]], double %[[grad2]], 1
; CHECK-NEXT:   %[[ins3:.+]] = insertvalue { double, double, double } %[[ins2]], double %[[differet]], 2
; CHECK-NEXT:   ret { double, double, double } %[[ins3]]
; CHECK-NEXT: }

; CHECK: !0 = !{}
; CHECK: !1 = !{i64 0}