; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -fpprofile-generate -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -fpprofile-generate -S | FileCheck %s

; Adapted from enzyme/test/Enzyme/ReverseMode/gradient-struct-ret.ll

%struct.Gradients = type { double, double }

; Function Attrs: noinline nounwind readnone uwtable
define dso_local double @muldd(double %x, double %y) {
entry:
  %mul = fmul fast double %x, %y
  ret double %mul
}

define dso_local %struct.Gradients @test_profile(double %x, double %y) {
entry:
  %call = call %struct.Gradients (i8*, ...) @__enzyme_fp_optimize(i8* bitcast (double (double, double)* @muldd to i8*), double %x, double %y, double 1.0e-6)
  ret %struct.Gradients %call
}

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fp_optimize(i8*, ...)

; CHECK: @ENZYME_FPPROFILE_RUNTIME_VAR = external global i32
; CHECK: @fpprofiled_preprocess_muldd = private unnamed_addr constant [17 x i8] c"preprocess_muldd\00", align 1

; CHECK: define internal { double, double } @instrmuldd(double %x, double %y, double %[[differet:.+]]) #{{[0-9]+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[mul:.+]] = fmul fast double %x, %y, !enzyme_active !0, !enzyme_fpprofile_idx !1
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr @fpprofiled_preprocess_muldd, i64 0, double %[[mul]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr @fpprofiled_preprocess_muldd, i64 0, double %[[mul]], double %[[differet]])
; CHECK-NEXT:   %[[grad1:.+]] = fmul fast double %[[differet]], %y
; CHECK-NEXT:   %[[grad2:.+]] = fmul fast double %[[differet]], %x
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[grad1]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[grad2]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]
; CHECK-NEXT: }

; CHECK: !0 = !{}
; CHECK: !1 = !{i64 0}