; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -fpprofile-generate -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -fpprofile-generate -S | FileCheck %s -dump-input=always

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

define double @test_profile(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fp_optimize(double (double, double)* nonnull @tester, double %x, double %y, double 1.0e-6)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fp_optimize(double (double, double)*, ...)

; CHECK: define internal { double, double } @instrtester(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[alloca:.+]] = alloca [2 x double], align 8
; CHECK-NEXT:   %[[fadd:.+]] = fadd fast double %x, %y, !enzyme_active !0, !enzyme_preprocess_origin !1
; CHECK-NEXT:   store double %x, ptr %[[alloca]], align 8
; CHECK-NEXT:   %[[gep:.+]] = getelementptr [2 x double], ptr %[[alloca]], i32 0, i32 1
; CHECK-NEXT:   store double %y, ptr %[[gep]], align 8
; CHECK-NEXT:   call void @enzymeLogValue(ptr {{.*}}, double %[[fadd]], i32 2, ptr %[[alloca]])
; CHECK-NEXT:   call void @enzymeLogGrad(ptr {{.*}}, double %[[fadd]], double %[[differet]])
; CHECK-NEXT:   %[[ins1:.+]] = insertvalue { double, double } undef, double %[[differet]], 0
; CHECK-NEXT:   %[[ins2:.+]] = insertvalue { double, double } %[[ins1]], double %[[differet]], 1
; CHECK-NEXT:   ret { double, double } %[[ins2]]
; CHECK-NEXT: }