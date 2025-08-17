; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s -dump-input=always

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

; CHECK: define double @test_profile(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[fadd:.+]] = call double @preprocess_tester(double %x, double %y)
; CHECK-NEXT:   ret double %[[fadd]]
; CHECK-NEXT: }

; CHECK: define double @preprocess_tester(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %0 = fadd fast double %x, %y, !enzyme_active !0, !enzyme_fpprofile_idx !1
; CHECK-NEXT:  ret double %0
; CHECK-NEXT: }

; CHECK: !0 = !{}
; CHECK: !1 = !{i64 0}