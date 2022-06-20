; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

declare <3 x double> @__enzyme_fwddiff(double (double)*, ...)


define dso_local double @fneg(double %x) {
entry:
  %fneg = fneg double %x
  ret double %fneg
}

define dso_local void @fnegd(double %x) {
entry:
  %0 = call <3 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @fneg,  metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 1.0, double 2.5, double 3.0>)
  ret void
}


; CHECK: define internal <3 x double> @fwddiffe3fneg(double %x, <3 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fneg fast <3 x double> %"x'"
; CHECK-NEXT:   ret <3 x double> %0
; CHECK-NEXT: }