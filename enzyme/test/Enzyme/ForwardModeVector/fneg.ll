; RUN: if [ %llvmver -ge 10 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 10 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

%struct.Gradients = type { double, double, double }

declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)


define dso_local double @fneg(double %x) {
entry:
  %fneg = fneg double %x
  ret double %fneg
}

define dso_local void @fnegd(double %x) {
entry:
  %0 = call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @fneg,  metadata !"enzyme_width", i64 3, double %x, double 1.0, double 2.5, double 3.0)
  ret void
}


; CHECK: define internal [3 x double] @fwddiffe3fneg(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[i1:.+]] = fneg fast double %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[i4:.+]] = fneg fast double %[[i3]]
; CHECK-NEXT:   %[[i6:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[i7:.+]] = fneg fast double %[[i6]]
; CHECK-NEXT:   %[[i2:.+]] = insertvalue [3 x double] undef, double %[[i1]], 0
; CHECK-NEXT:   %[[i5:.+]] = insertvalue [3 x double] %[[i2]], double %[[i4]], 1
; CHECK-NEXT:   %[[i8:.+]] = insertvalue [3 x double] %[[i5]], double %[[i7]], 2
; CHECK-NEXT:   ret [3 x double] %[[i8]]
; CHECK-NEXT: }
