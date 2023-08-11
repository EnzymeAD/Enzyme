; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse,adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

define double @square(double %x) {
entry:
  %mul = fmul fast double %x, %x
  ret double %mul
}

define %struct.Gradients @dsquare(double %x) {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @square, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 10.0, double 100.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [3 x double] @fwddiffe3square(double %x, [3 x double] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %[[i0]], %x
; CHECK-NEXT:   %[[i4:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i4]], %x
; CHECK-NEXT:   %[[i8:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[i9:.+]] = fmul fast double %[[i8]], %x
; CHECK-NEXT:   %[[i2:.+]] = fadd fast double %[[i1]], %[[i1]]
; CHECK-NEXT:   %[[i3:.+]] = insertvalue [3 x double] undef, double %[[i2]], 0
; CHECK-NEXT:   %[[i6:.+]] = fadd fast double %[[i5]], %[[i5]]
; CHECK-NEXT:   %[[i7:.+]] = insertvalue [3 x double] %[[i3]], double %[[i6]], 1
; CHECK-NEXT:   %[[i10:.+]] = fadd fast double %[[i9]], %[[i9]]
; CHECK-NEXT:   %[[i11:.+]] = insertvalue [3 x double] %[[i7]], double %[[i10]], 2
; CHECK-NEXT:   ret [3 x double] %[[i11]]
; CHECK-NEXT: }