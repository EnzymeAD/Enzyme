; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double,double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fdiv fast double %x, %y
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 0.0, double 1.0, double %y, double 1.0, double 0.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fdiv nnan ninf nsz arcp contract afn double %x, %y
; CHECK-NEXT:   %[[i1:.+]] = extractvalue [2 x double] %"y'", 0
; CHECK-NEXT:   %[[i2:.+]] = fmul nnan ninf nsz arcp contract afn double %[[i1]], %[[i0]]
; CHECK-NEXT:   %[[i3:.+]] = extractvalue [2 x double] %"y'", 1
; CHECK-NEXT:   %[[i4:.+]] = fmul nnan ninf nsz arcp contract afn double %[[i3]], %[[i0]]
; CHECK-NEXT:   %[[i5:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %[[i6:.+]] = fsub nnan ninf nsz arcp contract afn double %[[i5]], %[[i2]]
; CHECK-NEXT:   %[[i7:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %[[i8:.+]] = fsub nnan ninf nsz arcp contract afn double %[[i7]], %[[i4]]
; CHECK-NEXT:   %[[i9:.+]] = fdiv nnan ninf nsz arcp contract afn double %[[i6]], %y
; CHECK-NEXT:   %[[i10:.+]] = insertvalue [2 x double] undef, double %[[i9]], 0
; CHECK-NEXT:   %[[i11:.+]] = fdiv nnan ninf nsz arcp contract afn double %[[i8]], %y
; CHECK-NEXT:   %[[i12:.+]] = insertvalue [2 x double] %[[i10]], double %[[i11]], 1
; CHECK-NEXT:   ret [2 x double] %[[i12]]
; CHECK-NEXT: }
