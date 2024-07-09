; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { [2 x <3 x double>] }

declare %struct.Gradients @__enzyme_autodiff(...)

define double @square(<3 x double> %x) {
entry:
  %r = extractelement <3 x double> %x, i64 2
  ret double %r
}

define %struct.Gradients @dsquare(<3 x double> %x) {
entry:
  %0 = tail call %struct.Gradients (...) @__enzyme_autodiff(double (<3 x double>)* nonnull @square, metadata !"enzyme_width", i64 2, <3 x double> %x)
  ret %struct.Gradients %0
}

; CHECK: define internal { [2 x <3 x double>] } @diffe2square(<3 x double> %x, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca [2 x <3 x double>]
; CHECK-NEXT:   store [2 x <3 x double>] zeroinitializer, [2 x <3 x double>]* %"x'de"
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   %[[i1:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"x'de", i32 0, i64 0, i64 2
; CHECK-NEXT:   %[[i2:.+]] = load double, double* %[[i1]]
; CHECK-NEXT:   %[[i3:.+]] = fadd fast double %[[i2]], %[[i0]]
; CHECK-NEXT:   store double %[[i3]], double* %[[i1]]
; CHECK-NEXT:   %[[i4:.+]] = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   %[[i5:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"x'de", i32 0, i64 1, i64 2
; CHECK-NEXT:   %[[i6:.+]] = load double, double* %[[i5]]
; CHECK-NEXT:   %[[i7:.+]] = fadd fast double %[[i6]], %[[i4]]
; CHECK-NEXT:   store double %[[i7]], double* %[[i5]]
; CHECK-NEXT:   %[[i8:.+]] = load [2 x <3 x double>], [2 x <3 x double>]* %"x'de"
; CHECK-NEXT:   %[[i9:.+]] = insertvalue { [2 x <3 x double>] } {{(undef|poison)}}, [2 x <3 x double>] %[[i8]], 0
; CHECK-NEXT:   ret { [2 x <3 x double>] } %[[i9]]
; CHECK-NEXT: }
