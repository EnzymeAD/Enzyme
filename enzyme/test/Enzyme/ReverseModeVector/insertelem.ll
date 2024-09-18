; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { [2 x <3 x double>], [2 x double] }

declare %struct.Gradients @__enzyme_autodiff(...)

define <3 x double> @square(<3 x double> %x, double %y) {
entry:
  %r = insertelement <3 x double> %x, double %y, i64 2
  ret <3 x double> %r
}

define %struct.Gradients @dsquare(<3 x double> %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (...) @__enzyme_autodiff(<3 x double> (<3 x double>, double)* nonnull @square, metadata !"enzyme_width", i64 2, <3 x double> %x, double %y)
  ret %struct.Gradients %0
}

; CHECK: define internal { [2 x <3 x double>], [2 x double] } @diffe2square(<3 x double> %x, double %y, [2 x <3 x double>] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca [2 x <3 x double>]
; CHECK-NEXT:   store [2 x <3 x double>] zeroinitializer, [2 x <3 x double>]* %"x'de"
; CHECK-NEXT:   %"y'de" = alloca [2 x double]
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"y'de"
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [2 x <3 x double>] %differeturn, 0
; CHECK-NEXT:   %[[i1:.+]] = insertelement <3 x double> %[[i0]], double 0.000000e+00, i64 2
; CHECK-NEXT:   %[[i2:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"x'de", i32 0, i64 0
; CHECK-NEXT:   %[[i3:.+]] = load <3 x double>, <3 x double>* %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = fadd fast <3 x double> %[[i3]], %[[i1]]
; CHECK-NEXT:   store <3 x double> %[[i4]], <3 x double>* %[[i2]]
; CHECK-NEXT:   %[[i5:.+]] = extractvalue [2 x <3 x double>] %differeturn, 1
; CHECK-NEXT:   %[[i6:.+]] = insertelement <3 x double> %[[i5]], double 0.000000e+00, i64 2
; CHECK-NEXT:   %[[i7:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"x'de", i32 0, i64 1
; CHECK-NEXT:   %[[i8:.+]] = load <3 x double>, <3 x double>* %[[i7]]
; CHECK-NEXT:   %[[i9:.+]] = fadd fast <3 x double> %[[i8]], %[[i6]]
; CHECK-NEXT:   store <3 x double> %[[i9]], <3 x double>* %[[i7]]
; CHECK-NEXT:   %[[i10:.+]] = extractelement <3 x double> %[[i0]], i64 2
; CHECK-NEXT:   %[[i11:.+]] = getelementptr inbounds [2 x double], [2 x double]* %"y'de", i32 0, i64 0
; CHECK-NEXT:   %[[i12:.+]] = load double, double* %[[i11]]
; CHECK-NEXT:   %[[i13:.+]] = fadd fast double %[[i12]], %[[i10]]
; CHECK-NEXT:   store double %[[i13]], double* %[[i11]]
; CHECK-NEXT:   %[[i14:.+]] = extractelement <3 x double> %[[i5]], i64 2
; CHECK-NEXT:   %[[i15:.+]] = getelementptr inbounds [2 x double], [2 x double]* %"y'de", i32 0, i64 1
; CHECK-NEXT:   %[[i16:.+]] = load double, double* %[[i15]]
; CHECK-NEXT:   %[[i17:.+]] = fadd fast double %[[i16]], %[[i14]]
; CHECK-NEXT:   store double %[[i17]], double* %[[i15]]
; CHECK-NEXT:   %[[i18:.+]] = load [2 x <3 x double>], [2 x <3 x double>]* %"x'de"
; CHECK-NEXT:   %[[i19:.+]] = load [2 x double], [2 x double]* %"y'de"
; CHECK-NEXT:   %[[i20:.+]] = insertvalue { [2 x <3 x double>], [2 x double] } {{(undef|poison)}}, [2 x <3 x double>] %[[i18]], 0
; CHECK-NEXT:   %[[i21:.+]] = insertvalue { [2 x <3 x double>], [2 x double] } %[[i20]], [2 x double] %[[i19]], 1
; CHECK-NEXT:   ret { [2 x <3 x double>], [2 x double] } %[[i21]]
; CHECK-NEXT: }
