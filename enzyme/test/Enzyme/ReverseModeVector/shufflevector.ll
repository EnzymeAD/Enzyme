; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { [2 x <3 x double>], [2 x <3 x double>] }

declare %struct.Gradients @__enzyme_autodiff(...)

define <4 x double> @square(<3 x double> %x, <3 x double> %y) {
entry:
  %r = shufflevector <3 x double> %x, <3 x double> %y, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x double> %r
}

define %struct.Gradients @dsquare(<3 x double> %x, <3 x double> %y) {
entry:
  %0 = tail call %struct.Gradients (...) @__enzyme_autodiff(<4 x double> (<3 x double>, <3 x double>)* nonnull @square, metadata !"enzyme_width", i64 2, <3 x double> %x, <3 x double> %y)
  ret %struct.Gradients %0
}

; CHECK: define internal { [2 x <3 x double>], [2 x <3 x double>] } @diffe2square(<3 x double> %x, <3 x double> %y, [2 x <4 x double>] %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'de" = alloca [2 x <3 x double>]
; CHECK-NEXT:   store [2 x <3 x double>] zeroinitializer, [2 x <3 x double>]* %"x'de"
; CHECK-NEXT:   %"y'de" = alloca [2 x <3 x double>]
; CHECK-NEXT:   store [2 x <3 x double>] zeroinitializer, [2 x <3 x double>]* %"y'de"
; CHECK-NEXT:   %[[i0:.+]] = extractvalue [2 x <4 x double>] %differeturn, 0
; CHECK-NEXT:   %[[i1:.+]] = extractelement <4 x double> %[[i0]], i64 0
; CHECK-NEXT:   %[[i2:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"x'de", i32 0, i32 0, i32 0
; CHECK-NEXT:   %[[i3:.+]] = load double, double* %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = fadd fast double %[[i3]], %[[i1]]
; CHECK-NEXT:   store double %[[i4]], double* %[[i2]]
; CHECK-NEXT:   %[[i5:.+]] = extractvalue [2 x <4 x double>] %differeturn, 1
; CHECK-NEXT:   %[[i6:.+]] = extractelement <4 x double> %[[i5]], i64 0
; CHECK-NEXT:   %[[i7:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"x'de", i32 0, i32 1, i32 0
; CHECK-NEXT:   %[[i8:.+]] = load double, double* %[[i7]]
; CHECK-NEXT:   %[[i9:.+]] = fadd fast double %[[i8]], %[[i6]]
; CHECK-NEXT:   store double %[[i9]], double* %[[i7]]
; CHECK-NEXT:   %[[i10:.+]] = extractelement <4 x double> %[[i0]], i64 1
; CHECK-NEXT:   %[[i11:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"x'de", i32 0, i32 0, i32 1
; CHECK-NEXT:   %[[i12:.+]] = load double, double* %[[i11]]
; CHECK-NEXT:   %[[i13:.+]] = fadd fast double %[[i12]], %[[i10]]
; CHECK-NEXT:   store double %[[i13]], double* %[[i11]]
; CHECK-NEXT:   %[[i14:.+]] = extractelement <4 x double> %[[i5]], i64 1
; CHECK-NEXT:   %[[i15:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"x'de", i32 0, i32 1, i32 1
; CHECK-NEXT:   %[[i16:.+]] = load double, double* %[[i15]]
; CHECK-NEXT:   %[[i17:.+]] = fadd fast double %[[i16]], %[[i14]]
; CHECK-NEXT:   store double %[[i17]], double* %[[i15]]
; CHECK-NEXT:   %[[i18:.+]] = extractelement <4 x double> %[[i0]], i64 2
; CHECK-NEXT:   %[[i19:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"y'de", i32 0, i32 0, i32 1
; CHECK-NEXT:   %[[i20:.+]] = load double, double* %[[i19]]
; CHECK-NEXT:   %[[i21:.+]] = fadd fast double %[[i20]], %[[i18]]
; CHECK-NEXT:   store double %[[i21]], double* %[[i19]]
; CHECK-NEXT:   %[[i22:.+]] = extractelement <4 x double> %[[i5]], i64 2
; CHECK-NEXT:   %[[i23:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"y'de", i32 0, i32 1, i32 1
; CHECK-NEXT:   %[[i24:.+]] = load double, double* %[[i23]]
; CHECK-NEXT:   %[[i25:.+]] = fadd fast double %[[i24]], %[[i22]]
; CHECK-NEXT:   store double %[[i25]], double* %[[i23]]
; CHECK-NEXT:   %[[i26:.+]] = extractelement <4 x double> %[[i0]], i64 3
; CHECK-NEXT:   %[[i27:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"y'de", i32 0, i32 0, i32 2
; CHECK-NEXT:   %[[i28:.+]] = load double, double* %[[i27]]
; CHECK-NEXT:   %[[i29:.+]] = fadd fast double %[[i28]], %[[i26]]
; CHECK-NEXT:   store double %[[i29]], double* %[[i27]]
; CHECK-NEXT:   %[[i30:.+]] = extractelement <4 x double> %[[i5]], i64 3
; CHECK-NEXT:   %[[i31:.+]] = getelementptr inbounds [2 x <3 x double>], [2 x <3 x double>]* %"y'de", i32 0, i32 1, i32 2
; CHECK-NEXT:   %[[i32:.+]] = load double, double* %[[i31]]
; CHECK-NEXT:   %[[i33:.+]] = fadd fast double %[[i32]], %[[i30]]
; CHECK-NEXT:   store double %[[i33]], double* %[[i31]]
; CHECK-NEXT:   %[[i34:.+]] = load [2 x <3 x double>], [2 x <3 x double>]* %"x'de"
; CHECK-NEXT:   %[[i35:.+]] = load [2 x <3 x double>], [2 x <3 x double>]* %"y'de"
; CHECK-NEXT:   %[[i36:.+]] = insertvalue { [2 x <3 x double>], [2 x <3 x double>] } {{(undef|poison)}}, [2 x <3 x double>] %[[i34]], 0
; CHECK-NEXT:   %[[i37:.+]] = insertvalue { [2 x <3 x double>], [2 x <3 x double>] } %[[i36]], [2 x <3 x double>] %[[i35]], 1
; CHECK-NEXT:   ret { [2 x <3 x double>], [2 x <3 x double>] } %[[i37]]
; CHECK-NEXT: }
