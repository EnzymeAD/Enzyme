; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

declare i1 @cmp()

define void @f({ double, i32 }* %y, double %x, i32 %z) {
entry:
  %ins = insertvalue { double, i32 } undef, i32 %z, 1
  %ins2 = insertvalue { double, i32 } %ins, double %x, 0
  %e = icmp eq i32 %z, 12
  store { double, i32 } %ins2, { double, i32 }* %y
  ret void
}

declare double @__enzyme_autodiff(...)

define double @test({ double, i32 }* %y, { double, i32 }* %dy, double %x, i32 %z) {
entry:
  %r = call double (...) @__enzyme_autodiff(void ({ double, i32 }*, double, i32)* @f, metadata !"enzyme_dup", { double, i32 }* %y, { double, i32 }* %dy, double %x, i32 %z)
  ret double %r
}

; CHECK: define internal { double } @diffef({ double, i32 }* %y, { double, i32 }* %"y'", double %x, i32 %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = alloca { double, i32 }
; CHECK-NEXT:   %"ins2'de" = alloca { double, i32 }
; CHECK-NEXT:   store { double, i32 } zeroinitializer, { double, i32 }* %"ins2'de"
; CHECK-NEXT:   %[[i1:.+]] = alloca { double, i32 }
; CHECK-NEXT:   %"ins'ipiv" = insertvalue { double, i32 } undef, i32 %z, 1
; CHECK-NEXT:   %ins = insertvalue { double, i32 } undef, i32 %z, 1
; CHECK-NEXT:   %"ins2'ipiv" = insertvalue { double, i32 } %"ins'ipiv", double 0.000000e+00, 0
; CHECK-NEXT:   %ins2 = insertvalue { double, i32 } %ins, double %x, 0
; CHECK-NEXT:   store { double, i32 } %"ins2'ipiv", { double, i32 }* %[[i1]]
; CHECK-NEXT:   %[[i2:.+]] = bitcast { double, i32 }* %"y'" to i8*
; CHECK-NEXT:   %[[i3:.+]] = getelementptr inbounds i8, i8* %[[i2]], i64 8
; CHECK-NEXT:   %[[i4:.+]] = bitcast { double, i32 }* %[[i1]] to i8*
; CHECK-NEXT:   %[[i5:.+]] = getelementptr inbounds i8, i8* %[[i4]], i64 8
; CHECK-NEXT:   %[[i6:.+]] = bitcast i8* %[[i3]] to i64*
; CHECK-NEXT:   %[[i7:.+]] = bitcast i8* %[[i5]] to i64*
; CHECK-NEXT:   %[[i8:.+]] = load i64, i64* %[[i7]]
; CHECK-NEXT:   store i64 %[[i8]], i64* %[[i6]]
; CHECK-NEXT:   store { double, i32 } %ins2, { double, i32 }* %y
; CHECK-NEXT:   %[[i9:.+]] = load { double, i32 }, { double, i32 }* %"y'"
; CHECK-NEXT:   store { double, i32 } zeroinitializer, { double, i32 }* %[[i0]]
; CHECK-NEXT:   %[[i10:.+]] = bitcast { double, i32 }* %"y'" to i64*
; CHECK-NEXT:   %[[i11:.+]] = bitcast { double, i32 }* %[[i0]] to i64*
; CHECK-NEXT:   %[[i12:.+]] = load i64, i64* %[[i11]]
; CHECK-NEXT:   store i64 %[[i12]], i64* %[[i10]]
; CHECK-NEXT:   %[[i13:.+]] = extractvalue { double, i32 } %[[i9]], 0
; CHECK-NEXT:   %[[i14:.+]] = getelementptr inbounds { double, i32 }, { double, i32 }* %"ins2'de", i32 0, i32 0
; CHECK-NEXT:   %[[i15:.+]] = load double, double* %[[i14]]
; CHECK-NEXT:   %[[i16:.+]] = fadd fast double %[[i15]], %[[i13]]
; CHECK-NEXT:   store double %[[i16]], double* %[[i14]]
; CHECK-NEXT:   %[[i17:.+]] = load { double, i32 }, { double, i32 }* %"ins2'de", align 8
; CHECK-NEXT:   %[[i18:.+]] = extractvalue { double, i32 } %[[i17]], 0
; CHECK-NEXT:   %[[i19:.+]] = fadd fast double 0.000000e+00, %[[i18]]
; CHECK-NEXT:   s[[ito:.+]]re { double, i32 } zeroinitializer, { double, i32 }* %"ins2'de", align 8
; CHECK-NEXT:   %[[i20:.+]] = insertvalue { double } undef, double %[[i19]], 0
; CHECK-NEXT:   ret { double } %[[i20]]
; CHECK-NEXT: }

