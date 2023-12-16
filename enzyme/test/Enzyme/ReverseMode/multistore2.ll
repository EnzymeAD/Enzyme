; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

declare i1 @cmp()

define void @f({ double, i1 }* %y, double %x, i1 %z) {
entry:
  %ins = insertvalue { double, i1 } undef, double %x, 0
  %ins2 = insertvalue { double, i1 } %ins, i1 %z, 1
  store { double, i1 } %ins2, { double, i1 }* %y
  ret void
}

declare double @__enzyme_autodiff(...)

define double @test({ double, i32 }* %y, { double, i32 }* %dy, double %x, i1 %z) {
entry:
  %r = call double (...) @__enzyme_autodiff(void ({ double, i1 }*, double, i1)* @f, metadata !"enzyme_dup", { double, i32 }* %y, { double, i32 }* %dy, double %x, i1 %z)
  ret double %r
}

; CHECK: define internal { double } @diffef({ double, i1 }* %y, { double, i1 }* %"y'", double %x, i1 %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { double, i1 }, align 8
; CHECK-NEXT:   %"ins2'de" = alloca { double, i1 }, align 8
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"ins2'de", align 8
; CHECK-NEXT:   %"ins'de" = alloca { double, i1 }, align 8
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"ins'de", align 8
; CHECK-NEXT:   %ins = insertvalue { double, i1 } undef, double %x, 0
; CHECK-NEXT:   %ins2 = insertvalue { double, i1 } %ins, i1 %z, 1
; CHECK-NEXT:   store { double, i1 } %ins2, { double, i1 }* %y, align 8
; CHECK-NEXT:   %[[i0:.+]] = load { double, i1 }, { double, i1 }* %"y'", align 8
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %0, align 8

; CHECK-NEXT:   %2 = bitcast { double, i1 }* %"y'" to i64*
; CHECK-NEXT:   %3 = bitcast { double, i1 }* %0 to i64*
; CHECK-NEXT:   %4 = load i64, i64* %3, align 4
; CHECK-NEXT:   store i64 %4, i64* %2, align 8

; CHECK-NEXT:   %[[i2:.+]] = extractvalue { double, i1 } %[[i0]], 0
; CHECK-NEXT:   %[[i3:.+]] = getelementptr inbounds { double, i1 }, { double, i1 }* %"ins2'de", i32 0, i32 0
; CHECK-NEXT:   %[[i4:.+]] = load double, double* %[[i3]], align 8
; CHECK-NEXT:   %[[i5:.+]] = fadd fast double %[[i4]], %[[i2]]
; CHECK-NEXT:   store double %[[i5]], double* %[[i3]], align 8
; CHECK-NEXT:   %[[i6:.+]] = load { double, i1 }, { double, i1 }* %"ins2'de", align 8
; CHECK-NEXT:   %[[i8:.+]] = extractvalue { double, i1 } %[[i6]], 0
; CHECK-NEXT:   %[[i9:.+]] = getelementptr inbounds { double, i1 }, { double, i1 }* %"ins'de", i32 0, i32 0
; CHECK-NEXT:   %[[i10:.+]] = load double, double* %[[i9]], align 8
; CHECK-NEXT:   %[[i11:.+]] = fadd fast double %[[i10]], %[[i8]]
; CHECK-NEXT:   store double %[[i11]], double* %[[i9]], align 8
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"ins2'de", align 8
; CHECK-NEXT:   %[[i12:.+]] = load { double, i1 }, { double, i1 }* %"ins'de", align 8
; CHECK-NEXT:   %[[i13:.+]] = extractvalue { double, i1 } %[[i12]], 0
; CHECK-NEXT:   %[[i14:.+]] = fadd fast double 0.000000e+00, %[[i13]]
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"ins'de", align 8
; CHECK-NEXT:   %[[i15:.+]] = insertvalue { double } undef, double %[[i14]], 0
; CHECK-NEXT:   ret { double } %[[i15]]
; CHECK-NEXT: }
