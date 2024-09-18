; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

define void @f({ double, double* }* "enzyme_type"="{[-1,0]:Float@double, [-1,8]:Pointer}" %out, double %in, double* "enzyme_type"="{[-1]:Pointer}" %in2) {
entry:
  %ins1 = insertvalue { double, double* } undef, double %in, 0
  %ins2 = insertvalue { double, double* } %ins1, double* %in2, 1
  store { double, double* } %ins2, { double, double* }* %out
  ret void
}

define void @test_derivative(double* %x, double* %dx, double* %y, double* %dy) {
entry:
  call void (...) @__enzyme_augmentfwd(void ({ double, double* }*, double, double*)* nonnull @f, metadata !"enzyme_dup", i8* null, i8* null, double 1.0, metadata !"enzyme_dup", i8* null, i8* null)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_augmentfwd(...)


; CHECK: define internal i8* @augmented_f({ double, double* }* "enzyme_type"="{[-1,0]:Float@double, [-1,8]:Pointer}" %out, { double, double* }* "enzyme_type"="{[-1,0]:Float@double, [-1,8]:Pointer}" %"out'", double %in, double* "enzyme_type"="{[-1]:Pointer}" %in2, double* "enzyme_type"="{[-1]:Pointer}" %"in2'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = alloca { double, double* }
; CHECK-NEXT:   %ins1 = insertvalue { double, double* } {{(undef|poison)}}, double %in, 0
; CHECK-NEXT:   %"ins2'ipiv" = insertvalue { double, double* } { double 0.000000e+00, double* undef }, double* %"in2'", 1
; CHECK-NEXT:   %ins2 = insertvalue { double, double* } %ins1, double* %in2, 1
; CHECK-NEXT:   store { double, double* } %"ins2'ipiv", { double, double* }* %[[i0]]
; CHECK-NEXT:   %[[i1:.+]] = bitcast { double, double* }* %"out'" to i8*
; CHECK-NEXT:   %[[i2:.+]] = getelementptr inbounds i8, i8* %[[i1]], i64 8
; CHECK-NEXT:   %[[i3:.+]] = bitcast { double, double* }* %[[i0]] to i8*
; CHECK-NEXT:   %[[i4:.+]] = getelementptr inbounds i8, i8* %[[i3]], i64 8
; CHECK-NEXT:   %[[i5:.+]] = bitcast i8* %[[i2]] to i64*
; CHECK-NEXT:   %[[i6:.+]] = bitcast i8* %[[i4]] to i64*
; CHECK-NEXT:   %[[i7:.+]] = load i64, i64* %[[i6]]
; CHECK-NEXT:   store i64 %[[i7]], i64* %[[i5]]
; CHECK-NEXT:   store { double, double* } %ins2, { double, double* }* %out
; CHECK-NEXT:   ret i8* null
; CHECK-NEXT: }
