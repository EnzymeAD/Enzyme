; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -inline -enzyme -preserve-nvvm-end -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,inline,enzyme,preserve-nvvm-end,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

declare float @_ZL4cbrtf(float)
declare float @_ZL8copysignff(float, float)
declare float @_ZL4fminff(float, float)
declare float @_ZL4fmaxff(float, float)
declare float @_ZL3powfi(float, i32)
declare float @_ZL4rintf(float)
declare float @_ZL9nearbyintf(float)
declare i64 @_ZL5lrintf(float)
declare float @_ZL6myfuncf(float)

define internal float @__nv_cbrtf(float %a) #0 {
  ret float %a
}

define internal float @__nv_copysignf(float %a, float %b) #0 {
  ret float %a
}

define internal float @__nv_fminf(float %a, float %b) #0 {
  ret float %a
}

define internal float @__nv_fmaxf(float %a, float %b) #0 {
  ret float %a
}

define internal float @__nv_powif(float %a, i32 %b) #0 {
  ret float %a
}

define internal float @__nv_floorf(float %a) #0 {
  ret float %a
}

define internal i64 @_ZL6lrintff(float %a) #0 {
  %r = fptosi float %a to i64
  ret i64 %r
}

define float @tester_cbrt(float %x) {
  %r = call float @_ZL4cbrtf(float %x)
  ret float %r
}

define float @tester_copysign(float %x) {
  %r = call float @_ZL8copysignff(float %x, float -1.000000e+00)
  ret float %r
}

define float @tester_fmin(float %x) {
  %r = call float @_ZL4fminff(float %x, float 5.000000e-01)
  ret float %r
}

define float @tester_fmax(float %x) {
  %r = call float @_ZL4fmaxff(float %x, float 5.000000e-01)
  ret float %r
}

define float @tester_powi(float %x, i32 %n) {
  %r = call float @_ZL3powfi(float %x, i32 %n)
  ret float %r
}

define float @tester_rint_nearbyint(float %x, float %y) {
  %a = call float @_ZL4rintf(float %x)
  %b = call float @_ZL9nearbyintf(float %y)
  %c = call i64 @_ZL5lrintf(float %a)
  %d = sitofp i64 %c to float
  %e = call float @__nv_floorf(float %b)
  %sum0 = fadd float %d, %e
  %sum1 = fadd float %x, %sum0
  ret float %sum1
}

define float @untouched_user_decl(float %x) {
  %r = call float @_ZL6myfuncf(float %x)
  ret float %r
}

declare float @__enzyme_autodiff(float (float)*, ...)
declare float @__enzyme_autodiff_i(float (float, i32)*, ...)
declare { float, float } @__enzyme_autodiff_ff(float (float, float)*, ...)

define float @d_cbrt(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester_cbrt, float %x)
  ret float %r
}

define float @d_copysign(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester_copysign, float %x)
  ret float %r
}

define float @d_fmin(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester_fmin, float %x)
  ret float %r
}

define float @d_fmax(float %x) {
  %r = call float (float (float)*, ...) @__enzyme_autodiff(float (float)* nonnull @tester_fmax, float %x)
  ret float %r
}

define float @d_powi(float %x, i32 %n) {
  %r = call float (float (float, i32)*, ...) @__enzyme_autodiff_i(float (float, i32)* nonnull @tester_powi, float %x, i32 %n)
  ret float %r
}

define { float, float } @d_rint_nearbyint(float %x, float %y) {
  %r = call { float, float } (float (float, float)*, ...) @__enzyme_autodiff_ff(float (float, float)* nonnull @tester_rint_nearbyint, float %x, float %y)
  ret { float, float } %r
}

attributes #0 = { alwaysinline nounwind }

; CHECK-NOT: @_ZL4cbrtf
; CHECK-NOT: @_ZL8copysignff
; CHECK-NOT: @_ZL4fminff
; CHECK-NOT: @_ZL4fmaxff
; CHECK-NOT: @_ZL3powfi
; CHECK-NOT: @_ZL4rintf
; CHECK-NOT: @_ZL9nearbyintf
; CHECK-NOT: @_ZL5lrintf

; CHECK: declare float @_ZL6myfuncf(float)

; CHECK: define float @tester_rint_nearbyint(float %x, float %y)
; CHECK:   %[[RINT:.+]] = call float @llvm.rint.f32(float %x)
; CHECK:   %[[NINT:.+]] = call float @llvm.nearbyint.f32(float %y)

; CHECK: define internal { float } @diffetester_cbrt(float %x, float %differeturn)
; CHECK:   %[[C0:.+]] = call fast float @__nv_cbrtf(float %x)
; CHECK-NEXT:   %[[C1:.+]] = fmul fast float %differeturn, %[[C0]]
; CHECK-NEXT:   %[[C2:.+]] = fmul fast float 3.000000e+00, %x
; CHECK-NEXT:   %[[C3:.+]] = fdiv fast float %[[C1]], %[[C2]]
; CHECK-NEXT:   %[[C4:.+]] = insertvalue { float } undef, float %[[C3]], 0
; CHECK-NEXT:   ret { float } %[[C4]]

; CHECK: define internal { float } @diffetester_copysign(float %x, float %differeturn)
; CHECK:   %[[S0:.+]] = call fast float @__nv_copysignf(float 1.000000e+00, float %x)
; CHECK:   %[[S1:.+]] = call fast float @__nv_copysignf(float 1.000000e+00, float -1.000000e+00)
; CHECK:   %[[S2:.+]] = fmul fast float %[[S0]], %[[S1]]
; CHECK:   %[[S3:.+]] = fmul fast float %differeturn, %[[S2]]
; CHECK:   %[[S4:.+]] = insertvalue { float } undef, float %[[S3]], 0
; CHECK:   ret { float } %[[S4]]

; CHECK: define internal { float } @diffetester_fmin(float %x, float %differeturn)
; CHECK:   %[[MN0:.+]] = fcmp fast olt float %x, 5.000000e-01
; CHECK:   %[[MN1:.+]] = select fast i1 %[[MN0]], float %differeturn, float 0.000000e+00
; CHECK:   %[[MN2:.+]] = insertvalue { float } undef, float %[[MN1]], 0
; CHECK:   ret { float } %[[MN2]]

; CHECK: define internal { float } @diffetester_fmax(float %x, float %differeturn)
; CHECK:   %[[MX0:.+]] = fcmp fast olt float %x, 5.000000e-01
; CHECK:   %[[MX1:.+]] = select fast i1 %[[MX0]], float 0.000000e+00, float %differeturn
; CHECK:   %[[MX2:.+]] = insertvalue { float } undef, float %[[MX1]], 0
; CHECK:   ret { float } %[[MX2]]

; CHECK: define internal { float } @diffetester_powi(float %x, i32 %n, float %differeturn)
; CHECK:   call fast float @__nv_powif(float %x, i32 %{{.+}})
; CHECK:   ret { float } %{{.+}}

; CHECK: define internal { float, float } @diffetester_rint_nearbyint(float %x, float %y, float %differeturn)
; CHECK:   %[[RI0:.+]] = insertvalue { float, float } undef, float %differeturn, 0
; CHECK:   %[[RI1:.+]] = insertvalue { float, float } %[[RI0]], float 0.000000e+00, 1
; CHECK:   ret { float, float } %[[RI1]]
