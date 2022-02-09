;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define float @wrapper(i32 %n, float* %x, i32 %incx) {
entry:
  %call = tail call float @cblas_snrm2(i32 %n, float* %x, i32 %incx)
  ret float %call
}

declare float @cblas_snrm2(i32, float*, i32)

define float @wrapperMod(i32 %n, float* %x, i32 %incx) {
entry:
  %call = tail call float @cblas_snrm2(i32 %n, float* %x, i32 %incx)
  store float 0.000000e+00, float* %x, align 4
  ret float %call
}

define void @active(i32 %n, float* %x, float* %_x, i32 %incx) {
entry:
  %call = tail call float (i8*, ...) @__enzyme_autodiff(i8* bitcast (float (i32, float*, i32)* @wrapper to i8*), i32 %n, float* %x, float* %_x, i32 %incx)
  ret void
}

declare float @__enzyme_autodiff(i8*, ...)

define void @activeMod(i32 %n, float* %x, float* %_x, i32 %incx) {
entry:
  %call = tail call float (i8*, ...) @__enzyme_autodiff(i8* bitcast (float (i32, float*, i32)* @wrapperMod to i8*), i32 %n, float* %x, float* %_x, i32 %incx)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[active:.+]](

;CHECK: define void @activeMod
;CHECK-NEXT: entry
;CHECK-NEXT: call void @[[activeMod:.+]](

;CHECK:define internal void @[[active]](i32 %n, float* %x, float* %"x'", i32 %incx, float %differeturn)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %call = tail call float @cblas_snrm2(i32 %n, float* %x, i32 %incx)
;CHECK-NEXT:  %0 = fdiv fast float %differeturn, %call
;CHECK-NEXT:  call void @cblas_saxpy(i32 %n, float %0, float* %x, i32 %incx, float* %"x'", i32 %incx)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:define internal void @[[activeMod]](i32 %n, float* %x, float* %"x'", i32 %incx, float %differeturn)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0stride(float* %1, float* %x, i32 %n, i32 %incx)
;CHECK-NEXT:  %call = tail call float @cblas_snrm2(i32 %n, float* %x, i32 %incx)
;CHECK-NEXT:  store float 0.000000e+00, float* %x, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"x'", align 4
;CHECK-NEXT:  %2 = fdiv fast float %differeturn, %call
;CHECK-NEXT:  call void @cblas_saxpy(i32 %n, float %2, float* %1, i32 1, float* %"x'", i32 1)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
