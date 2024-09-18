; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

declare void @__enzyme_fwddiff(...)

define void @caller(double* %arg2, double* %arg3) {
  call void (...) @__enzyme_fwddiff(void (double, double*)* @julia_gradient_deferred__5305, metadata !"enzyme_dup", double 1.0, double 1.0, metadata !"enzyme_dupnoneed", double* %arg2, double* %arg3)
  ret void
}

define void @julia_gradient_deferred__5305(double %i95, double* %i346) {
  %i73 = call noalias nonnull i8* @malloc(i64 100)
  %i74 = bitcast i8* %i73 to double*
  store double %i95, double* %i74, align 8
  %i257 = load double, double* %i74, align 8
  store double %i257, double* %i346, align 8
  call void @free(i8* nonnull %i73)
  ret void
}

declare void @free(i8*)

declare i8* @malloc(i64)

; CHECK: define internal void @fwddiffejulia_gradient_deferred__5305(double %i95, double %"i95'", double* %i346, double* %"i346'")
; CHECK-NEXT:   %[[i1:.+]] = call noalias nonnull i8* @malloc(i64 100)
; CHECK-NEXT:   %"i74'ipc" = bitcast i8* %[[i1]] to double*
; CHECK-NEXT:   store double %"i95'", double* %"i74'ipc", align 8
; CHECK-NEXT:   %"i257'ipl" = load double, double* %"i74'ipc", align 8
; CHECK-NEXT:   store double %"i257'ipl", double* %"i346'"
; CHECK-NEXT:   call void @free(i8* nonnull %[[i1]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
