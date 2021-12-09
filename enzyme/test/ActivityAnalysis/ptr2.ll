; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -activity-analysis-inactive-args -o /dev/null | FileCheck %s

source_filename = "/mnt/pci4/wmdata/Enzyme/enzyme/test/ActivityAnalysis/ptr2.ll"

; Function Attrs: noinline nosync readonly
define private fastcc double @julia_func_1396(double** noalias nocapture nonnull readonly %arg) #0 {
top:
  %i3 = load double*, double** %arg, align 8
  %i4 = load double, double* %i3, align 8
  ret double %i4
}

declare double** @jl_alloc()

define double @f(double %arg) {
entry:
  ; No active arguments
  ; => result comes from allocation or global
  %i2 = call noalias nonnull double** @jl_alloc()
  %i4 = load double*, double** %i2, align 8
  store double %arg, double* %i4, align 8
  %i5 = call fastcc double @julia_func_1396(double** noalias nocapture nonnull readonly %i2) #1
  %c = fadd double %i5, %arg
  ret double %c
}

; CHECK: double %arg: icv:1
; CHECK-NEXT: entry
; CHECK-NEXT:   %i2 = call noalias nonnull double** @jl_alloc(): icv:0 ici:1
; CHECK-NEXT:   %i4 = load double*, double** %i2, align 8: icv:0 ici:1
; CHECK-NEXT:   store double %arg, double* %i4, align 8: icv:1 ici:1
; CHECK-NEXT:   %i5 = call fastcc double @julia_func_1396(double** noalias nocapture nonnull readonly %i2) #1: icv:0 ici:0
; CHECK-NEXT:   %c = fadd double %i5, %arg: icv:0 ici:0
; CHECK-NEXT:   ret double %c: icv:1 ici:1

attributes #0 = { noinline nosync readonly }
attributes #1 = { readonly }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2, !5}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, nameTableKind: None)
!3 = !DIFile(filename: "/mnt/Data/git/Enzyme.jl/dr.jl", directory: ".")
!4 = !{}
!5 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !3, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, nameTableKind: None)
