; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=discarded_vcat -o /dev/null | FileCheck %s --check-prefix=BRANCH; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=discarded_vcat -S -o /dev/null | FileCheck %s --check-prefix=BRANCH
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-activity-analysis -activity-analysis-func=single_block_vcat -o /dev/null | FileCheck %s --check-prefix=SINGLE; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=single_block_vcat -S -o /dev/null | FileCheck %s --check-prefix=SINGLE

declare noalias ptr @malloc(i64)
declare double @compute_active(double)
declare void @vcat(ptr) #0

; BRANCH: i1 %cond: icv:1
; BRANCH-NEXT: double %x: icv:0
; BRANCH-NEXT: entry
; BRANCH-NEXT:   br i1 %cond, label %active_branch, label %discard_branch: icv:1 ici:1
; BRANCH-NEXT: active_branch
; BRANCH-NEXT:   %active_result = call double @compute_active(double %x): icv:0 ici:0
; BRANCH-NEXT:   ret double %active_result: icv:1 ici:1
; BRANCH-NEXT: discard_branch
; BRANCH-NEXT:   %array = call noalias ptr @malloc(i64 16): icv:1 ici:1
; BRANCH-NEXT:   %outer = alloca { ptr, double }, align 8: icv:1 ici:1
; BRANCH-NEXT:   store ptr %array, ptr %outer, align 8: icv:1 ici:1
; BRANCH-NEXT:   %outer_value = getelementptr inbounds { ptr, double }, ptr %outer, i64 0, i32 1: icv:1 ici:1
; BRANCH-NEXT:   store double %x, ptr %outer_value, align 8: icv:1 ici:1
; BRANCH-NEXT:   %array_next = getelementptr inbounds ptr, ptr %array, i64 1: icv:1 ici:1
; BRANCH-NEXT:   store ptr %array_next, ptr %array, align 8: icv:1 ici:1
; BRANCH-NEXT:   call void @vcat(ptr %outer): icv:1 ici:1
; BRANCH-NEXT:   ret double %x: icv:1 ici:1
define double @discarded_vcat(i1 %cond, double %x) {
entry:
  br i1 %cond, label %active_branch, label %discard_branch

active_branch:
  %active_result = call double @compute_active(double %x)
  ret double %active_result

discard_branch:
  %array = call noalias ptr @malloc(i64 16)
  %outer = alloca { ptr, double }, align 8
  store ptr %array, ptr %outer, align 8
  %outer_value = getelementptr inbounds { ptr, double }, ptr %outer, i64 0, i32 1
  store double %x, ptr %outer_value, align 8
  %array_next = getelementptr inbounds ptr, ptr %array, i64 1
  store ptr %array_next, ptr %array, align 8
  call void @vcat(ptr %outer)
  ret double %x
}

; SINGLE: double %x: icv:0
; SINGLE-NEXT: entry
; SINGLE-NEXT:   %array = call noalias ptr @malloc(i64 16): icv:1 ici:1
; SINGLE-NEXT:   %outer = alloca { ptr, double }, align 8: icv:1 ici:1
; SINGLE-NEXT:   store ptr %array, ptr %outer, align 8: icv:1 ici:1
; SINGLE-NEXT:   %outer_value = getelementptr inbounds { ptr, double }, ptr %outer, i64 0, i32 1: icv:1 ici:1
; SINGLE-NEXT:   store double %x, ptr %outer_value, align 8: icv:1 ici:1
; SINGLE-NEXT:   %array_next = getelementptr inbounds ptr, ptr %array, i64 1: icv:1 ici:1
; SINGLE-NEXT:   store ptr %array_next, ptr %array, align 8: icv:1 ici:1
; SINGLE-NEXT:   call void @vcat(ptr %outer): icv:1 ici:1
; SINGLE-NEXT:   ret double %x: icv:1 ici:1
define double @single_block_vcat(double %x) {
entry:
  %array = call noalias ptr @malloc(i64 16)
  %outer = alloca { ptr, double }, align 8
  store ptr %array, ptr %outer, align 8
  %outer_value = getelementptr inbounds { ptr, double }, ptr %outer, i64 0, i32 1
  store double %x, ptr %outer_value, align 8
  %array_next = getelementptr inbounds ptr, ptr %array, i64 1
  store ptr %array_next, ptr %array, align 8
  call void @vcat(ptr %outer)
  ret double %x
}

attributes #0 = { "enzyme_LocalReadOnlyOrThrow" }
