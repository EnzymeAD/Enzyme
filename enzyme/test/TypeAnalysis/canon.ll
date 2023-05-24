; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

declare void @__enzyme_float(float* %x, i64)

define void @caller() {
entry:
  %ptr = alloca float*, align 8
  %ld = load float*, float** %ptr, align 8
  call void @__enzyme_float(float* %ld, i64 -1)
  ret void
}

; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %ptr = alloca float*, align 8: {[-1]:Pointer, [-1,-1]:Pointer, [-1,-1,-1]:Float@float}
; CHECK-NEXT:   %ld = load float*, float** %ptr, align 8: {[-1]:Pointer, [-1,-1]:Float@float}
; CHECK-NEXT:   call void @__enzyme_float(float* %ld, i64 -1): {}
; CHECK-NEXT:   ret void: {}
