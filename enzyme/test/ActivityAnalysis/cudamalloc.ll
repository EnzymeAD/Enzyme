; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=aa_cudaMalloc -S | FileCheck %s; fi

declare i32 @cudaMalloc(ptr, i64)
declare ptr @get_ptr(ptr) readnone

define void @aa_cudaMalloc(ptr %arr, ptr %pp) {
entry:
  %p2 = call ptr @get_ptr(ptr %arr)
  %v = load double, ptr %p2, align 8
  %a = call i32 @cudaMalloc(ptr %pp, i64 8)
  ret void
}

; CHECK: ptr %arr: icv:0
; CHECK: ptr %pp: icv:0
; CHECK: entry
; CHECK-NEXT:   %p2 = call ptr @get_ptr(ptr %arr): icv:1 ici:1
; CHECK-NEXT:   %v = load double, ptr %p2, align 8: icv:1 ici:1
; CHECK-NEXT:   %a = call i32 @cudaMalloc(ptr %pp, i64 8): icv:1 ici:0
; CHECK-NEXT:   ret void: icv:1 ici:1
