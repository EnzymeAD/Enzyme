; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

; An open run holds in the function that declares it, where every access stays
; within the actual argument, but not in its callers, where the same memory
; (here one descriptor temporary) is reused for data of another type.

define void @rd8(ptr "enzyme_type"="{[-1]:Pointer, [-1,0+1*?]:Float@double}" %a) {
entry:
  %v = load i64, ptr %a
  store i64 %v, ptr %a
  ret void
}

define void @rd4(ptr "enzyme_type"="{[-1]:Pointer, [-1,0+1*?]:Float@float}" %a) {
entry:
  %v = load i32, ptr %a
  store i32 %v, ptr %a
  ret void
}

define void @caller() {
entry:
  %tmp = alloca [64 x i8]
  call void @rd8(ptr %tmp)
  call void @rd4(ptr %tmp)
  ret void
}

; CHECK: rd8 - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: ptr %a: {[-1]:Pointer, [-1,0+1*?]:Float@double}
; CHECK: %v = load i64, ptr %a, align 4: {[-1]:Float@double}
; CHECK: rd4 - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: ptr %a: {[-1]:Pointer, [-1,0+1*?]:Float@float}
; CHECK: %v = load i32, ptr %a, align 4: {[-1]:Float@float}
; CHECK: caller - {} |
; CHECK: %tmp = alloca [64 x i8], align 1: {[-1]:Pointer}
