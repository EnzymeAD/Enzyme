; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null -opaque-pointers | FileCheck %s

define void @caller(ptr %p) {
entry:
  %int = ptrtoint ptr %p to i64
  %add = add i64 %int, 63
  %and = and i64 %add, -64
  %ptr = inttoptr i64 %and to ptr
  store float 0.0, ptr %ptr, align 4
  ret void
}

; CHECK: caller - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: ptr %p: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %int = ptrtoint ptr %p to i64: {[-1]:Pointer}
; CHECK-NEXT:   %add = add i64 %int, 63: {[-1]:Pointer}
; CHECK-NEXT:   %and = and i64 %add, -64: {[-1]:Pointer}
; CHECK-NEXT:   %ptr = inttoptr i64 %and to ptr: {[-1]:Pointer}
; CHECK-NEXT:   store float 0.000000e+00, ptr %ptr, align 4: {}
; CHECK-NEXT:   ret void: {}
