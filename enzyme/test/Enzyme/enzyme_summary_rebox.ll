; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme-summary" -disable-output | FileCheck %s; fi

; flang passes an assumed-shape array on in a new descriptor on the stack
; (a rebox) holding the base address loaded from the argument's descriptor.
; What the callee does with the local descriptor applies to the argument,
; and loads and stores through the base address loaded back from it are
; reads and writes of the argument; the stores into the local descriptor
; itself are not.

declare void @callee(ptr)

define void @f(ptr %desc) {
  %box = alloca { ptr, i64 }
  %base = load ptr, ptr %desc
  store ptr %base, ptr %box
  %len = getelementptr inbounds { ptr, i64 }, ptr %box, i32 0, i32 1
  store i64 8, ptr %len
  call void @callee(ptr %box)
  %b = load ptr, ptr %box
  %v = load double, ptr %b
  %m = fmul double %v, %v
  store double %m, ptr %b
  ret void
}

; CHECK: "args": [
; CHECK-NEXT: {
; CHECK-NEXT: "escape": false,
; CHECK-NEXT: "read": true,
; CHECK-NEXT: "write": true
; CHECK: "edges": [
; CHECK-NEXT: [
; CHECK-NEXT: "a0",
; CHECK-NEXT: "callee",
; CHECK-NEXT: 0
