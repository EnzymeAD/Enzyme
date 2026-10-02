; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme-summary" -disable-output | FileCheck %s; fi

; Handing a global or an argument to another function is not a write of
; it by the caller: the thin-link step composes what the callee does with
; it from the callee's summary (the edges). Only the caller's own stores
; count, and the flang runtime's writes.

@grid_ = common global [64 x i8] zeroinitializer
@state_ = common global [64 x i8] zeroinitializer

declare void @use(ptr, ptr)

define void @f(ptr %a, ptr %b) {
  call void @use(ptr @grid_, ptr %a)
  store double 1.0, ptr @state_
  store double 2.0, ptr %b
  ret void
}

; CHECK: "args_write_any": [
; CHECK-NEXT: false,
; CHECK-NEXT: true
; CHECK: "edges": [
; CHECK-DAG: "ggrid_",
; CHECK-DAG: "a0",
; CHECK: "globals_write_any": [
; CHECK-NEXT: "state_"
; CHECK-NEXT: ]
